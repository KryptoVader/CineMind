"""
Question Quality Profiler for CineMind.

Computes offline quality profiles for candidate questions across entity populations:
- Coverage: Proportion of entities with valid, non-missing evidence.
- Balance: Metric evaluating how strongly the question splits the population (avoiding 99.9% vs 0.1%).
- Missingness: Proportion of entities returning UNKNOWN/NOT_APPLICABLE.
- Reliability: Initial metadata/rule-based reliability confidence.
- Answerability: Estimated human answerability score.
- Redundancy: Clustering duplicate or near-duplicate predicate response vectors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from cinemind.data.entity import Entity
from cinemind.questions.schema import PlayerAnswer, Question


@dataclass
class QualityProfile:
    coverage: float
    balance: float
    missingness: float
    reliability: float
    answerability: float
    redundancy_group: Optional[str] = None

    @property
    def quality_score(self) -> float:
        """Composite quality score combining balance, coverage, reliability, and answerability."""
        return float(self.balance * 0.4 + self.coverage * 0.3 + self.reliability * 0.15 + self.answerability * 0.15)


class QuestionQualityProfiler:
    """Offline profiler for candidate questions against a canonical entity population."""

    def __init__(self, entities: List[Entity]) -> None:
        self.entities = entities
        self.num_entities = len(entities)

    def profile_question(self, question: Question) -> QualityProfile:
        """Compute metrics for a single question against the entity population."""
        if self.num_entities == 0:
            profile = QualityProfile(coverage=0.0, balance=0.0, missingness=1.0, reliability=question.reliability, answerability=question.answerability)
            question.coverage = profile.coverage
            question.balance = profile.balance
            question.missingness = profile.missingness
            return profile

        yes_count = 0
        no_count = 0
        unknown_count = 0

        for entity in self.entities:
            ans = question.evaluate(entity)
            if ans == PlayerAnswer.YES:
                yes_count += 1
            elif ans == PlayerAnswer.NO:
                no_count += 1
            else:
                unknown_count += 1

        coverage = (yes_count + no_count) / self.num_entities
        missingness = unknown_count / self.num_entities

        # Compute balance split ratio
        if yes_count == 0 or no_count == 0:
            balance = 0.0
        else:
            balance = min(yes_count, no_count) / max(yes_count, no_count)

        profile = QualityProfile(
            coverage=round(coverage, 4),
            balance=round(balance, 4),
            missingness=round(missingness, 4),
            reliability=round(question.reliability, 4),
            answerability=round(question.answerability, 4),
        )

        # Update question object metadata in place
        question.coverage = profile.coverage
        question.balance = profile.balance
        question.missingness = profile.missingness

        return profile

    def profile_all(
        self,
        questions: List[Question],
        min_coverage: float = 0.01,
        min_balance: float = 0.001,
    ) -> List[Question]:
        """Profile all candidate questions and return those satisfying minimum quality thresholds."""
        qualified_questions: List[Question] = []
        for q in questions:
            profile = self.profile_question(q)
            if profile.coverage >= min_coverage and profile.balance >= min_balance:
                qualified_questions.append(q)

        # Detect redundancy among qualified questions
        self.detect_redundancy(qualified_questions)

        return qualified_questions

    def detect_redundancy(self, questions: List[Question]) -> None:
        """
        Computes response vectors across entities to tag duplicate questions with a redundancy_group.
        """
        if not questions or self.num_entities == 0:
            return

        # Build binary/trinary response matrix for sample entities
        sample_size = min(200, self.num_entities)
        sample_entities = self.entities[:sample_size]

        group_map: Dict[Tuple[int, ...], str] = {}
        group_counter = 0

        for q in questions:
            response_tuple = tuple(q.evaluate(e).value for e in sample_entities)
            if response_tuple in group_map:
                q.redundancy_group = group_map[response_tuple]
            else:
                group_counter += 1
                group_name = f"rg_{group_counter:03d}"
                group_map[response_tuple] = group_name
                q.redundancy_group = group_name
