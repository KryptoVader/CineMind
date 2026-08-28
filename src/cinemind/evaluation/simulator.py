"""
Offline Guessing Simulator for CineMind Milestone 2A & 2B.

Includes TargetSampler, RandomQuestionPolicy, FastQuestionEvaluator, EliminationEngine, SimulationResult, and GameSimulator.
Enforces strict target isolation: guessing procedure has zero knowledge or access to target entity.
Supports Random, EntropySplit, and ExpectedCandidateReduction policies with vectorized matrix scoring.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from cinemind.data.entity import Entity
from cinemind.evaluation.oracle import SimulationOracle
from cinemind.questions.catalog import QuestionCatalog
from cinemind.questions.schema import PlayerAnswer, Question


class TargetSampler:
    """Helper for target sampling strategies (uniform entity, popularity-weighted)."""

    @staticmethod
    def sample_uniform(entities: List[Entity], rng: np.random.RandomState) -> Entity:
        """Sample target entity uniformly at random."""
        idx = rng.randint(0, len(entities))
        return entities[idx]

    @staticmethod
    def sample_popularity_weighted(entities: List[Entity], rng: np.random.RandomState) -> Entity:
        """Sample target entity weighted by logarithmic popularity and vote count."""
        weights = []
        for e in entities:
            pop = float(e.get_feature("popularity") or 0.0)
            votes = float(e.get_feature("vote_count") or 0.0)
            weight = np.log1p(max(0.0, pop) + max(0.0, votes))
            weights.append(weight if weight > 0 else 0.001)

        weights_arr = np.array(weights, dtype=np.float64)
        sum_w = weights_arr.sum()
        if sum_w <= 0:
            probs = np.ones(len(entities)) / len(entities)
        else:
            probs = weights_arr / sum_w

        idx = rng.choice(len(entities), p=probs)
        return entities[idx]


class RandomQuestionPolicy:
    """Random question selection policy using random seed and existing catalog."""

    def __init__(self, catalog: QuestionCatalog, rng: np.random.RandomState) -> None:
        self.questions = catalog.list_questions()
        self.rng = rng

    def select_next_question(
        self,
        candidates: Optional[List[Entity]] = None,
        asked_question_ids: Optional[Set[str]] = None,
        evaluator: Optional[Any] = None,
        cand_indices: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Question], None]:
        """Return next random unused question from catalog."""
        asked = asked_question_ids or set()
        available = [q for q in self.questions if q.question_id not in asked]
        if not available:
            return None, None
        idx = self.rng.randint(0, len(available))
        return available[idx], None


class FastQuestionEvaluator:
    """Precomputed evaluation cache mapping (question_id, cinemind_id) -> PlayerAnswer."""

    def __init__(self, catalog: QuestionCatalog, entities: List[Entity]) -> None:
        self._cache: Dict[Tuple[str, str], PlayerAnswer] = {}
        for q in catalog.list_questions():
            for e in entities:
                self._cache[(q.question_id, e.cinemind_id)] = q.evaluate(e)

    def evaluate(self, question: Question, entity: Entity) -> PlayerAnswer:
        key = (question.question_id, entity.cinemind_id)
        if key in self._cache:
            return self._cache[key]
        ans = question.evaluate(entity)
        self._cache[key] = ans
        return ans


class EliminationEngine:
    """
    Deterministic Elimination Engine.
    Operates ONLY on candidate entities, question objects, and oracle answers.
    Has ZERO knowledge of the target entity.
    """

    def __init__(self, candidates: List[Entity], evaluator: Optional[Any] = None) -> None:
        self._all_candidates = list(candidates)
        self._evaluator = evaluator
        if evaluator and hasattr(evaluator, "e_id_to_idx"):
            self.cand_indices = np.array(
                [evaluator.e_id_to_idx[e.cinemind_id] for e in candidates if e.cinemind_id in evaluator.e_id_to_idx],
                dtype=np.int32,
            )
        else:
            self.cand_indices = None
            self._candidates = list(candidates)

    @property
    def candidate_count(self) -> int:
        if self.cand_indices is not None:
            return len(self.cand_indices)
        return len(self._candidates)

    @property
    def candidates(self) -> List[Entity]:
        if self.cand_indices is not None:
            return [self._all_candidates[idx] for idx in self.cand_indices]
        return list(self._candidates)

    def apply_answer(self, question: Question, answer: PlayerAnswer) -> None:
        """
        Filters candidates based on oracle answer:
        - YES: keep entities where question.evaluate(e) == YES
        - NO: keep entities where question.evaluate(e) == NO
        - UNKNOWN: keep ALL candidates (no elimination)
        """
        if answer == PlayerAnswer.UNKNOWN:
            # UNKNOWN answers MUST NEVER eliminate candidates
            return

        if self.cand_indices is not None and self._evaluator and hasattr(self._evaluator, "ans_matrix"):
            q_idx = self._evaluator.q_id_to_idx.get(question.question_id)
            if q_idx is not None:
                target_val = 1 if answer == PlayerAnswer.YES else 2
                sub_ans = self._evaluator.ans_matrix[q_idx, self.cand_indices]
                self.cand_indices = self.cand_indices[sub_ans == target_val]
                return

        filtered = []
        for e in self.candidates:
            if question.evaluate(e) == answer:
                filtered.append(e)

        self._candidates = filtered


@dataclass
class SimulationResult:
    target_entity_id: str
    target_title: str
    guessed_entity_id: Optional[str]
    guessed_title: Optional[str]
    correct: bool
    questions_asked: int
    remaining_candidates: int
    history: List[Dict[str, Any]]
    terminated_reason: str
    zero_candidate_info: Optional[Dict[str, Any]] = None
    policy_name: str = "random"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_entity_id": self.target_entity_id,
            "target_title": self.target_title,
            "guessed_entity_id": self.guessed_entity_id,
            "guessed_title": self.guessed_title,
            "correct": self.correct,
            "questions_asked": self.questions_asked,
            "remaining_candidates": self.remaining_candidates,
            "terminated_reason": self.terminated_reason,
            "policy_name": self.policy_name,
            "history": self.history,
            "zero_candidate_info": self.zero_candidate_info,
        }


class GameSimulator:
    """Coordinates simulation of a single game using strict target isolation."""

    def __init__(
        self,
        candidate_universe: List[Entity],
        question_catalog: QuestionCatalog,
        max_questions: int = 25,
        policy_type: str = "random",
        precompute_cache: bool = True,
    ) -> None:
        self.candidate_universe = candidate_universe
        self.question_catalog = question_catalog
        self.max_questions = max_questions
        self.policy_type = policy_type.lower()

        self._title_map: Dict[str, str] = {e.cinemind_id: e.title for e in candidate_universe}

        if precompute_cache:
            from cinemind.evaluation.adaptive_policy import VectorizedQuestionEvaluator
            self._evaluator = VectorizedQuestionEvaluator(catalog=question_catalog, entities=candidate_universe)
        else:
            self._evaluator = None

    def run_game(self, target_entity: Entity, seed: int = 42) -> SimulationResult:
        """Run a single simulated game with a hidden target entity."""
        rng = np.random.RandomState(seed)

        # 1. Instantiate Oracle with hidden target
        oracle = SimulationOracle(target_entity=target_entity)

        # 2. Instantiate Elimination Engine with full candidate universe (target hidden)
        engine = EliminationEngine(candidates=self.candidate_universe, evaluator=self._evaluator)

        # 3. Instantiate Policy
        if self.policy_type == "entropy":
            from cinemind.evaluation.adaptive_policy import EntropySplitPolicy
            policy = EntropySplitPolicy(catalog=self.question_catalog)
            policy_name = "EntropySplit"
        elif self.policy_type in ("expected_reduction", "reduction"):
            from cinemind.evaluation.adaptive_policy import ExpectedCandidateReductionPolicy
            policy = ExpectedCandidateReductionPolicy(catalog=self.question_catalog)
            policy_name = "ExpectedReduction"
        else:
            policy = RandomQuestionPolicy(catalog=self.question_catalog, rng=rng)
            policy_name = "Random"

        asked_question_ids: Set[str] = set()
        history: List[Dict[str, Any]] = []
        zero_candidate_info: Optional[Dict[str, Any]] = None
        terminated_reason = "max_questions_reached"

        step = 0
        while step < self.max_questions:
            if engine.candidate_count == 1:
                terminated_reason = "unique_candidate"
                break
            elif engine.candidate_count == 0:
                terminated_reason = "empty_candidate_set"
                break

            # Avoid triggering candidate entity list generation when cand_indices are present
            cand_list = None if engine.cand_indices is not None else engine.candidates

            # Select next question from policy using vectorized evaluator and active candidate indices
            question, split_score = policy.select_next_question(
                candidates=cand_list,
                asked_question_ids=asked_question_ids,
                evaluator=self._evaluator,
                cand_indices=engine.cand_indices,
            )

            if question is None:
                terminated_reason = "no_more_questions"
                break

            asked_question_ids.add(question.question_id)

            # Get answer from oracle (target isolated inside oracle)
            answer = oracle.answer(question)

            count_before = engine.candidate_count
            engine.apply_answer(question, answer)
            count_after = engine.candidate_count

            step += 1

            # Format step history entry with policy diagnostics
            h_entry: Dict[str, Any] = {
                "step": step,
                "question_id": question.question_id,
                "text": question.text,
                "question_family": question.question_family,
                "oracle_answer": answer.value,
                "remaining_candidates": count_after,
            }

            if split_score:
                h_entry["score"] = split_score.score
                h_entry["yes_fraction"] = split_score.yes_fraction
                h_entry["no_fraction"] = split_score.no_fraction
                h_entry["unknown_fraction"] = split_score.unknown_fraction
                h_entry["expected_remaining"] = split_score.expected_remaining_candidates
                h_entry["expected_reduction_ratio"] = split_score.expected_reduction_ratio

            history.append(h_entry)

            if count_after == 0 and zero_candidate_info is None:
                zero_candidate_info = {
                    "zero_candidate_step": step,
                    "zero_candidate_question_id": question.question_id,
                    "zero_candidate_question_text": question.text,
                    "zero_candidate_answer": answer.value,
                    "candidate_count_before": count_before,
                    "candidate_count_after": 0,
                }
                terminated_reason = "empty_candidate_set"
                break

        guessed_id: Optional[str] = None
        if engine.candidate_count > 0:
            guessed_id = engine.candidates[0].cinemind_id
            if engine.candidate_count == 1 and terminated_reason == "max_questions_reached":
                terminated_reason = "unique_candidate"

        correct = (guessed_id == oracle.target_id) if guessed_id else False

        target_title = self._title_map.get(oracle.target_id, oracle.target_id)
        guessed_title = self._title_map.get(guessed_id, guessed_id) if guessed_id else None

        return SimulationResult(
            target_entity_id=oracle.target_id,
            target_title=target_title,
            guessed_entity_id=guessed_id,
            guessed_title=guessed_title,
            correct=correct,
            questions_asked=len(history),
            remaining_candidates=engine.candidate_count,
            history=history,
            terminated_reason=terminated_reason,
            zero_candidate_info=zero_candidate_info,
            policy_name=policy_name,
        )
