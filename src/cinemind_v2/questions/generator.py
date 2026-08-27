"""
CineMind V2 — Question Generator
Generates question library from FeatureStore and computes population statistics.
"""

import math
import numpy as np
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.questions.question import Question
from cinemind_v2.questions.templates import format_question_text
from cinemind_v2.questions.validator import validate_question


class QuestionGenerator:
    """
    Generates, decorates with population stats, and filters candidate questions from a FeatureStore.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        min_coverage: float = 0.0005,
        max_yes_rate: float = 0.98,
        min_yes_rate: float = 0.001,
        max_unknown_rate: float = 0.50,
    ):
        self.feature_store = feature_store
        self.min_coverage = min_coverage
        self.max_yes_rate = max_yes_rate
        self.min_yes_rate = min_yes_rate
        self.max_unknown_rate = max_unknown_rate

    def generate_all_questions(self) -> list[Question]:
        """Generates and filters all valid questions from the feature store."""
        questions: list[Question] = []
        seen_ids: set[str] = set()

        for fid in self.feature_store.feature_ids:
            if fid in seen_ids:
                continue

            meta = self.feature_store.get_feature_metadata(fid)
            mask = self.feature_store.get_feature_values(fid)
            unknown_mask = self.feature_store.get_unknown_values(fid)

            num_entities = len(mask)
            if num_entities == 0:
                continue

            num_yes = int(np.sum(mask))
            num_unknown = int(np.sum(unknown_mask))

            yes_rate = num_yes / num_entities
            unknown_rate = num_unknown / num_entities
            coverage = (num_entities - num_unknown) / num_entities

            # Compute binary Shannon entropy of YES/NO split
            p_y = yes_rate
            p_n = 1.0 - p_y
            if p_y > 1e-12 and p_n > 1e-12:
                split_entropy = - (p_y * math.log2(p_y) + p_n * math.log2(p_n))
            else:
                split_entropy = 0.0

            category = meta.get("category", "structured")
            desc = meta.get("description", "")
            text = format_question_text(category=category, key=fid, default_desc=desc)

            # Assign reliability based on category
            if category in {"media_type", "language", "decade", "origin_country"}:
                reliability = 0.95
            else:
                reliability = 0.85

            q = Question(
                id=fid,
                type=category,
                text=text,
                feature_id=fid,
                reliability=reliability,
                coverage=coverage,
                yes_rate=yes_rate,
                unknown_rate=unknown_rate,
                entropy=split_entropy,
                source="structured_metadata",
            )

            if validate_question(
                q,
                min_coverage=self.min_coverage,
                max_yes_rate=self.max_yes_rate,
                min_yes_rate=self.min_yes_rate,
                max_unknown_rate=self.max_unknown_rate,
            ):
                questions.append(q)
                seen_ids.add(fid)

        return questions
