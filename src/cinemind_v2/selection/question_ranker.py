"""
CineMind V2 — Question Ranker
Scores and ranks candidate questions using Information Gain.
"""

import numpy as np
from cinemind_v2.questions.question import Question
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.inference.posterior import BayesianPosterior
from cinemind_v2.selection.information_gain import calculate_information_gain


class QuestionRanker:
    """
    Ranks candidate questions using dynamic Information Gain calculated on the current posterior.
    """

    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store

    def rank_questions(
        self,
        questions: list[Question],
        posterior: BayesianPosterior,
        asked_ids: set[str],
        top_n: int = 10,
    ) -> list[tuple[Question, float]]:
        """
        Rank candidate questions by expected Information Gain over current posterior.
        Returns list of (Question, ig_score) tuples sorted by IG descending.
        """
        # Filter out already asked questions
        candidates = [q for q in questions if q.id not in asked_ids and q.feature_id not in asked_ids]
        if not candidates:
            return []

        posterior_probs = posterior.get_probs()
        num_candidates = len(candidates)
        num_entities = len(posterior_probs)

        # Build p_yes matrix for candidate questions
        p_yes_matrix = np.zeros((num_candidates, num_entities), dtype=np.float64)

        for i, q in enumerate(candidates):
            mask = self.feature_store.get_feature_values(q.feature_id)
            unknown_mask = self.feature_store.get_unknown_values(q.feature_id)
            p_yes_vec = posterior.compute_likelihood_vector(
                feature_mask=mask,
                unknown_mask=unknown_mask,
                reliability=q.reliability,
            )
            p_yes_matrix[i] = p_yes_vec

        # Calculate IG scores for all candidates
        ig_scores = calculate_information_gain(
            posterior_probs=posterior_probs,
            p_yes_matrix=p_yes_matrix,
        )

        ranked_indices = np.argsort(ig_scores)[::-1]

        ranked_results = [
            (candidates[idx], float(ig_scores[idx]))
            for idx in ranked_indices[:top_n]
        ]
        return ranked_results
