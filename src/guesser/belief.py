"""
CineMind Belief Tracker (Phase 2).

Manages Bayesian belief distribution P(e_i | history) with empirical likelihoods:
- Metadata likelihoods: 0.80 / 0.20 (soft Laplace smoothing)
- Concept Cluster likelihoods: sigmoid(scale * (sum_tfidf - threshold))
- Contrastive Keyword likelihoods: sigmoid(scale * (tfidf_iw - threshold))

Tracks entropy H(P) and top-candidate dominance margins for convergence control.
"""

import math
from typing import Any, Optional
import numpy as np
import pandas as pd

from guesser.knowledge import KnowledgeBase
from guesser.feedback import get_calibrated_likelihood_vector


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class BeliefTracker:
    """Bayesian Belief Tracker with Empirical Likelihoods & Feedback Recalibration."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.num_entities = kb.num_entities

        # Compute log popularity priors P(e_i)
        votes = kb.df["vote_count"].fillna(0).values.astype(np.float64)
        users = kb.df["num_list_users"].fillna(0).values.astype(np.float64)
        pops = kb.df["popularity"].fillna(0).values.astype(np.float64)

        raw_weights = np.log1p(votes + users + pops + 1.0)
        priors = raw_weights / np.sum(raw_weights)
        self.log_priors = np.log(priors + 1e-12)

        # Active game state
        self.log_posterior = self.log_priors.copy()
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset belief distribution for a new game."""
        self.log_posterior = self.log_priors.copy()
        self.history.clear()

    def get_posterior_probs(self) -> np.ndarray:
        """Normalized posterior probabilities P(e_i | history)."""
        max_log = np.max(self.log_posterior)
        shifted = np.exp(self.log_posterior - max_log)
        total = np.sum(shifted)
        if total > 0:
            return shifted / total
        return np.ones(self.num_entities, dtype=np.float64) / self.num_entities

    def get_entropy(self) -> float:
        """Compute current Shannon entropy H(P) in bits."""
        probs = self.get_posterior_probs()
        nonzero = probs[probs > 1e-12]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def get_empirical_p_yes_metadata(self, meta_feature_idx: int) -> np.ndarray:
        """
        Empirical P(YES | e_i) for metadata features with per-category reliability:
        - 0.90 / 0.10 for high-reliability fields (media_type, language, decade, origin_country)
        - 0.80 / 0.20 for fuzzier semantic genres / ratings / source / episodes
        """
        m_def = self.kb.meta_defs[meta_feature_idx]
        matches = self.kb.meta_matrix[meta_feature_idx]

        high_rel_categories = {"media_type", "language", "decade", "origin_country"}
        if m_def.category in high_rel_categories:
            p_yes_prior = np.where(matches, 0.90, 0.10)
        else:
            p_yes_prior = np.where(matches, 0.80, 0.20)

        # Wire in feedback recalibration
        return get_calibrated_likelihood_vector(self.kb.df["cinemind_id"], m_def.feature_id, p_yes_prior, k=5.0)

    def get_empirical_p_yes_concept(self, concept_idx: int, scale: float = 8.0) -> np.ndarray:
        """
        Empirical P(YES | e_i) for a TF-IDF concept cluster using 75th percentile threshold
        of nonzero TF-IDF sums so 'yes' corresponds to strong matches.
        """
        q_id = f"concept_cluster_{concept_idx}"
        sums = self.kb.concept_sums_matrix[concept_idx]
        nonzero_sums = sums[sums > 0]

        # Requirement 4: 75th percentile of nonzero TF-IDF sums
        if len(nonzero_sums) > 0:
            threshold = float(np.percentile(nonzero_sums, 75))
        else:
            threshold = 0.05

        prior_vec = sigmoid(scale * (sums - threshold))

        # Wire in feedback recalibration
        return get_calibrated_likelihood_vector(self.kb.df["cinemind_id"], q_id, prior_vec, k=5.0)

    def get_empirical_p_yes_word(self, word: str, scale: float = 8.0, threshold: float = 0.05) -> np.ndarray:
        """Empirical P(YES | e_i) for a contrastive word via TF-IDF relevance with feedback recalibration."""
        q_id = f"contrastive_{word}"
        if word not in self.kb.word_to_idx:
            prior_vec = np.full(self.num_entities, 0.20)
        else:
            w_idx = self.kb.word_to_idx[word]
            tfidf_col = self.kb.tfidf_sparse[:, w_idx].toarray().ravel()
            prior_vec = sigmoid(scale * (tfidf_col - threshold))

        return get_calibrated_likelihood_vector(self.kb.df["cinemind_id"], q_id, prior_vec, k=5.0)

    def update_belief(self, p_yes_vec: np.ndarray, answer: str, q_id: str) -> None:
        """Update Bayesian log-posterior vector given user answer."""
        answer = answer.lower().strip()
        p_yes_vec = np.clip(p_yes_vec, 1e-4, 1.0 - 1e-4)
        p_no_vec = 1.0 - p_yes_vec

        if answer in ["y", "yes"]:
            self.log_posterior += np.log(p_yes_vec)
        elif answer in ["n", "no"]:
            self.log_posterior += np.log(p_no_vec)
        elif answer in ["probably", "py"]:
            self.log_posterior += 0.5 * np.log(p_yes_vec)
        elif answer in ["probably_not", "pn"]:
            self.log_posterior += 0.5 * np.log(p_no_vec)
        # "dont_know" / "k" -> 0 log update (neutral marginalization)

        self.history.append({"q_id": q_id, "answer": answer})

    def get_top_candidates(self, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Get Top-K candidate entities and their posterior probabilities."""
        probs = self.get_posterior_probs()
        top_idx = np.argpartition(probs, -k)[-k:]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
        return [(self.kb.df.iloc[i].to_dict(), float(probs[i])) for i in top_idx]

    def check_convergence(self, max_questions: int = 30) -> bool:
        """
        Convergence Controller:
        Requirement 2: Stop if:
          Primary:   P(top) >= 0.35 AND Top/RunnerUp Margin >= 2.5
          Secondary: P(top) >= 0.15 AND Top/RunnerUp Margin >= 5.0 (adaptive early exit)
        OR
          Question budget exhausted
        """
        top_k = self.get_top_candidates(k=2)
        top_p = top_k[0][1]
        runner_p = top_k[1][1] if len(top_k) > 1 else 1e-6
        margin = top_p / runner_p

        # Primary condition
        if top_p >= 0.35 and margin >= 2.5:
            return True

        # Requirement 2: Secondary adaptive early exit
        if top_p >= 0.15 and margin >= 5.0:
            return True

        if len(self.history) >= max_questions:
            return True

        return False

