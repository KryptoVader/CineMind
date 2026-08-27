"""
CineMind V2 — Bayesian Posterior Tracker
Manages belief distribution over entities in log-space with soft likelihood updates.
"""

import math
from typing import Optional, Union, Any
import numpy as np
from scipy.special import logsumexp
from cinemind_v2.questions.question import Question


class BayesianPosterior:
    """
    Bayesian belief engine operating purely in log-probability space.
    Maintains P(entity) across all canonical entities.
    """

    def __init__(
        self,
        num_entities: int,
        cinemind_ids: list[str],
        prior_probs: Optional[np.ndarray] = None,
        p_yes_if_true: float = 0.95,
        p_yes_if_false: float = 0.05,
        p_yes_if_unknown: float = 0.50,
        floor: float = 1e-4,
        ceiling: float = 1.0 - 1e-4,
    ):
        self.num_entities = num_entities
        self.cinemind_ids = [str(cid) for cid in cinemind_ids]
        self._id_to_idx = {cid: idx for idx, cid in enumerate(self.cinemind_ids)}

        if len(self.cinemind_ids) != self.num_entities:
            raise ValueError("Length of cinemind_ids does not match num_entities")

        self.p_yes_if_true = p_yes_if_true
        self.p_yes_if_false = p_yes_if_false
        self.p_yes_if_unknown = p_yes_if_unknown
        self.floor = floor
        self.ceiling = ceiling

        if prior_probs is not None:
            if len(prior_probs) != self.num_entities:
                raise ValueError("Prior probabilities array length mismatch")
            priors = np.clip(prior_probs, 1e-12, None)
            priors = priors / np.sum(priors)
            self.log_priors = np.log(priors)
        else:
            # Uniform prior
            self.log_priors = np.full(self.num_entities, -math.log(self.num_entities), dtype=np.float64)

        self.log_posterior = self.log_priors.copy()
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset belief distribution to prior state."""
        self.log_posterior = self.log_priors.copy()
        self.history.clear()

    def get_probs(self) -> np.ndarray:
        """Compute normalized posterior probability vector P(e_i)."""
        log_norm = self.log_posterior - logsumexp(self.log_posterior)
        return np.exp(log_norm)

    def get_entropy(self) -> float:
        """Compute Shannon entropy H(E) in bits."""
        probs = self.get_probs()
        nonzero = probs[probs > 1e-15]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def compute_likelihood_vector(
        self,
        feature_mask: np.ndarray,
        unknown_mask: Optional[np.ndarray] = None,
        reliability: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute P(YES | e_i, Q) vector for every entity given feature mask.
        Supports soft reliability override.
        """
        p_true = reliability if reliability is not None else self.p_yes_if_true
        p_false = (1.0 - reliability) if reliability is not None else self.p_yes_if_false

        p_yes = np.where(feature_mask, p_true, p_false)

        if unknown_mask is not None:
            p_yes = np.where(unknown_mask, self.p_yes_if_unknown, p_yes)

        return np.clip(p_yes, self.floor, self.ceiling)

    def update(
        self,
        question: Question,
        answer: str,
        feature_mask: np.ndarray,
        unknown_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform Bayesian log-posterior update given answer ("YES", "NO", or "UNKNOWN").
        """
        answer_upper = answer.upper().strip()
        if answer_upper not in {"YES", "NO", "UNKNOWN"}:
            raise ValueError(f"Invalid answer '{answer}'. Must be 'YES', 'NO', or 'UNKNOWN'")

        p_yes_vec = self.compute_likelihood_vector(
            feature_mask=feature_mask,
            unknown_mask=unknown_mask,
            reliability=question.reliability,
        )

        if answer_upper == "YES":
            l_vec = p_yes_vec
        elif answer_upper == "NO":
            l_vec = 1.0 - p_yes_vec
        else:  # UNKNOWN
            l_vec = np.full(self.num_entities, 0.50, dtype=np.float64)

        # Soft floor/ceiling clipping on likelihoods
        l_vec = np.clip(l_vec, self.floor, self.ceiling)
        log_l_vec = np.log(l_vec)

        # Update log-posterior
        self.log_posterior += log_l_vec

        # Normalize with logsumexp
        self.log_posterior -= logsumexp(self.log_posterior)

        # Record history
        self.history.append({
            "question_id": question.id,
            "question_text": question.text,
            "answer": answer_upper,
            "entropy_after": self.get_entropy(),
            "top_candidate": self.get_top_k(1)[0],
        })

        return self.get_probs()

    def get_top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Returns list of (cinemind_id, probability) for top-k candidates."""
        probs = self.get_probs()
        k_actual = min(k, self.num_entities)
        top_indices = np.argpartition(probs, -k_actual)[-k_actual:]
        sorted_top = top_indices[np.argsort(probs[top_indices])[::-1]]
        return [(self.cinemind_ids[i], float(probs[i])) for i in sorted_top]

    def get_rank(self, cinemind_id: str) -> int:
        """Returns 1-based rank of target entity in current posterior."""
        idx = self._id_to_idx.get(str(cinemind_id))
        if idx is None:
            raise KeyError(f"Entity ID '{cinemind_id}' not found")

        probs = self.get_probs()
        target_p = probs[idx]
        # Rank is 1 + count of entities with strictly greater probability
        rank = int(np.sum(probs > target_p)) + 1
        return rank
