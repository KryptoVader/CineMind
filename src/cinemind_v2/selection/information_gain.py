"""
CineMind V2 — Vectorized Information Gain
Computes expected Shannon Information Gain IG(Q) across candidate questions.
"""

import math
from typing import Optional
import numpy as np
from cinemind_v2.selection.entropy import calculate_entropy
from cinemind_v2.questions.question import Question
from cinemind_v2.inference.posterior import BayesianPosterior


def calculate_information_gain(
    posterior_probs: np.ndarray,
    p_yes_matrix: np.ndarray,  # shape (num_questions, num_entities)
    include_unknown: bool = False,
) -> np.ndarray:
    """
    Vectorized computation of Shannon Information Gain for all questions simultaneously.

    IG(Q) = H(E) - E_a[H(E | a, Q)]

    Returns array of IG scores in bits of shape (num_questions,).
    """
    num_qs, num_entities = p_yes_matrix.shape

    if num_entities == 0 or num_qs == 0:
        return np.zeros(num_qs, dtype=np.float64)

    # Shannon Entropy of current posterior H(E)
    h_e = calculate_entropy(posterior_probs)

    # Clip p_yes to soft bounds
    p_yes_matrix = np.clip(p_yes_matrix, 1e-4, 1.0 - 1e-4)
    p_no_matrix = 1.0 - p_yes_matrix

    # Marginal probabilities P(YES) and P(NO) for each question across posterior distribution
    # shape (num_questions,)
    p_y = p_yes_matrix @ posterior_probs
    p_n = p_no_matrix @ posterior_probs

    ig_scores = np.zeros(num_qs, dtype=np.float64)

    # Vectorized post-answer posterior calculation
    for i in range(num_qs):
        if p_y[i] <= 1e-12 or p_n[i] <= 1e-12:
            ig_scores[i] = 0.0
            continue

        # Posterior given YES
        unnorm_y = posterior_probs * p_yes_matrix[i]
        post_y = unnorm_y / p_y[i]
        h_y = calculate_entropy(post_y)

        # Posterior given NO
        unnorm_n = posterior_probs * p_no_matrix[i]
        post_n = unnorm_n / p_n[i]
        h_n = calculate_entropy(post_n)

        if include_unknown:
            # P(UNKNOWN) is uniform 0.5 across entities, so posterior given UNKNOWN equals current posterior!
            # h_u = h_e
            # Assume 5% chance user answers UNKNOWN
            p_u = 0.05
            p_y_w = (1.0 - p_u) * p_y[i]
            p_n_w = (1.0 - p_u) * p_n[i]
            exp_h = p_y_w * h_y + p_n_w * h_n + p_u * h_e
        else:
            exp_h = p_y[i] * h_y + p_n[i] * h_n

        ig_scores[i] = max(0.0, h_e - exp_h)

    return ig_scores
