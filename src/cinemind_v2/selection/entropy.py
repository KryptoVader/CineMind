"""
CineMind V2 — Shannon Entropy Calculation
Pure vectorized entropy computation.
"""

import numpy as np


def calculate_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy H(P) in bits for probability vector.
    H(P) = - sum p_i log2(p_i)
    """
    if len(probs) == 0:
        return 0.0

    nonzero = probs[probs > 1e-15]
    if len(nonzero) == 0:
        return 0.0

    return float(-np.sum(nonzero * np.log2(nonzero)))
