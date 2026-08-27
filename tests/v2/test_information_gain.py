"""
Unit tests for CineMind V2 Information Gain Calculation.
"""

import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.selection.information_gain import calculate_information_gain


def test_information_gain_perfect_splitter():
    # 4 entities with uniform posterior [0.25, 0.25, 0.25, 0.25]
    posterior_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    # Feature splits population exactly 50/50: [True, True, False, False]
    p_yes_matrix = np.array([
        [0.999, 0.999, 0.001, 0.001]
    ], dtype=np.float64)

    ig = calculate_information_gain(posterior_probs, p_yes_matrix)

    # Prior entropy H(E) = 2.0 bits.
    # After YES: 2 entities remaining -> H(E|YES) = 1.0 bit.
    # After NO: 2 entities remaining -> H(E|NO) = 1.0 bit.
    # Expected IG = 2.0 - (0.5 * 1.0 + 0.5 * 1.0) = 1.0 bit.
    assert math.isclose(ig[0], 1.0, abs_tol=0.03), f"Expected IG ~1.0 bit, got {ig[0]}"


def test_information_gain_uninformative_feature():
    # All entities have the same p_yes value -> 0 information gain
    posterior_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    p_yes_matrix = np.array([
        [0.5, 0.5, 0.5, 0.5]
    ], dtype=np.float64)

    ig = calculate_information_gain(posterior_probs, p_yes_matrix)
    assert math.isclose(ig[0], 0.0, abs_tol=1e-6), f"Expected IG 0.0, got {ig[0]}"


if __name__ == "__main__":
    test_information_gain_perfect_splitter()
    test_information_gain_uninformative_feature()
    print("All Information Gain tests passed successfully!")
