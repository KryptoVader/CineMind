"""
Unit tests for CineMind V2 Entropy Calculation.
"""

import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.selection.entropy import calculate_entropy


def test_entropy_uniform_distribution():
    # 4 entities with uniform probability 0.25 -> H = log2(4) = 2.0 bits
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    h = calculate_entropy(probs)
    assert math.isclose(h, 2.0, abs_tol=1e-6), f"Expected 2.0 bits, got {h}"


def test_entropy_certainty():
    # 1 entity with probability 1.0 -> H = 0.0 bits
    probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    h = calculate_entropy(probs)
    assert math.isclose(h, 0.0, abs_tol=1e-6), f"Expected 0.0 bits, got {h}"


def test_entropy_binary_split():
    # 2 entities with probability 0.5 each -> H = 1.0 bit
    probs = np.array([0.5, 0.5], dtype=np.float64)
    h = calculate_entropy(probs)
    assert math.isclose(h, 1.0, abs_tol=1e-6), f"Expected 1.0 bit, got {h}"


if __name__ == "__main__":
    test_entropy_uniform_distribution()
    test_entropy_certainty()
    test_entropy_binary_split()
    print("All entropy tests passed successfully!")
