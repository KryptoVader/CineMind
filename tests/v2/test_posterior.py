"""
Unit tests for CineMind V2 Bayesian Posterior Tracker.
"""

import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.inference.posterior import BayesianPosterior
from cinemind_v2.questions.question import Question


def test_posterior_normalization_and_updates():
    ids = ["ent_1", "ent_2", "ent_3", "ent_4"]
    posterior = BayesianPosterior(num_entities=4, cinemind_ids=ids)

    # Initial uniform probabilities: [0.25, 0.25, 0.25, 0.25]
    probs = posterior.get_probs()
    assert np.allclose(probs, [0.25, 0.25, 0.25, 0.25])
    assert math.isclose(np.sum(probs), 1.0)
    assert posterior.get_rank("ent_1") == 1

    # Question where ent_1 and ent_2 are True, ent_3 and ent_4 are False
    feature_mask = np.array([True, True, False, False], dtype=bool)
    q = Question(id="q1", type="metadata", text="Is it a movie?", feature_id="q1", reliability=0.95)

    # 1. Update YES: ent_1 and ent_2 should gain probability, ent_3 and ent_4 should lose
    probs_yes = posterior.update(question=q, answer="YES", feature_mask=feature_mask)
    assert math.isclose(np.sum(probs_yes), 1.0)
    assert probs_yes[0] > probs_yes[2]
    assert probs_yes[1] > probs_yes[3]

    # 2. Reset and Update NO: ent_3 and ent_4 should gain probability
    posterior.reset()
    probs_no = posterior.update(question=q, answer="NO", feature_mask=feature_mask)
    assert math.isclose(np.sum(probs_no), 1.0)
    assert probs_no[2] > probs_no[0]
    assert probs_no[3] > probs_no[1]

    # 3. Reset and Update UNKNOWN: relative probabilities should remain unchanged
    posterior.reset()
    probs_unk = posterior.update(question=q, answer="UNKNOWN", feature_mask=feature_mask)
    assert math.isclose(np.sum(probs_unk), 1.0)
    assert np.allclose(probs_unk, [0.25, 0.25, 0.25, 0.25])


def test_target_rank_calculation():
    ids = ["ent_1", "ent_2", "ent_3", "ent_4"]
    posterior = BayesianPosterior(num_entities=4, cinemind_ids=ids)

    # Question that isolates ent_3 (ent_3 is True, others False)
    feature_mask = np.array([False, False, True, False], dtype=bool)
    q = Question(id="q_target", type="genre", text="Is it sci-fi?", feature_id="q_target", reliability=0.95)

    posterior.update(question=q, answer="YES", feature_mask=feature_mask)

    # ent_3 should now be rank 1
    assert posterior.get_rank("ent_3") == 1
    assert posterior.get_top_k(1)[0][0] == "ent_3"


if __name__ == "__main__":
    test_posterior_normalization_and_updates()
    test_target_rank_calculation()
    print("All posterior tests passed successfully!")
