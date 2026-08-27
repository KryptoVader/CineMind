"""
Unit tests for CineMind V2 Question Generation and Validation.
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.questions.question import Question
from cinemind_v2.questions.validator import validate_question
from cinemind_v2.questions.generator import QuestionGenerator


def test_question_validation_filters():
    # Valid question
    q_valid = Question(
        id="q1", type="genre", text="Is it action?", feature_id="f1",
        coverage=0.8, yes_rate=0.4, unknown_rate=0.1
    )
    assert validate_question(q_valid) is True

    # Nearly always yes -> invalid
    q_always_yes = Question(
        id="q2", type="genre", text="Is it something?", feature_id="f2",
        coverage=1.0, yes_rate=0.99, unknown_rate=0.0
    )
    assert validate_question(q_always_yes) is False

    # Nearly always no -> invalid
    q_always_no = Question(
        id="q3", type="genre", text="Is it rare?", feature_id="f3",
        coverage=1.0, yes_rate=0.00001, unknown_rate=0.0
    )
    assert validate_question(q_always_no) is False


def test_question_generator():
    data = {
        "cinemind_id": [f"id_{i}" for i in range(100)],
        "title": [f"Movie {i}" for i in range(100)],
        "media_type": ["movie"] * 50 + ["tv"] * 50,
        "release_year": [2015] * 50 + [2021] * 50,
        "original_language": ["en"] * 70 + ["ja"] * 30,
        "rating": [8.0] * 100,
        "genres": [["Action"]] * 40 + [["Drama"]] * 60,
    }
    df = pd.DataFrame(data)

    store = EntityStore(df)
    fs = FeatureStore(store, df)
    gen = QuestionGenerator(fs)

    questions = gen.generate_all_questions()
    q_ids = [q.id for q in questions]

    assert "media:movie" in q_ids
    assert "media:tv" in q_ids
    assert "language:english" in q_ids
    assert "language:japanese" in q_ids


if __name__ == "__main__":
    test_question_validation_filters()
    test_question_generator()
    print("All question generator tests passed successfully!")
