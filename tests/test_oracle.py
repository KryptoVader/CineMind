"""
Unit tests for SimulationOracle.
"""

import pytest
from cinemind.data.entity import Entity
from cinemind.evaluation.oracle import SimulationOracle
from cinemind.questions.schema import Operator, PlayerAnswer, Question


def test_oracle_answers_known_true_false_and_unknown():
    entity_data = {
        "cinemind_id": "cm_target_1",
        "title": "Interstellar",
        "media_type": "movie",
        "genres": ["Science Fiction", "Drama"],
        "release_year": 2014.0,
        "runtime": 169.0,
        # missing original_language
    }
    target = Entity(entity_data)
    oracle = SimulationOracle(target)

    assert oracle.target_id == "cm_target_1"
    assert oracle.target_title == "Interstellar"

    # Known TRUE
    q_scifi = Question.create("Is it a science fiction title?", "genres", Operator.CONTAINS, "Science Fiction", "genre")
    assert oracle.answer(q_scifi) == PlayerAnswer.YES

    # Known FALSE
    q_comedy = Question.create("Is it a comedy?", "genres", Operator.CONTAINS, "Comedy", "genre")
    assert oracle.answer(q_comedy) == PlayerAnswer.NO

    # Missing metadata -> UNKNOWN
    q_lang = Question.create("Is the language Japanese?", "original_language", Operator.EQUALS, "ja", "language")
    assert oracle.answer(q_lang) == PlayerAnswer.UNKNOWN
    assert oracle.answer(q_lang) != PlayerAnswer.NO
