"""
Unit tests for Entity domain model.
"""

import pytest
from cinemind.data.entity import Entity
from cinemind.data.schemas import FeatureState
from cinemind.questions.schema import Operator, PlayerAnswer, Question


def test_entity_creation_and_feature_access():
    raw_data = {
        "cinemind_id": "cm_123",
        "title": "Spirited Away",
        "media_type": "anime",
        "genres": '["Fantasy", "Adventure"]',
        "release_year": 2001.0,
        "runtime": 125.0,
        "original_language": "ja",
        "origin_country": '["JP"]',
    }
    entity = Entity(raw_data)

    assert entity.cinemind_id == "cm_123"
    assert entity.title == "Spirited Away"
    assert entity.media_type == "anime"

    # Verify JSON string auto-parsing
    genres = entity.get_feature("genres")
    assert isinstance(genres, list)
    assert "Fantasy" in genres
    assert "Adventure" in genres

    assert entity.get_feature("release_year") == 2001.0
    assert entity.get_feature("original_language") == "ja"


def test_entity_question_evaluation():
    raw_data = {
        "cinemind_id": "cm_456",
        "title": "Inception",
        "media_type": "movie",
        "genres": '["Action", "Science Fiction"]',
        "release_year": 2010.0,
        "runtime": 148.0,
        "original_language": "en",
    }
    entity = Entity(raw_data)

    # Genre question
    q_genre = Question.create("Is it an action title?", "genres", Operator.CONTAINS, "Action", "genre")
    assert q_genre.evaluate(entity) == PlayerAnswer.YES

    q_horror = Question.create("Is it a horror title?", "genres", Operator.CONTAINS, "Horror", "genre")
    assert q_horror.evaluate(entity) == PlayerAnswer.NO

    # Time question
    q_time = Question.create("Was it released after 2000?", "release_year", Operator.GREATER_THAN, 2000, "time")
    assert q_time.evaluate(entity) == PlayerAnswer.YES

    q_old = Question.create("Was it released before 1990?", "release_year", Operator.LESS_THAN, 1990, "time")
    assert q_old.evaluate(entity) == PlayerAnswer.NO
