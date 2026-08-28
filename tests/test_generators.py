"""
Unit tests for Question Generators.
"""

import pytest
from cinemind.data.entity import Entity
from cinemind.questions.generators import (
    CompositeQuestionGenerator,
    GenreGenerator,
    LanguageGenerator,
    MediaGenerator,
    TimeGenerator,
)
from cinemind.questions.schema import Operator, Question


@pytest.fixture
def sample_entities():
    data = [
        {
            "cinemind_id": "e1",
            "title": "Movie A",
            "media_type": "movie",
            "genres": ["Action", "Comedy"],
            "release_year": 2015.0,
            "original_language": "en",
            "origin_country": ["US"],
            "runtime": 105.0,
            "rating": 7.5,
        },
        {
            "cinemind_id": "e2",
            "title": "Anime B",
            "media_type": "anime",
            "genres": ["Action", "Fantasy"],
            "release_year": 1998.0,
            "original_language": "ja",
            "origin_country": ["JP"],
            "num_episodes": 26.0,
            "rating": 8.8,
        },
        {
            "cinemind_id": "e3",
            "title": "TV C",
            "media_type": "tv",
            "genres": ["Comedy", "Drama"],
            "release_year": 2021.0,
            "original_language": "en",
            "origin_country": ["US"],
            "num_episodes": 10.0,
            "rating": 6.8,
        },
    ]
    return [Entity(d) for d in data]


def test_media_generator(sample_entities):
    gen = MediaGenerator()
    questions = gen.generate(sample_entities)
    families = {q.question_family for q in questions}
    assert "media" in families
    texts = [q.text for q in questions]
    assert "Is it a movie?" in texts
    assert "Is it a TV series?" in texts
    assert "Is it an anime?" in texts


def test_genre_generator(sample_entities):
    gen = GenreGenerator(min_occurrences=1)
    questions = gen.generate(sample_entities)
    genres_generated = {q.value for q in questions}
    assert "Action" in genres_generated
    assert "Comedy" in genres_generated
    assert "Fantasy" in genres_generated


def test_time_generator(sample_entities):
    gen = TimeGenerator()
    questions = gen.generate(sample_entities)
    assert len(questions) > 0
    # Ensure threshold and decade questions are present
    features = {q.feature for q in questions}
    assert "release_year" in features


def test_composite_generator(sample_entities):
    gen = CompositeQuestionGenerator()
    questions = gen.generate(sample_entities)
    assert len(questions) > 10
    # Verify no exact float questions for ratings
    rating_questions = [q for q in questions if q.feature == "rating"]
    for q in rating_questions:
        assert q.operator in (Operator.GREATER_EQUAL, Operator.LESS_EQUAL, Operator.GREATER_THAN, Operator.LESS_THAN)
        assert q.value in (7.0, 8.0, 8.5)
