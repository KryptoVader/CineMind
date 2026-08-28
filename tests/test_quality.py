"""
Unit tests for Question Quality Profiler.
"""

import pytest
from cinemind.data.entity import Entity
from cinemind.questions.quality import QuestionQualityProfiler
from cinemind.questions.schema import Operator, Question


@pytest.fixture
def sample_entities():
    data = [
        {"cinemind_id": "1", "media_type": "movie", "genres": ["Comedy"], "release_year": 2010.0},
        {"cinemind_id": "2", "media_type": "movie", "genres": ["Comedy"], "release_year": 2012.0},
        {"cinemind_id": "3", "media_type": "tv", "genres": ["Action"], "release_year": 1995.0},
        {"cinemind_id": "4", "media_type": "anime", "genres": ["Action"], "release_year": 2020.0},
    ]
    return [Entity(d) for d in data]


def test_quality_profiler(sample_entities):
    profiler = QuestionQualityProfiler(sample_entities)

    # 50/50 split question
    q_movie = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")
    profile_movie = profiler.profile_question(q_movie)

    assert profile_movie.coverage == 1.0
    assert profile_movie.balance == 1.0  # 2 movie vs 2 non-movie -> 2/2 = 1.0
    assert profile_movie.missingness == 0.0

    # 75/25 split question
    q_action = Question.create("Is it an action title?", "genres", Operator.CONTAINS, "Action", "genre")
    profile_action = profiler.profile_question(q_action)

    assert profile_action.coverage == 1.0
    assert profile_action.balance == 2 / 2  # 2 action vs 2 comedy -> 1.0


def test_redundancy_detection(sample_entities):
    profiler = QuestionQualityProfiler(sample_entities)
    q1 = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")
    q2 = Question.create("Is it a feature movie?", "media_type", Operator.EQUALS, "movie", "media")

    questions = [q1, q2]
    profiler.detect_redundancy(questions)

    assert q1.redundancy_group is not None
    assert q1.redundancy_group == q2.redundancy_group
