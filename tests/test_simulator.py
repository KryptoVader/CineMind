"""
Unit tests for Simulator, EliminationEngine, and Target Isolation.
"""

import pytest
import numpy as np

from cinemind.data.entity import Entity
from cinemind.evaluation.oracle import SimulationOracle
from cinemind.evaluation.simulator import EliminationEngine, GameSimulator, TargetSampler
from cinemind.questions.catalog import QuestionCatalog
from cinemind.questions.schema import Operator, PlayerAnswer, Question


@pytest.fixture
def synthetic_universe():
    data = [
        {"cinemind_id": "A", "title": "Alpha", "media_type": "movie", "genres": ["Action"], "release_year": 2010.0},
        {"cinemind_id": "B", "title": "Beta", "media_type": "movie", "genres": ["Comedy"], "release_year": 2012.0},
        {"cinemind_id": "C", "title": "Gamma", "media_type": "tv", "genres": ["Action"], "release_year": 1995.0},
        {"cinemind_id": "D", "title": "Delta", "media_type": "anime", "genres": ["Action"], "release_year": 2020.0},
    ]
    return [Entity(d) for d in data]


def test_elimination_engine_yes_no_unknown(synthetic_universe):
    engine = EliminationEngine(synthetic_universe)
    assert engine.candidate_count == 4

    q_movie = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")

    # Applying YES keeps Alpha & Beta
    engine.apply_answer(q_movie, PlayerAnswer.YES)
    assert engine.candidate_count == 2
    c_ids = [e.cinemind_id for e in engine.candidates]
    assert "A" in c_ids and "B" in c_ids

    # Applying UNKNOWN on remaining candidates retains all of them
    q_missing = Question.create("Is language Japanese?", "original_language", Operator.EQUALS, "ja", "language")
    engine.apply_answer(q_missing, PlayerAnswer.UNKNOWN)
    assert engine.candidate_count == 2  # No elimination on UNKNOWN!


def test_simulator_full_run_with_target_isolation(synthetic_universe):
    q1 = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")
    q2 = Question.create("Is it an action title?", "genres", Operator.CONTAINS, "Action", "genre")
    q3 = Question.create("Is it a TV series?", "media_type", Operator.EQUALS, "tv", "media")

    catalog = QuestionCatalog(questions=[q1, q2, q3])
    simulator = GameSimulator(candidate_universe=synthetic_universe, question_catalog=catalog, max_questions=10)

    target = synthetic_universe[0]  # Entity A ("Alpha")
    result = simulator.run_game(target_entity=target, seed=42)

    assert result.target_entity_id == "A"
    assert result.target_title == "Alpha"
    assert result.questions_asked > 0
    assert len(result.history) > 0

    # Ensure title was not used in history
    for h in result.history:
        assert "target_title" not in h
        assert "target_entity_id" not in h


def test_target_sampler(synthetic_universe):
    rng = np.random.RandomState(42)
    t1 = TargetSampler.sample_uniform(synthetic_universe, rng=rng)
    assert t1.cinemind_id in ["A", "B", "C", "D"]

    t2 = TargetSampler.sample_popularity_weighted(synthetic_universe, rng=rng)
    assert t2.cinemind_id in ["A", "B", "C", "D"]
