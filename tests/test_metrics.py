"""
Unit tests for EvaluationMetrics calculations.
"""

import pytest
from cinemind.evaluation.metrics import EvaluationMetrics
from cinemind.evaluation.simulator import SimulationResult


def test_metrics_calculation_and_rate_separation():
    # Game 1: Correct guess
    r1 = SimulationResult(
        target_entity_id="t1",
        target_title="Title 1",
        guessed_entity_id="t1",
        guessed_title="Title 1",
        correct=True,
        questions_asked=5,
        remaining_candidates=1,
        history=[
            {"step": 1, "question_id": "q1", "text": "Is movie?", "question_family": "media", "oracle_answer": "YES", "remaining_candidates": 50},
            {"step": 2, "question_id": "q2", "text": "Is action?", "question_family": "genre", "oracle_answer": "YES", "remaining_candidates": 1},
        ],
        terminated_reason="unique_candidate",
    )

    # Game 2: Incorrect guess (max questions reached with ambiguous pool)
    r2 = SimulationResult(
        target_entity_id="t2",
        target_title="Title 2",
        guessed_entity_id="t3",
        guessed_title="Title 3",
        correct=False,
        questions_asked=25,
        remaining_candidates=4,
        history=[
            {"step": 1, "question_id": "q1", "text": "Is movie?", "question_family": "media", "oracle_answer": "YES", "remaining_candidates": 50},
        ],
        terminated_reason="max_questions_reached",
    )

    # Game 3: Zero candidate diagnostic failure
    r3 = SimulationResult(
        target_entity_id="t4",
        target_title="Title 4",
        guessed_entity_id=None,
        guessed_title=None,
        correct=False,
        questions_asked=3,
        remaining_candidates=0,
        history=[],
        terminated_reason="empty_candidate_set",
        zero_candidate_info={
            "zero_candidate_step": 3,
            "zero_candidate_question_id": "q9",
            "zero_candidate_question_text": "Is comedy?",
            "zero_candidate_answer": "YES",
            "candidate_count_before": 2,
            "candidate_count_after": 0,
        },
    )

    results = [r1, r2, r3]
    metrics = EvaluationMetrics(results=results, initial_universe_size=100)

    assert metrics.total_games == 3
    assert metrics.correct_guesses == 1
    assert round(metrics.accuracy, 4) == round(1 / 3, 4)

    # failure_rate = 2 unsuccessful / 3 total = 0.6667
    assert round(metrics.failure_rate, 4) == round(2 / 3, 4)

    # zero_candidate_rate = 1 zero candidate game / 3 total = 0.3333
    assert round(metrics.zero_candidate_rate, 4) == round(1 / 3, 4)

    assert len(metrics.zero_candidate_diagnostics) == 1
    assert metrics.zero_candidate_diagnostics[0]["zero_candidate_step"] == 3

    assert len(metrics.difficult_targets) == 2  # r2 and r3 were unsuccessful

    report = metrics.summary_report()
    assert "Accuracy:" in report
    assert "Failure Rate:" in report
    assert "Zero-Candidate Rate:" in report
