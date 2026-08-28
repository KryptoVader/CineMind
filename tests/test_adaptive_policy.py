"""
Unit tests for AdaptiveQuestionPolicy (EntropySplitPolicy and ExpectedCandidateReductionPolicy).
Includes exact mathematical verification of E[|C'|] and R(q) with non-zero UNKNOWN branch.
"""

import pytest
import numpy as np

from cinemind.data.entity import Entity
from cinemind.evaluation.adaptive_policy import (
    EntropySplitPolicy,
    ExpectedCandidateReductionPolicy,
    QuestionSplitScore,
    compute_question_split_score,
)
from cinemind.questions.catalog import QuestionCatalog
from cinemind.questions.schema import Operator, PlayerAnswer, Question


def test_expected_candidate_reduction_formula_with_unknown_branch():
    """
    EXACT MATHEMATICAL VERIFICATION:
    Total candidates N = 100
    C_Y = 40 (p_Y = 0.4)
    C_N = 40 (p_N = 0.4)
    C_U = 20 (p_U = 0.2)

    E[|C'|] = p_Y * |C_Y| + p_N * |C_N| + p_U * |C|
            = 0.4 * 40 + 0.4 * 40 + 0.2 * 100
            = 16 + 16 + 20 = 52

    R(q) = 1 - E[|C'|] / |C| = 1 - 52/100 = 0.48
    """
    # Construct 100 synthetic entities
    # 40 with media_type="movie", 40 with media_type="tv", 20 with missing media_type
    entities = []
    for i in range(40):
        entities.append(Entity({"cinemind_id": f"y_{i}", "media_type": "movie"}))
    for i in range(40):
        entities.append(Entity({"cinemind_id": f"n_{i}", "media_type": "tv"}))
    for i in range(20):
        entities.append(Entity({"cinemind_id": f"u_{i}", "media_type": None}))

    q_movie = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")
    split = compute_question_split_score(q_movie, entities)

    assert split.total_candidates == 100
    assert split.yes_count == 40
    assert split.no_count == 40
    assert split.unknown_count == 20

    assert pytest.approx(split.yes_fraction, 0.001) == 0.4
    assert pytest.approx(split.no_fraction, 0.001) == 0.4
    assert pytest.approx(split.unknown_fraction, 0.001) == 0.2

    # Verify exact formula E[|C'|] = 52
    assert pytest.approx(split.expected_remaining_candidates, 0.001) == 52.0

    # Verify exact formula R(q) = 0.48
    assert pytest.approx(split.expected_reduction_ratio, 0.001) == 0.48

    # Verify Answer Outcome Entropy H(A_q) = -(0.4 log2 0.4 + 0.4 log2 0.4 + 0.2 log2 0.2)
    expected_h = -(0.4 * np.log2(0.4) + 0.4 * np.log2(0.4) + 0.2 * np.log2(0.2))
    assert pytest.approx(split.answer_entropy, 0.001) == expected_h


def test_entropy_split_policy_selects_balanced_question():
    """
    Verifies that EntropySplitPolicy selects a 50/50 split question over an unbalanced
    or UNKNOWN-only question.
    """
    data = [
        {"cinemind_id": "A", "media_type": "movie", "genres": ["Action"]},
        {"cinemind_id": "B", "media_type": "movie", "genres": ["Action"]},
        {"cinemind_id": "C", "media_type": "tv", "genres": ["Comedy"]},
        {"cinemind_id": "D", "media_type": "tv", "genres": ["Comedy"]},
    ]
    candidates = [Entity(d) for d in data]

    # Q1: 50/50 split (A,B -> YES, C,D -> NO) => H(A_q) = 1.0 bit
    q1 = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")

    # Q2: Unbalanced split (A -> YES, B,C,D -> NO)
    q2 = Question.create("Is it Alpha?", "genres", Operator.CONTAINS, "UnusedGenre", "genre")

    # Q3: All UNKNOWN (A,B,C,D -> UNKNOWN) => H(A_q) = 0.0 bits
    q3 = Question.create("Is language Japanese?", "original_language", Operator.EQUALS, "ja", "language")

    catalog = QuestionCatalog(questions=[q1, q2, q3])
    policy = EntropySplitPolicy(catalog=catalog)

    selected_q, split = policy.select_next_question(candidates, asked_question_ids=set())

    assert selected_q is not None
    assert selected_q.question_id == q1.question_id
    assert split.answer_entropy == 1.0


def test_expected_reduction_policy_selects_balanced_question():
    data = [
        {"cinemind_id": "A", "media_type": "movie"},
        {"cinemind_id": "B", "media_type": "movie"},
        {"cinemind_id": "C", "media_type": "tv"},
        {"cinemind_id": "D", "media_type": "tv"},
    ]
    candidates = [Entity(d) for d in data]

    q1 = Question.create("Is it a movie?", "media_type", Operator.EQUALS, "movie", "media")
    q_unk = Question.create("Is language Japanese?", "original_language", Operator.EQUALS, "ja", "language")

    catalog = QuestionCatalog(questions=[q1, q_unk])
    policy = ExpectedCandidateReductionPolicy(catalog=catalog)

    selected_q, split = policy.select_next_question(candidates, asked_question_ids=set())

    assert selected_q is not None
    assert selected_q.question_id == q1.question_id
    assert split.expected_reduction_ratio == 0.5  # 50% candidate pool reduction


def test_deterministic_tie_breaking():
    data = [{"cinemind_id": "A", "media_type": "movie"}, {"cinemind_id": "B", "media_type": "tv"}]
    candidates = [Entity(d) for d in data]

    # Two questions with identical 50/50 split scores but different coverage/reliability
    q_low_cov = Question.create("Q Low Cov", "media_type", Operator.EQUALS, "movie", "media", reliability=0.8)
    q_low_cov.coverage = 0.5

    q_high_cov = Question.create("Q High Cov", "media_type", Operator.EQUALS, "movie", "media", reliability=0.95)
    q_high_cov.coverage = 1.0

    catalog = QuestionCatalog(questions=[q_low_cov, q_high_cov])
    policy = EntropySplitPolicy(catalog=catalog)

    selected_q, _ = policy.select_next_question(candidates, asked_question_ids=set())

    # High coverage & reliability should win tie-break
    assert selected_q.question_id == q_high_cov.question_id
