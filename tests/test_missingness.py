"""
Unit tests for Missingness Semantics.
"""

import pytest
from cinemind.data.schemas import FeatureState, is_missing_value, resolve_feature_state


def test_is_missing_value():
    assert is_missing_value(None) is True
    assert is_missing_value(float("nan")) is True
    assert is_missing_value([]) is True
    assert is_missing_value("") is True
    assert is_missing_value("   ") is True
    assert is_missing_value("nan") is True
    assert is_missing_value("null") is True
    assert is_missing_value("[]") is True

    assert is_missing_value("Action") is False
    assert is_missing_value(2020) is False
    assert is_missing_value(["Action"]) is False


def test_missing_values_do_not_collapse_to_false():
    # Missing genre value should yield UNKNOWN, not KNOWN_FALSE
    state = resolve_feature_state(
        entity_val=None,
        predicate_type="contains",
        target_value="Comedy",
        feature_name="genres",
    )
    assert state == FeatureState.UNKNOWN
    assert state != FeatureState.KNOWN_FALSE


def test_not_applicable_domain_heuristics():
    # Standalone movie should have NOT_APPLICABLE for episode counts
    state = resolve_feature_state(
        entity_val=None,
        predicate_type="gte",
        target_value=12,
        media_type="movie",
        feature_name="num_episodes",
    )
    assert state == FeatureState.NOT_APPLICABLE


def test_known_true_and_known_false():
    # Known affirmative
    state_true = resolve_feature_state(
        entity_val=["Action", "Comedy"],
        predicate_type="contains",
        target_value="Action",
    )
    assert state_true == FeatureState.KNOWN_TRUE

    # Known negative
    state_false = resolve_feature_state(
        entity_val=["Action", "Comedy"],
        predicate_type="contains",
        target_value="Horror",
    )
    assert state_false == FeatureState.KNOWN_FALSE
