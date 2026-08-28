"""
Unit tests for FeatureRegistry and Zero-Leakage Protection.
"""

import pytest
from cinemind.data.feature_registry import FeatureCategory, FeatureDefinition, FeatureRegistry, DEFAULT_FEATURE_REGISTRY


def test_leakage_fields_classified_as_never_expose():
    registry = FeatureRegistry()
    leakage_fields = [
        "cinemind_id",
        "tmdb_id",
        "mal_id",
        "source_id",
        "source",
        "title",
        "original_title",
        "alternative_titles",
    ]
    for field_name in leakage_fields:
        assert registry.get_category(field_name) == FeatureCategory.NEVER_EXPOSE
        assert registry.is_leakage(field_name) is True


def test_validate_for_gameplay_raises_on_leakage():
    registry = FeatureRegistry()
    with pytest.raises(ValueError, match="LEAKAGE PROTECTION ERROR"):
        registry.validate_for_gameplay("title")

    with pytest.raises(ValueError, match="LEAKAGE PROTECTION ERROR"):
        registry.validate_for_gameplay("tmdb_id")

    with pytest.raises(ValueError, match="LEAKAGE PROTECTION ERROR"):
        registry.validate_for_gameplay("cinemind_id")


def test_validate_for_gameplay_raises_on_learning_only():
    registry = FeatureRegistry()
    with pytest.raises(ValueError, match="GAMEPLAY RESTRICTION ERROR"):
        registry.validate_for_gameplay("vote_count")

    with pytest.raises(ValueError, match="GAMEPLAY RESTRICTION ERROR"):
        registry.validate_for_gameplay("popularity")


def test_gameplay_eligible_features():
    registry = FeatureRegistry()
    registry.validate_for_gameplay("media_type")
    registry.validate_for_gameplay("genres")
    registry.validate_for_gameplay("release_year")
    registry.validate_for_gameplay("runtime")
    registry.validate_for_gameplay("original_language")

    gameplay_features = registry.get_gameplay_features()
    names = [f.name for f in gameplay_features]
    assert "media_type" in names
    assert "genres" in names
    assert "release_year" in names
    assert "title" not in names
    assert "vote_count" not in names
