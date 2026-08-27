"""
Unit tests for CineMind V2 Knowledge Store and Row Alignment.
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.feature_store import FeatureStore


def test_entity_store_and_row_alignment():
    # Build toy dataframe
    data = {
        "cinemind_id": ["id_100", "id_200", "id_300"],
        "title": ["Inception", "The Matrix", "Spirited Away"],
        "media_type": ["movie", "movie", "anime"],
        "release_year": [2010, 1999, 2001],
        "original_language": ["en", "en", "ja"],
        "rating": [8.8, 8.7, 8.6],
        "genres": [["Action", "Sci-Fi"], ["Action", "Sci-Fi"], ["Animation", "Fantasy"]],
    }
    df = pd.DataFrame(data)

    store = EntityStore(df)
    assert store.num_entities == 3
    assert store.get_index("id_200") == 1
    assert store.get_entity("id_200").title == "The Matrix"

    # Test FeatureStore alignment
    fs = FeatureStore(store, df)
    assert fs.validate_alignment() is True

    # Test feature retrieval
    movie_mask = fs.get_feature_values("media:movie")
    assert np.array_equal(movie_mask, [True, True, False])

    anime_mask = fs.get_feature_values("media:anime")
    assert np.array_equal(anime_mask, [False, False, True])


def test_missing_feature_behavior():
    # Dataframe with missing/NaN release_year and rating
    data = {
        "cinemind_id": ["id_1", "id_2"],
        "title": ["Unknown Movie", "Known Movie"],
        "media_type": ["movie", "movie"],
        "release_year": [np.nan, 2022],
        "rating": [np.nan, 8.0],
        "genres": [None, ["Drama"]],
    }
    df = pd.DataFrame(data)

    store = EntityStore(df)
    fs = FeatureStore(store, df)

    # Check decade unknown mask
    decade_2020_unknown = fs.get_unknown_values("decade:2020s")
    assert decade_2020_unknown[0] is True or decade_2020_unknown[0] == 1
    assert decade_2020_unknown[1] is False or decade_2020_unknown[1] == 0


if __name__ == "__main__":
    test_entity_store_and_row_alignment()
    test_missing_feature_behavior()
    print("All knowledge store tests passed successfully!")
