"""
Unit tests for CineMind V2 Decision Tree Model.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.models.decision_tree import DecisionTreeModel


def test_decision_tree_criterion_and_training():
    data = {
        "cinemind_id": ["ent_1", "ent_2", "ent_3", "ent_4"],
        "title": ["Movie A", "Movie B", "TV C", "Anime D"],
        "media_type": ["movie", "movie", "tv", "anime"],
        "release_year": [2010, 2012, 2021, 2019],
        "original_language": ["en", "en", "en", "ja"],
        "rating": [8.5, 7.0, 9.0, 8.0],
        "genres": [["Action"], ["Action"], ["Drama"], ["Animation"]],
    }
    df = pd.DataFrame(data)
    store = EntityStore(df)
    fs = FeatureStore(store, df)

    # 1. Test entropy criterion requirement
    dt = DecisionTreeModel(criterion="entropy", random_state=42)
    dt.fit(fs, target_type="entity_id")

    assert dt.criterion == "entropy"
    assert dt.clf is not None
    assert dt.clf.criterion == "entropy"
    assert dt.depth >= 1
    assert dt.n_leaves >= 2

    # 2. Test feature importances dimensions
    importances = dt.feature_importances
    assert len(importances) == len(fs.feature_ids)
    assert sum(importances.values()) > 0.0

    # 3. Test extracted split features
    splits = dt.split_features
    assert len(splits) >= 1
    for s_fid in splits:
        assert s_fid in fs.feature_ids

    # 4. Test deterministic output
    dt2 = DecisionTreeModel(criterion="entropy", random_state=42)
    dt2.fit(fs, target_type="entity_id")
    assert dt.depth == dt2.depth
    assert dt.n_leaves == dt2.n_leaves
    assert dt.feature_importances == dt2.feature_importances


def test_decision_tree_traversal():
    data = {
        "cinemind_id": ["ent_1", "ent_2", "ent_3", "ent_4"],
        "title": ["Movie A", "Movie B", "TV C", "Anime D"],
        "media_type": ["movie", "movie", "tv", "anime"],
        "release_year": [2010, 2012, 2021, 2019],
        "original_language": ["en", "en", "en", "ja"],
        "rating": [8.5, 7.0, 9.0, 8.0],
        "genres": [["Action"], ["Action"], ["Drama"], ["Animation"]],
    }
    df = pd.DataFrame(data)
    store = EntityStore(df)
    fs = FeatureStore(store, df)

    dt = DecisionTreeModel(criterion="entropy", random_state=42)
    dt.fit(fs, target_type="entity_id")

    # Traverse for entity 0
    ent_feat = fs.feature_matrix[:, 0]
    path, leaf_id, leaf_ents = dt.traverse(ent_feat)

    assert isinstance(path, list)
    assert len(path) >= 1
    assert "ent_1" in leaf_ents


if __name__ == "__main__":
    test_decision_tree_criterion_and_training()
    test_decision_tree_traversal()
    print("All Decision Tree unit tests passed successfully!")
