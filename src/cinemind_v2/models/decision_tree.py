"""
CineMind V2 — Decision Tree Model
Classical sklearn DecisionTreeClassifier using entropy criterion for feature split discovery and fixed tree traversal.
"""

from typing import Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from cinemind_v2.knowledge.feature_store import FeatureStore


class DecisionTreeModel:
    """
    Entropy-based Decision Tree model for structural feature discovery
    and fixed-tree questioning baseline.
    """

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        if criterion != "entropy":
            raise ValueError("DecisionTreeModel must use criterion='entropy'")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.clf: Optional[DecisionTreeClassifier] = None
        self.feature_ids: list[str] = []
        self.cinemind_ids: list[str] = []
        self.target_type: str = ""
        self.leaf_entity_map: dict[int, list[str]] = {}

    def fit(self, feature_store: FeatureStore, target_type: str = "entity_id") -> "DecisionTreeModel":
        """
        Train DecisionTreeClassifier over feature_store.
        X shape: (num_entities, num_features)
        y: target_type can be 'entity_id' (partitioning entities) or 'media_type' (structural category).
        """
        self.feature_ids = list(feature_store.feature_ids)
        self.cinemind_ids = list(feature_store.entity_store.get_cinemind_ids())
        self.target_type = target_type

        # X is (N, F)
        X = feature_store.feature_matrix.T.astype(np.float32)
        num_entities, num_features = X.shape

        if target_type == "entity_id":
            # Discrete entity index target [0..N-1]
            y = np.arange(num_entities)
        elif target_type == "media_type":
            df = feature_store.df
            y = df["media_type"].astype(str).values
        else:
            raise ValueError(f"Unsupported target_type '{target_type}'. Choose 'entity_id' or 'media_type'")

        self.clf = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )
        self.clf.fit(X, y)

        # Build leaf entity map
        leaf_nodes = self.clf.apply(X)
        self.leaf_entity_map = {}
        for idx, leaf_id in enumerate(leaf_nodes):
            leaf_id_int = int(leaf_id)
            if leaf_id_int not in self.leaf_entity_map:
                self.leaf_entity_map[leaf_id_int] = []
            self.leaf_entity_map[leaf_id_int].append(self.cinemind_ids[idx])

        return self

    @property
    def depth(self) -> int:
        if self.clf is None:
            raise RuntimeError("Model is not fitted yet")
        return int(self.clf.get_depth())

    @property
    def n_leaves(self) -> int:
        if self.clf is None:
            raise RuntimeError("Model is not fitted yet")
        return int(self.clf.get_n_leaves())

    @property
    def feature_importances(self) -> dict[str, float]:
        """Returns dict of feature_id -> float importance score."""
        if self.clf is None:
            raise RuntimeError("Model is not fitted yet")
        importances = self.clf.feature_importances_
        return {fid: float(imp) for fid, imp in zip(self.feature_ids, importances)}

    @property
    def split_features(self) -> list[str]:
        """Returns ordered list of feature IDs used as internal split nodes."""
        if self.clf is None:
            raise RuntimeError("Model is not fitted yet")
        tree = self.clf.tree_
        feature_indices = tree.feature[tree.feature >= 0]
        return [self.feature_ids[idx] for idx in feature_indices]

    @property
    def split_counts(self) -> dict[str, int]:
        """Returns dict of feature_id -> count of times used as a split node in tree."""
        counts: dict[str, int] = {}
        for fid in self.split_features:
            counts[fid] = counts.get(fid, 0) + 1
        return counts

    def traverse(self, entity_features: np.ndarray) -> tuple[list[str], int, list[str]]:
        """
        Traverse the decision tree for a single entity's feature vector of shape (F,).
        Returns:
            - path_feature_ids: list of feature IDs evaluated along decision path
            - leaf_id: internal leaf node index
            - leaf_entities: list of cinemind_ids falling into this leaf node
        """
        if self.clf is None:
            raise RuntimeError("Model is not fitted yet")

        tree = self.clf.tree_
        x_2d = entity_features.reshape(1, -1).astype(np.float32)

        # Extract decision path
        node_indicator = self.clf.decision_path(x_2d)
        node_indices = node_indicator.indices

        path_feature_ids = []
        for node_id in node_indices:
            # Check if internal node
            if tree.feature[node_id] >= 0:
                feat_idx = tree.feature[node_id]
                path_feature_ids.append(self.feature_ids[feat_idx])

        leaf_id = int(self.clf.apply(x_2d)[0])
        leaf_entities = self.leaf_entity_map.get(leaf_id, [])

        return path_feature_ids, leaf_id, leaf_entities
