"""
CineMind V2 — Feature Store
Stores boolean/float feature matrices and enforces strict alignment with EntityStore.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.predicates import build_structured_predicates


class FeatureStore:
    """
    FeatureStore maintains structured entity features and explicitly validates
    row alignment against an EntityStore.
    """

    def __init__(self, entity_store: EntityStore, df: Optional[pd.DataFrame] = None):
        self.entity_store = entity_store
        self.df = df if df is not None else entity_store.df
        self.num_entities = entity_store.num_entities

        # Validate entity count
        if len(self.df) != self.num_entities:
            raise ValueError("DataFrame row count does not match EntityStore count")

        # Build feature definitions
        predicates = build_structured_predicates(self.df)

        self.feature_ids: list[str] = [p["feature_id"] for p in predicates]
        self._feature_id_to_idx: dict[str, int] = {fid: idx for idx, fid in enumerate(self.feature_ids)}
        self._feature_metadata: dict[str, dict[str, Any]] = {p["feature_id"]: p for p in predicates}

        # Construct 2D boolean feature matrix of shape (num_features, num_entities)
        num_features = len(self.feature_ids)
        self.feature_matrix = np.zeros((num_features, self.num_entities), dtype=bool)
        self.unknown_matrix = np.zeros((num_features, self.num_entities), dtype=bool)

        for idx, p in enumerate(predicates):
            self.feature_matrix[idx] = p["mask"]
            self.unknown_matrix[idx] = p["unknown_mask"]

        # Validate row alignment
        self.validate_alignment()

    def validate_alignment(self) -> bool:
        """Explicitly validate ID ordering alignment."""
        store_ids = self.entity_store.get_cinemind_ids()
        df_ids = list(self.df["cinemind_id"].astype(str))
        if store_ids != df_ids:
            raise ValueError("Row alignment mismatch between EntityStore and FeatureStore!")
        return True

    def has_feature(self, feature_id: str) -> bool:
        return feature_id in self._feature_id_to_idx

    def get_feature_index(self, feature_id: str) -> int:
        idx = self._feature_id_to_idx.get(feature_id)
        if idx is None:
            raise KeyError(f"Feature '{feature_id}' not found in FeatureStore")
        return idx

    def get_feature_values(self, feature_id: str) -> np.ndarray:
        """Returns boolean array of shape (num_entities,) for feature_id."""
        idx = self.get_feature_index(feature_id)
        return self.feature_matrix[idx]

    def get_unknown_values(self, feature_id: str) -> np.ndarray:
        """Returns boolean array of shape (num_entities,) indicating unknown values."""
        idx = self.get_feature_index(feature_id)
        return self.unknown_matrix[idx]

    def get_entities_with_feature(self, feature_id: str) -> list[str]:
        """Returns list of cinemind_ids that have feature = True."""
        mask = self.get_feature_values(feature_id)
        cinemind_ids = self.entity_store.get_cinemind_ids()
        return [cinemind_ids[i] for i in range(self.num_entities) if mask[i]]

    def get_feature(self, entity_id: str, feature_id: str) -> bool:
        """Returns feature boolean value for a specific entity ID."""
        ent_idx = self.entity_store.get_index(entity_id)
        feat_idx = self.get_feature_index(feature_id)
        return bool(self.feature_matrix[feat_idx, ent_idx])

    def get_feature_metadata(self, feature_id: str) -> dict[str, Any]:
        return self._feature_metadata[feature_id]
