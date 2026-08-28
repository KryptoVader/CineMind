"""
Entity Domain Model for CineMind.

Decouples the guessing engine and question evaluation from raw DataFrame column access.
"""

from typing import Any, Dict, Optional, Union
import json

from cinemind.data.feature_registry import FeatureCategory, FeatureRegistry, DEFAULT_FEATURE_REGISTRY
from cinemind.data.schemas import FeatureState, is_missing_value, resolve_feature_state


class Entity:
    """Canonical Entity Abstraction wrapping entity attributes."""

    def __init__(
        self,
        data: Dict[str, Any],
        registry: Optional[FeatureRegistry] = None,
    ) -> None:
        self._data = dict(data)
        self.registry = registry or DEFAULT_FEATURE_REGISTRY

        # Basic identity shortcuts (accessible via methods or properties for internal entity management, not gameplay)
        self.cinemind_id: str = str(self._data.get("cinemind_id", ""))
        self.title: str = str(self._data.get("title", ""))
        self.media_type: str = str(self._data.get("media_type", "")).lower()

    def get_feature(self, feature_name: str) -> Any:
        """
        Get normalized raw feature value for an entity.
        Returns parsed lists for list-based features (e.g., genres, origin_country).
        """
        val = self._data.get(feature_name)
        if is_missing_value(val):
            return None

        # Parse stringified JSON arrays if present
        if isinstance(val, str) and (val.startswith("[") and val.endswith("]")):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        return val

    def get_feature_state(
        self,
        feature_name: str,
        predicate_type: str = "equals",
        target_value: Any = None,
    ) -> FeatureState:
        """
        Returns explicit FeatureState (KNOWN_TRUE, KNOWN_FALSE, UNKNOWN, NOT_APPLICABLE)
        for a given feature and optional predicate target.
        """
        raw_val = self.get_feature(feature_name)
        return resolve_feature_state(
            entity_val=raw_val,
            predicate_type=predicate_type,
            target_value=target_value,
            media_type=self.media_type,
            feature_name=feature_name,
        )

    def is_known(self, feature_name: str) -> bool:
        """Check if entity has a valid non-missing value for feature."""
        return not is_missing_value(self.get_feature(feature_name))

    def evaluate_predicate(self, feature_name: str, operator: str, value: Any) -> FeatureState:
        """Evaluate a specific predicate against entity state."""
        raw_val = self.get_feature(feature_name)
        return resolve_feature_state(
            entity_val=raw_val,
            predicate_type=operator,
            target_value=value,
            media_type=self.media_type,
            feature_name=feature_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return raw entity data dictionary."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"<Entity id={self.cinemind_id!r} title={self.title!r} media_type={self.media_type!r}>"
