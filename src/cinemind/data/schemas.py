"""
Missingness Semantics and Data Schemas for CineMind.

CRITICAL REQUIREMENT:
Do NOT collapse missing values into False.
We explicitly distinguish internally between:
- KNOWN_TRUE: Explicitly confirmed affirmative.
- KNOWN_FALSE: Explicitly confirmed negative.
- UNKNOWN: Missing, null, or empty metadata in source data.
- NOT_APPLICABLE: Feature does not apply to entity's domain or media format.
"""

from enum import Enum
from typing import Any, List, Optional, Set, Union


class FeatureState(Enum):
    KNOWN_TRUE = "KNOWN_TRUE"
    KNOWN_FALSE = "KNOWN_FALSE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    @property
    def is_known(self) -> bool:
        return self in (FeatureState.KNOWN_TRUE, FeatureState.KNOWN_FALSE)

    @property
    def is_usable_evidence(self) -> bool:
        """Returns True if this state provides informative evidence for splitting."""
        return self.is_known


def is_missing_value(val: Any) -> bool:
    """Check if raw python/pandas value represents missingness (None, NaN, empty list/dict, empty string)."""
    if val is None:
        return True
    if isinstance(val, float) and (val != val or str(val) == "nan"):  # NaN check
        return True
    if isinstance(val, (list, set, dict, tuple)) and len(val) == 0:
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("", "nan", "none", "null", "[]", "{}"):
            return True
    return False


def resolve_feature_state(
    entity_val: Any,
    predicate_type: str,
    target_value: Any,
    media_type: Optional[str] = None,
    feature_name: Optional[str] = None,
) -> FeatureState:
    """
    Evaluates entity feature value against a target predicate without collapsing missingness into False.

    Args:
        entity_val: The raw value of the feature for the entity.
        predicate_type: 'equals', 'contains', 'gte', 'lte', 'in_set', 'between'
        target_value: Target benchmark value or collection.
        media_type: Entity's media type ('movie', 'tv', 'anime') to check domain applicability.
        feature_name: Feature name to check entity domain applicability.

    Returns:
        FeatureState (KNOWN_TRUE, KNOWN_FALSE, UNKNOWN, or NOT_APPLICABLE).
    """

    # 1. Check for domain Applicability
    if media_type == "movie" and feature_name in ("num_episodes", "end_date"):
        return FeatureState.NOT_APPLICABLE

    # 2. Check for missingness in entity data
    if is_missing_value(entity_val):
        return FeatureState.UNKNOWN

    # 3. Evaluate Predicate
    try:
        p_type = predicate_type.lower()

        if p_type == "equals":
            if isinstance(entity_val, str) and isinstance(target_value, str):
                match = entity_val.strip().lower() == target_value.strip().lower()
            else:
                match = entity_val == target_value
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type == "contains":
            if isinstance(entity_val, (list, tuple, set)):
                # Normalizing string matching inside collections
                if isinstance(target_value, str):
                    target_norm = target_value.strip().lower()
                    match = any(
                        isinstance(item, str) and item.strip().lower() == target_norm
                        for item in entity_val
                    )
                else:
                    match = target_value in entity_val
            elif isinstance(entity_val, str) and isinstance(target_value, str):
                match = target_value.strip().lower() in entity_val.strip().lower()
            else:
                match = False
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type in ("gte", "greater_equal", ">="):
            match = float(entity_val) >= float(target_value)
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type in ("lte", "less_equal", "<="):
            match = float(entity_val) <= float(target_value)
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type in ("gt", "greater_than", ">"):
            match = float(entity_val) > float(target_value)
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type in ("lt", "less_than", "<"):
            match = float(entity_val) < float(target_value)
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type in ("in_set", "in"):
            if isinstance(target_value, (list, set, tuple)):
                if isinstance(entity_val, str):
                    e_norm = entity_val.strip().lower()
                    match = any(isinstance(t, str) and t.strip().lower() == e_norm for t in target_value)
                else:
                    match = entity_val in target_value
            else:
                match = False
            return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE

        elif p_type == "between":
            if isinstance(target_value, (list, tuple)) and len(target_value) == 2:
                low, high = target_value
                match = float(low) <= float(entity_val) <= float(high)
                return FeatureState.KNOWN_TRUE if match else FeatureState.KNOWN_FALSE
            return FeatureState.UNKNOWN

    except (ValueError, TypeError):
        # Evaluation type error results in UNKNOWN rather than silent False
        return FeatureState.UNKNOWN

    return FeatureState.UNKNOWN
