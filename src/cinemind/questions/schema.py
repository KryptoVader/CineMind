"""
Question Schema and Representation for CineMind.

Supports first-class representation of player questions, predicate evaluation against entities,
and strict non-leakage enforcement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union
import hashlib
import json

from cinemind.data.entity import Entity
from cinemind.data.feature_registry import FeatureRegistry, DEFAULT_FEATURE_REGISTRY
from cinemind.data.schemas import FeatureState


class PlayerAnswer(Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class Operator(Enum):
    EQUALS = "equals"
    CONTAINS = "contains"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_SET = "in_set"
    BETWEEN = "between"


@dataclass
class Question:
    question_id: str
    text: str
    feature: str
    operator: Union[Operator, str]
    value: Any
    question_family: str
    source: str = "structured_generator"

    # Quality and profiling metadata
    coverage: float = 0.0
    reliability: float = 1.0
    answerability: float = 1.0
    balance: float = 0.0
    missingness: float = 0.0
    redundancy_group: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce operator enum normalization
        if isinstance(self.operator, str):
            try:
                self.operator = Operator(self.operator.lower())
            except ValueError:
                pass

        # STRICT LEAKAGE PROTECTION: Validate feature against registry
        DEFAULT_FEATURE_REGISTRY.validate_for_gameplay(self.feature)

    @classmethod
    def create(
        cls,
        text: str,
        feature: str,
        operator: Union[Operator, str],
        value: Any,
        question_family: str,
        source: str = "structured_generator",
        reliability: float = 1.0,
        answerability: float = 1.0,
    ) -> "Question":
        """Factory method to construct Question with deterministic question_id."""
        op_str = operator.value if isinstance(operator, Operator) else str(operator)
        raw_id_str = f"{question_family}:{feature}:{op_str}:{value}"
        hash_id = hashlib.sha256(raw_id_str.encode("utf-8")).hexdigest()[:12]
        question_id = f"q_{question_family}_{hash_id}"

        return cls(
            question_id=question_id,
            text=text,
            feature=feature,
            operator=operator,
            value=value,
            question_family=question_family,
            source=source,
            reliability=reliability,
            answerability=answerability,
        )

    def evaluate(self, entity: Entity) -> PlayerAnswer:
        """
        Evaluate question against entity state.

        Returns:
            PlayerAnswer.YES if KNOWN_TRUE
            PlayerAnswer.NO if KNOWN_FALSE
            PlayerAnswer.UNKNOWN if UNKNOWN or NOT_APPLICABLE
        """
        op_str = self.operator.value if isinstance(self.operator, Operator) else str(self.operator)
        state = entity.evaluate_predicate(
            feature_name=self.feature,
            operator=op_str,
            value=self.value,
        )

        if state == FeatureState.KNOWN_TRUE:
            return PlayerAnswer.YES
        elif state == FeatureState.KNOWN_FALSE:
            return PlayerAnswer.NO
        else:
            return PlayerAnswer.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert question object to serializable dictionary."""
        op_str = self.operator.value if isinstance(self.operator, Operator) else str(self.operator)
        return {
            "question_id": self.question_id,
            "text": self.text,
            "feature": self.feature,
            "operator": op_str,
            "value": json.dumps(self.value) if isinstance(self.value, (list, dict)) else str(self.value),
            "question_family": self.question_family,
            "source": self.source,
            "coverage": float(self.coverage),
            "reliability": float(self.reliability),
            "answerability": float(self.answerability),
            "balance": float(self.balance),
            "missingness": float(self.missingness),
            "redundancy_group": self.redundancy_group,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        """Reconstruct question object from dictionary."""
        val = d["value"]
        if isinstance(val, str) and (val.startswith("[") or val.startswith("{")):
            try:
                val = json.loads(val)
            except Exception:
                pass

        # Try parsing numeric values if applicable
        if isinstance(val, str):
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass

        q = cls(
            question_id=d["question_id"],
            text=d["text"],
            feature=d["feature"],
            operator=d["operator"],
            value=val,
            question_family=d["question_family"],
            source=d.get("source", "structured_generator"),
            coverage=float(d.get("coverage", 0.0)),
            reliability=float(d.get("reliability", 1.0)),
            answerability=float(d.get("answerability", 1.0)),
            balance=float(d.get("balance", 0.0)),
            missingness=float(d.get("missingness", 0.0)),
            redundancy_group=d.get("redundancy_group"),
        )
        return q
