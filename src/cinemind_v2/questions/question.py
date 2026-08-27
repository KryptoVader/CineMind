"""
CineMind V2 — Question Representation
Formal Question dataclass and quality attributes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Question:
    id: str
    type: str  # metadata, genre, language, decade, rating, etc.
    text: str
    feature_id: str
    reliability: float = 0.90
    coverage: float = 0.0
    yes_rate: float = 0.0
    unknown_rate: float = 0.0
    entropy: float = 0.0
    source: str = "structured_metadata"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "feature_id": self.feature_id,
            "reliability": self.reliability,
            "coverage": self.coverage,
            "yes_rate": self.yes_rate,
            "unknown_rate": self.unknown_rate,
            "entropy": self.entropy,
            "source": self.source,
        }
