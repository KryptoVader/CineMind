"""
CineMind V2 — Entity Store
Exposes canonical entity lookup and stable ID-to-index mapping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class Entity:
    cinemind_id: str
    title: str
    media_type: str
    release_year: Optional[int] = None
    original_language: Optional[str] = None
    origin_country: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    rating: Optional[float] = None
    vote_count: Optional[float] = None
    popularity: Optional[float] = None
    runtime: Optional[float] = None
    num_episodes: Optional[float] = None
    source_material: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    studios: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


class EntityStore:
    """
    EntityStore holds canonical entity metadata and provides O(1) index/ID lookup.
    Guarantees strict indexing alignment.
    """

    def __init__(self, df: pd.DataFrame):
        if "cinemind_id" not in df.columns:
            raise ValueError("DataFrame must contain 'cinemind_id' column")

        self.df = df.reset_index(drop=True)
        self.num_entities = len(self.df)

        # Build ID mapping
        self._id_to_index: dict[str, int] = {
            str(cid): idx for idx, cid in enumerate(self.df["cinemind_id"])
        }
        if len(self._id_to_index) != self.num_entities:
            raise ValueError("Duplicate 'cinemind_id' detected in dataset!")

        self._cinemind_ids: list[str] = list(self.df["cinemind_id"].astype(str))

    @classmethod
    def from_parquet(cls, parquet_path: Union[str, Path], limit: Optional[int] = None) -> "EntityStore":
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at {path}")

        df = pd.read_parquet(path)
        if limit is not None and limit > 0:
            df = df.head(limit).copy()
        return cls(df)

    def get_entity_by_index(self, idx: int) -> Entity:
        if idx < 0 or idx >= self.num_entities:
            raise IndexError(f"Index {idx} out of range [0, {self.num_entities})")

        row = self.df.iloc[idx]
        return self._row_to_entity(row)

    def get_entity(self, cinemind_id: str) -> Entity:
        idx = self.get_index(cinemind_id)
        return self.get_entity_by_index(idx)

    def get_index(self, cinemind_id: str) -> int:
        idx = self._id_to_index.get(str(cinemind_id))
        if idx is None:
            raise KeyError(f"Entity ID '{cinemind_id}' not found in EntityStore")
        return idx

    def get_cinemind_ids(self) -> list[str]:
        return self._cinemind_ids

    def _row_to_entity(self, row: pd.Series) -> Entity:
        def _to_list(val: Any) -> list[str]:
            if isinstance(val, (list, np.ndarray, tuple)):
                return [str(x) for x in val if pd.notna(x)]
            if isinstance(val, str):
                val_str = val.strip()
                if val_str.startswith("[") and val_str.endswith("]"):
                    import json
                    try:
                        parsed = json.loads(val_str)
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed if x]
                    except Exception:
                        pass
                return [val_str] if val_str else []
            return []

        release_year = int(row["release_year"]) if pd.notna(row.get("release_year")) else None
        rating = float(row["rating"]) if pd.notna(row.get("rating")) else None
        vote_count = float(row["vote_count"]) if pd.notna(row.get("vote_count")) else None
        popularity = float(row["popularity"]) if pd.notna(row.get("popularity")) else None
        runtime = float(row["runtime"]) if pd.notna(row.get("runtime")) else None
        num_episodes = float(row["num_episodes"]) if pd.notna(row.get("num_episodes")) else None

        return Entity(
            cinemind_id=str(row["cinemind_id"]),
            title=str(row.get("title", "")),
            media_type=str(row.get("media_type", "")),
            release_year=release_year,
            original_language=str(row["original_language"]) if pd.notna(row.get("original_language")) else None,
            origin_country=_to_list(row.get("origin_country")),
            genres=_to_list(row.get("genres")),
            rating=rating,
            vote_count=vote_count,
            popularity=popularity,
            runtime=runtime,
            num_episodes=num_episodes,
            source_material=str(row["source_material"]) if pd.notna(row.get("source_material")) else None,
            keywords=_to_list(row.get("keywords")),
            studios=_to_list(row.get("studios")),
            raw_data=row.to_dict(),
        )
