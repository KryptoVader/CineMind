"""
DataLoader for CineMind Canonical Entity Datasets.
"""

from pathlib import Path
from typing import List, Optional, Union
import json
import pandas as pd

from cinemind.data.entity import Entity
from cinemind.data.feature_registry import FeatureRegistry, DEFAULT_FEATURE_REGISTRY

# Default canonical file paths
DEFAULT_CANONICAL_PATH = Path("data/canonical/canonical_entities.parquet")
DIVERSE_100K_PATH = Path("data/canonical/diverse_100k.parquet")


class DataLoader:
    """Loads canonical parquet dataset and instantiates domain Entity objects."""

    def __init__(
        self,
        filepath: Union[str, Path] = DEFAULT_CANONICAL_PATH,
        registry: Optional[FeatureRegistry] = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.registry = registry or DEFAULT_FEATURE_REGISTRY

    def load_dataframe(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load canonical Parquet file into pandas DataFrame."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Canonical dataset file not found at: {self.filepath.resolve()}")

        df = pd.read_parquet(self.filepath)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df

    def load_entities(self, sample_size: Optional[int] = None) -> List[Entity]:
        """
        Load dataset and convert rows into Entity domain objects with parsed list features.
        """
        df = self.load_dataframe(sample_size=sample_size)
        records = df.to_dict(orient="records")
        entities: List[Entity] = []

        json_list_cols = {
            "genres", "origin_country", "keywords", "studios",
            "production_companies", "production_countries",
            "themes", "demographics", "alternative_titles", "discovered_from",
            "source_presence",
        }

        for record in records:
            # Clean and parse JSON list columns
            for col in json_list_cols:
                if col in record:
                    val = record[col]
                    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                        try:
                            record[col] = json.loads(val)
                        except Exception:
                            record[col] = []
                    elif isinstance(val, float) and (val != val):  # NaN check
                        record[col] = None

            entities.append(Entity(data=record, registry=self.registry))

        return entities
