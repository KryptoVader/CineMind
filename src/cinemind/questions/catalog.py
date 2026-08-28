"""
Question Catalog Management and Serialization for CineMind.

Compiles, profiles, and serializes the reproducible question catalog to Parquet and JSON.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pandas as pd

from cinemind.data.entity import Entity
from cinemind.data.loader import DataLoader
from cinemind.questions.generators import CompositeQuestionGenerator
from cinemind.questions.quality import QuestionQualityProfiler
from cinemind.questions.schema import Question

DEFAULT_CATALOG_PARQUET_PATH = Path("data/model/question_catalog.parquet")
DEFAULT_CATALOG_JSON_PATH = Path("data/model/question_catalog.json")


class QuestionCatalog:
    """Manages the reproducible catalog of profiled questions."""

    def __init__(self, questions: Optional[List[Question]] = None) -> None:
        self._questions: List[Question] = questions or []
        self._by_id: Dict[str, Question] = {q.question_id: q for q in self._questions}

    def add(self, question: Question) -> None:
        """Add a single question to catalog."""
        self._questions.append(question)
        self._by_id[question.question_id] = question

    def get_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve question by unique ID."""
        return self._by_id.get(question_id)

    def get_by_family(self, family: str) -> List[Question]:
        """Retrieve questions belonging to a specific family."""
        return [q for q in self._questions if q.question_family == family]

    def list_questions(self) -> List[Question]:
        """Return all questions in catalog."""
        return list(self._questions)

    def __len__(self) -> int:
        return len(self._questions)

    @classmethod
    def build_catalog(
        cls,
        entities: List[Entity],
        min_coverage: float = 0.01,
        min_balance: float = 0.001,
    ) -> "QuestionCatalog":
        """
        Generates candidate questions, profiles coverage/balance/missingness/redundancy,
        and constructs a reproducible QuestionCatalog.
        """
        generator = CompositeQuestionGenerator()
        candidates = generator.generate(entities)

        profiler = QuestionQualityProfiler(entities)
        qualified_questions = profiler.profile_all(
            candidates,
            min_coverage=min_coverage,
            min_balance=min_balance,
        )

        return cls(questions=qualified_questions)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert catalog to pandas DataFrame."""
        rows = [q.to_dict() for q in self._questions]
        return pd.DataFrame(rows)

    def save(
        self,
        parquet_path: Union[str, Path] = DEFAULT_CATALOG_PARQUET_PATH,
        json_path: Optional[Union[str, Path]] = DEFAULT_CATALOG_JSON_PATH,
    ) -> None:
        """Save catalog to disk (Parquet and JSON)."""
        p_path = Path(parquet_path)
        p_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_parquet(p_path, index=False)

        if json_path:
            j_path = Path(json_path)
            j_path.parent.mkdir(parents=True, exist_ok=True)
            with open(j_path, "w", encoding="utf-8") as f:
                json.dump([q.to_dict() for q in self._questions], f, indent=2)

    @classmethod
    def load(cls, parquet_path: Union[str, Path] = DEFAULT_CATALOG_PARQUET_PATH) -> "QuestionCatalog":
        """Load question catalog from Parquet file."""
        p_path = Path(parquet_path)
        if not p_path.exists():
            raise FileNotFoundError(f"Question catalog not found at: {p_path.resolve()}")

        df = pd.read_parquet(p_path)
        records = df.to_dict(orient="records")
        questions = [Question.from_dict(r) for r in records]
        return cls(questions=questions)


if __name__ == "__main__":
    print("Building reproducible CineMind Question Catalog...")
    loader = DataLoader(filepath=Path("data/canonical/canonical_entities.parquet"))
    print("Loading entity dataset...")
    # Using 10,000 entities for fast profiling demo or full dataset
    entities = loader.load_entities(sample_size=20000)
    print(f"Loaded {len(entities)} entities.")

    catalog = QuestionCatalog.build_catalog(entities)
    print(f"Compiled {len(catalog)} qualified questions across {len(set(q.question_family for q in catalog.list_questions()))} families.")

    catalog.save()
    print(f"Catalog successfully saved to {DEFAULT_CATALOG_PARQUET_PATH} and {DEFAULT_CATALOG_JSON_PATH}.")
