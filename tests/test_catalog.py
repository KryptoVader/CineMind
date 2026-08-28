"""
Unit tests for QuestionCatalog compilation and serialization.
"""

import pytest
from pathlib import Path
from cinemind.data.entity import Entity
from cinemind.questions.catalog import QuestionCatalog
from cinemind.questions.schema import Operator, Question


@pytest.fixture
def sample_entities():
    data = [
        {"cinemind_id": f"id_{i}", "media_type": "movie" if i % 2 == 0 else "tv", "genres": ["Action" if i < 5 else "Comedy"], "release_year": 2000.0 + i}
        for i in range(10)
    ]
    return [Entity(d) for d in data]


def test_catalog_build_and_serialization(sample_entities, tmp_path):
    catalog = QuestionCatalog.build_catalog(sample_entities, min_coverage=0.01, min_balance=0.001)
    assert len(catalog) > 0

    parquet_file = tmp_path / "catalog.parquet"
    json_file = tmp_path / "catalog.json"

    catalog.save(parquet_path=parquet_file, json_path=json_file)
    assert parquet_file.exists()
    assert json_file.exists()

    loaded_catalog = QuestionCatalog.load(parquet_path=parquet_file)
    assert len(loaded_catalog) == len(catalog)
    first_q = loaded_catalog.list_questions()[0]
    assert isinstance(first_q, Question)
    assert first_q.question_id.startswith("q_")
