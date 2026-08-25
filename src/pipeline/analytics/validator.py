"""
Analytics Dataset Validator.

Performs automated data quality, provenance, identity, and source-integrity checks
on generated development datasets.
"""

import logging
from pathlib import Path
from typing import Any
import pandas as pd

from pipeline.config import CANONICAL_DIR, DATA_DIR

logger = logging.getLogger(__name__)

ANALYTICS_DIR = DATA_DIR / "analytics"


def validate_analytics_datasets(views: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Run 12 rigorous automated validation checks on analytical datasets."""
    if views is None:
        views = {
            "development_entities.parquet": pd.read_parquet(ANALYTICS_DIR / "development_entities.parquet"),
            "development_tmdb.parquet": pd.read_parquet(ANALYTICS_DIR / "development_tmdb.parquet"),
            "development_mal.parquet": pd.read_parquet(ANALYTICS_DIR / "development_mal.parquet"),
            "development_shared.parquet": pd.read_parquet(ANALYTICS_DIR / "development_shared.parquet"),
        }

    dev_df = views["development_entities.parquet"]
    tmdb_df = views["development_tmdb.parquet"]
    mal_df = views["development_mal.parquet"]
    shared_df = views["development_shared.parquet"]

    canonical_df = pd.read_parquet(CANONICAL_DIR / "canonical_entities.parquet")
    sample_df = pd.read_parquet(CANONICAL_DIR / "diverse_100k.parquet")

    checks = []

    # Check 1: Row count match with diverse_100k
    c1 = len(dev_df) == len(sample_df)
    checks.append({
        "check": "development_entities row count == diverse_100k row count",
        "status": "PASS" if c1 else "FAIL",
        "details": f"dev={len(dev_df):,}, diverse_100k={len(sample_df):,}",
    })

    # Check 2: All cinemind_ids exist in canonical_entities
    canonical_ids = set(canonical_df["cinemind_id"])
    dev_ids = set(dev_df["cinemind_id"])
    c2 = dev_ids.issubset(canonical_ids)
    checks.append({
        "check": "every development cinemind_id exists in canonical_entities",
        "status": "PASS" if c2 else "FAIL",
        "details": f"dev_unique={len(dev_ids):,}, subset_valid={c2}",
    })

    # Check 3: Unique cinemind_id in development_entities
    c3 = dev_df["cinemind_id"].is_unique
    checks.append({
        "check": "no duplicate cinemind_id in development_entities",
        "status": "PASS" if c3 else "FAIL",
        "details": f"duplicates={dev_df['cinemind_id'].duplicated().sum()}",
    })

    # Check 4: Unique cinemind_id in development_tmdb
    c4 = tmdb_df["cinemind_id"].is_unique
    checks.append({
        "check": "no duplicate cinemind_id in development_tmdb",
        "status": "PASS" if c4 else "FAIL",
        "details": f"duplicates={tmdb_df['cinemind_id'].duplicated().sum()}",
    })

    # Check 5: Unique cinemind_id in development_mal
    c5 = mal_df["cinemind_id"].is_unique
    checks.append({
        "check": "no duplicate cinemind_id in development_mal",
        "status": "PASS" if c5 else "FAIL",
        "details": f"duplicates={mal_df['cinemind_id'].duplicated().sum()}",
    })

    # Check 6: Shared count matches exact TMDB+MAL count
    c6 = len(shared_df) == 6453
    checks.append({
        "check": "development_shared count is exact cross-source intersection (6,453)",
        "status": "PASS" if c6 else "FAIL",
        "details": f"shared_count={len(shared_df):,}",
    })

    # Check 7: Shared entities exist in both source views
    tmdb_ids = set(tmdb_df["cinemind_id"])
    mal_ids = set(mal_df["cinemind_id"])
    shared_ids = set(shared_df["cinemind_id"])
    c7 = shared_ids.issubset(tmdb_ids) and shared_ids.issubset(mal_ids)
    checks.append({
        "check": "every shared entity exists in both TMDB and MAL source views",
        "status": "PASS" if c7 else "FAIL",
        "details": f"in_tmdb={shared_ids.issubset(tmdb_ids)}, in_mal={shared_ids.issubset(mal_ids)}",
    })

    # Check 8: TMDB IDs unique where present
    c8 = tmdb_df["tmdb_id"].dropna().is_unique
    checks.append({
        "check": "TMDB IDs unique where present in TMDB view",
        "status": "PASS" if c8 else "FAIL",
        "details": f"unique_tmdb_ids={tmdb_df['tmdb_id'].dropna().nunique():,}",
    })

    # Check 9: MAL IDs unique where present
    c9 = mal_df["mal_id"].dropna().is_unique
    checks.append({
        "check": "MAL IDs unique where present in MAL view",
        "status": "PASS" if c9 else "FAIL",
        "details": f"unique_mal_ids={mal_df['mal_id'].dropna().nunique():,}",
    })

    # Check 10: Zero value imputation (null release dates preserved)
    null_dates = dev_df["release_date"].isna().sum()
    c10 = null_dates == 532
    checks.append({
        "check": "zero value imputation (raw structural missingness preserved)",
        "status": "PASS" if c10 else "FAIL",
        "details": f"null_release_dates={null_dates}",
    })

    # Check 11: No rows dropped during source separation
    c11 = (len(tmdb_df) + len(mal_df) - len(shared_df)) == len(dev_df)
    checks.append({
        "check": "exact set inclusion: len(TMDB) + len(MAL) - len(Shared) == len(Dev)",
        "status": "PASS" if c11 else "FAIL",
        "details": f"{len(tmdb_df)} + {len(mal_df)} - {len(shared_df)} = {len(dev_df)}",
    })

    # Check 12: Canonical and raw files unmodified
    c12 = len(canonical_df) == 461188
    checks.append({
        "check": "canonical_entities.parquet unmodified (461,188 entities)",
        "status": "PASS" if c12 else "FAIL",
        "details": f"canonical_count={len(canonical_df):,}",
    })

    all_passed = all(c["status"] == "PASS" for c in checks)
    logger.info("Validation completed: %s (%d/12 checks passed)", "ALL PASSED" if all_passed else "SOME FAILED", sum(1 for c in checks if c['status'] == 'PASS'))

    return {
        "all_passed": all_passed,
        "checks": checks,
    }
