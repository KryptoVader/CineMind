"""
TMDB Keyword Enrichment Crawler.

Iterates over the exact 444,138 source_ids in data/staging/tmdb_normalized.parquet,
fetches keywords via /movie/{id}/keywords and /tv/{id}/keywords, checkpointing state
cooperatively.

Outputs to data/staging/tmdb_keywords.parquet and provides merge_tmdb_keywords()
to LEFT JOIN enriched keywords back onto tmdb_normalized.parquet with strict row-count assertions.
"""

import json
import logging
from typing import Any
import numpy as np
import pandas as pd

from pipeline.checkpoint import CheckpointManager, GracefulShutdown
from pipeline.config import STAGING_DIR
from pipeline.tmdb.client import TMDBClient

logger = logging.getLogger(__name__)


class TMDBKeywordEnricher:
    """Enricher crawler for TMDB keywords with checkpointing and atomic writes."""

    def __init__(self) -> None:
        self.staging_path = STAGING_DIR / "tmdb_normalized.parquet"
        self.enrichment_path = STAGING_DIR / "tmdb_keywords.parquet"
        self.checkpoint = CheckpointManager("tmdb_keywords")
        GracefulShutdown.install()

    def run(self, max_items: int | None = None) -> None:
        """Iterate un-enriched TMDB entities from tmdb_normalized.parquet and fetch keywords."""
        if not self.staging_path.exists():
            logger.error("tmdb_normalized.parquet not found at %s", self.staging_path)
            return

        try:
            tmdb_df = pd.read_parquet(self.staging_path)
        except Exception:
            tmdb_df = pd.read_parquet(self.staging_path, engine="fastparquet")

        total_entities = len(tmdb_df)
        logger.info("Loaded TMDB staging table: %d entities", total_entities)

        # Load existing enrichment records if available
        records: dict[int, list[str]] = {}
        if self.enrichment_path.exists():
            try:
                try:
                    existing_df = pd.read_parquet(self.enrichment_path)
                except Exception:
                    existing_df = pd.read_parquet(self.enrichment_path, engine="fastparquet")
                for _, row in existing_df.iterrows():
                    sid = int(row["source_id"])
                    kws = row["keywords"]
                    records[sid] = list(kws) if isinstance(kws, (list, tuple)) else []
            except Exception as exc:
                logger.warning("Could not read existing keywords artifact: %s", exc)

        client = TMDBClient()
        processed_this_run = 0

        try:
            for idx, row in tmdb_df.iterrows():
                if GracefulShutdown.is_requested():
                    logger.info("Shutdown requested. Gracefully pausing TMDB keyword crawl...")
                    break

                sid = int(row["source_id"])
                task_key = str(sid)

                if self.checkpoint.is_complete(task_key):
                    continue

                media_type = str(row.get("source_media_type", "movie")).lower()
                try:
                    if "tv" in media_type:
                        kws = client.get_tv_keywords(sid)
                    else:
                        kws = client.get_movie_keywords(sid)
                except Exception as exc:
                    logger.warning("Error fetching keywords for TMDB %d (%s): %s", sid, media_type, exc)
                    kws = []

                records[sid] = kws
                self.checkpoint.complete_task(task_key)
                processed_this_run += 1

                if processed_this_run % 250 == 0:
                    self._save_enrichment_artifact(records)
                    logger.info(
                        "Progress: %d/%d (%d checkpointed, %d fetched this run)",
                        self.checkpoint.completed_count, total_entities, len(records), processed_this_run,
                    )

                if max_items and processed_this_run >= max_items:
                    logger.info("Reached requested max_items cap of %d", max_items)
                    break

        finally:
            self._save_enrichment_artifact(records)
            client.close()

        logger.info(
            "TMDB keyword crawl finished. Total checkpointed: %d/%d",
            self.checkpoint.completed_count, total_entities,
        )

    def _save_enrichment_artifact(self, records: dict[int, list[str]]) -> None:
        """Write enriched keywords to separate Parquet artifact."""
        if not records:
            return
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        rows = [{"source_id": sid, "keywords": kws} for sid, kws in sorted(records.items())]
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(self.enrichment_path, index=False)
        except Exception:
            df.to_parquet(self.enrichment_path, index=False, engine="fastparquet")


def merge_tmdb_keywords() -> pd.DataFrame:
    """
    LEFT JOIN tmdb_keywords.parquet back onto tmdb_normalized.parquet in-place,
    asserting exact row-count and source_id invariants.
    """
    staging_path = STAGING_DIR / "tmdb_normalized.parquet"
    enrichment_path = STAGING_DIR / "tmdb_keywords.parquet"

    if not staging_path.exists():
        raise FileNotFoundError(f"TMDB staging table missing: {staging_path}")

    try:
        tmdb_df = pd.read_parquet(staging_path)
    except Exception:
        tmdb_df = pd.read_parquet(staging_path, engine="fastparquet")

    initial_len = len(tmdb_df)

    if not enrichment_path.exists():
        logger.warning("No tmdb_keywords.parquet artifact found. Initializing empty keywords column.")
        tmdb_df["keywords"] = [[] for _ in range(initial_len)]
        try:
            tmdb_df.to_parquet(staging_path, index=False)
        except Exception:
            tmdb_df.to_parquet(staging_path, index=False, engine="fastparquet")
        return tmdb_df

    try:
        kw_df = pd.read_parquet(enrichment_path)
    except Exception:
        kw_df = pd.read_parquet(enrichment_path, engine="fastparquet")

    # Drop existing keywords column if present to ensure clean left join
    if "keywords" in tmdb_df.columns:
        tmdb_df = tmdb_df.drop(columns=["keywords"])

    merged = tmdb_df.merge(kw_df[["source_id", "keywords"]], on="source_id", how="left")

    # Fill NaN keywords with empty list
    merged["keywords"] = merged["keywords"].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])

    # INVARIANT ASSERTION
    assert len(merged) == initial_len, f"Row count mismatch after LEFT JOIN: expected {initial_len}, got {len(merged)}"
    assert list(merged["source_id"]) == list(tmdb_df["source_id"]), "Source ID ordering shifted during LEFT JOIN"

    try:
        merged.to_parquet(staging_path, index=False)
    except Exception:
        merged.to_parquet(staging_path, index=False, engine="fastparquet")

    kw_populated = (merged["keywords"].apply(len) > 0).sum()
    logger.info(
        "Successfully merged TMDB keywords: %d/%d entities enriched (%.2f%% coverage)",
        kw_populated, initial_len, (kw_populated / initial_len) * 100.0,
    )
    return merged
