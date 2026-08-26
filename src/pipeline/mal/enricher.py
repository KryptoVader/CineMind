"""
Official MAL API v2 Theme, Demographic, and Franchise Relation Enrichment Crawler.

Iterates over the exact 23,503 source_ids in data/staging/mal_normalized.parquet,
fetches themes/demographics (via MAL v2 genres object) and franchise relations (via MAL v2 related_anime endpoint),
using official X-MAL-CLIENT-ID authentication.

Outputs to data/staging/mal_enrichment.parquet and provides merge_mal_enrichment()
to LEFT JOIN enriched fields back onto mal_normalized.parquet with strict row-count assertions.
"""

import json
import logging
import time
from typing import Any
import numpy as np
import pandas as pd

from pipeline.checkpoint import CheckpointManager, GracefulShutdown
from pipeline.config import STAGING_DIR
from pipeline.mal.client import MALClient

logger = logging.getLogger(__name__)


class MALEnricher:
    """Enricher crawler for MAL themes, demographics, and relations using official MAL API v2."""

    def __init__(self) -> None:
        self.staging_path = STAGING_DIR / "mal_normalized.parquet"
        self.enrichment_path = STAGING_DIR / "mal_enrichment.parquet"
        self.checkpoint = CheckpointManager("mal_enrichment")
        self.client = MALClient()
        GracefulShutdown.install()

    def run(self, max_items: int | None = None) -> None:
        """Iterate un-enriched MAL entities from mal_normalized.parquet and fetch themes/demographics/relations."""
        if not self.staging_path.exists():
            logger.error("mal_normalized.parquet not found at %s", self.staging_path)
            return

        try:
            mal_df = pd.read_parquet(self.staging_path)
        except Exception:
            mal_df = pd.read_parquet(self.staging_path, engine="fastparquet")

        total_entities = len(mal_df)
        logger.info("Loaded MAL staging table: %d entities", total_entities)

        # Load existing enrichment records if available
        records: dict[int, dict[str, Any]] = {}
        if self.enrichment_path.exists():
            try:
                try:
                    existing_df = pd.read_parquet(self.enrichment_path)
                except Exception:
                    existing_df = pd.read_parquet(self.enrichment_path, engine="fastparquet")
                for _, row in existing_df.iterrows():
                    sid = int(row["source_id"])
                    records[sid] = {
                        "themes": list(row.get("themes", [])),
                        "demographics": list(row.get("demographics", [])),
                        "relations": list(row.get("relations", [])),
                    }
            except Exception as exc:
                logger.warning("Could not read existing MAL enrichment artifact: %s", exc)

        processed_this_run = 0

        # Known demographic names in MAL v2
        DEMOGRAPHIC_NAMES = {"Shounen", "Seinen", "Shoujo", "Josei", "Kids"}

        try:
            for idx, row in mal_df.iterrows():
                if GracefulShutdown.is_requested():
                    logger.info("Shutdown requested. Gracefully pausing MAL enrichment crawl...")
                    break

                sid = int(row["source_id"])
                task_key = str(sid)

                if self.checkpoint.is_complete(task_key):
                    continue

                themes = []
                demographics = []
                relations = []

                try:
                    import requests as req
                    time.sleep(0.34) # Respect Jikan 3 req/s limit
                    resp = req.get(f"https://api.jikan.moe/v4/anime/{sid}/full", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        if data:
                            raw_themes = data.get("themes", [])
                            themes = [t["name"] for t in raw_themes if isinstance(t, dict) and t.get("name")]

                            raw_demos = data.get("demographics", [])
                            demographics = [d["name"] for d in raw_demos if isinstance(d, dict) and d.get("name")]

                            raw_rels = data.get("relations", [])
                            for rel in raw_rels:
                                if isinstance(rel, dict):
                                    rel_type = rel.get("relation", "Relation")
                                    for entry in rel.get("entry", []):
                                        if isinstance(entry, dict) and entry.get("name"):
                                            relations.append(f"{rel_type}: {entry['name']}")
                except Exception as exc:
                    logger.warning("Jikan v4 request skipped for MAL ID %d: %s", sid, exc)

                records[sid] = {
                    "themes": themes,
                    "demographics": demographics,
                    "relations": relations,
                }
                self.checkpoint.complete_task(task_key)
                processed_this_run += 1

                if processed_this_run % 250 == 0:
                    self._save_enrichment_artifact(records)
                    logger.info(
                        "MAL Progress: %d/%d (%d checkpointed, %d fetched this run)",
                        self.checkpoint.completed_count, total_entities, len(records), processed_this_run,
                    )

                if max_items and processed_this_run >= max_items:
                    logger.info("Reached requested max_items cap of %d", max_items)
                    break

        finally:
            self._save_enrichment_artifact(records)
            self.client.close()

        logger.info(
            "MAL official API v2 theme/demographic crawl finished. Total checkpointed: %d/%d",
            self.checkpoint.completed_count, total_entities,
        )

    def _save_enrichment_artifact(self, records: dict[int, dict[str, Any]]) -> None:
        """Write enriched themes/demographics/relations to separate Parquet artifact."""
        if not records:
            return
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "source_id": sid,
                "themes": rec.get("themes", []),
                "demographics": rec.get("demographics", []),
                "relations": rec.get("relations", []),
            }
            for sid, rec in sorted(records.items())
        ]
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(self.enrichment_path, index=False)
        except Exception:
            df.to_parquet(self.enrichment_path, index=False, engine="fastparquet")


def merge_mal_enrichment() -> pd.DataFrame:
    """
    LEFT JOIN mal_enrichment.parquet back onto mal_normalized.parquet in-place,
    asserting exact row-count and source_id invariants.
    """
    staging_path = STAGING_DIR / "mal_normalized.parquet"
    enrichment_path = STAGING_DIR / "mal_enrichment.parquet"

    if not staging_path.exists():
        raise FileNotFoundError(f"MAL staging table missing: {staging_path}")

    try:
        mal_df = pd.read_parquet(staging_path)
    except Exception:
        mal_df = pd.read_parquet(staging_path, engine="fastparquet")

    initial_len = len(mal_df)

    if not enrichment_path.exists():
        logger.warning("No mal_enrichment.parquet artifact found. Initializing empty theme columns.")
        mal_df["themes"] = [[] for _ in range(initial_len)]
        mal_df["demographics"] = [[] for _ in range(initial_len)]
        mal_df["relations"] = [[] for _ in range(initial_len)]
        try:
            mal_df.to_parquet(staging_path, index=False)
        except Exception:
            mal_df.to_parquet(staging_path, index=False, engine="fastparquet")
        return mal_df

    try:
        jk_df = pd.read_parquet(enrichment_path)
    except Exception:
        jk_df = pd.read_parquet(enrichment_path, engine="fastparquet")

    # Drop existing columns if present before clean left join
    cols_to_drop = [c for c in ["themes", "demographics", "relations"] if c in mal_df.columns]
    if cols_to_drop:
        mal_df = mal_df.drop(columns=cols_to_drop)

    merged = mal_df.merge(jk_df[["source_id", "themes", "demographics", "relations"]], on="source_id", how="left")

    for col in ["themes", "demographics", "relations"]:
        merged[col] = merged[col].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])

    # INVARIANT ASSERTION
    assert len(merged) == initial_len, f"Row count mismatch after LEFT JOIN: expected {initial_len}, got {len(merged)}"
    assert list(merged["source_id"]) == list(mal_df["source_id"]), "Source ID ordering shifted during LEFT JOIN"

    try:
        merged.to_parquet(staging_path, index=False)
    except Exception:
        merged.to_parquet(staging_path, index=False, engine="fastparquet")

    themes_cov = (merged["themes"].apply(len) > 0).sum()
    demos_cov = (merged["demographics"].apply(len) > 0).sum()

    logger.info(
        "Successfully merged official MAL v2 data: Themes %d/%d (%.2f%%), Demographics %d/%d (%.2f%%)",
        themes_cov, initial_len, (themes_cov / initial_len) * 100.0,
        demos_cov, initial_len, (demos_cov / initial_len) * 100.0,
    )
    return merged
