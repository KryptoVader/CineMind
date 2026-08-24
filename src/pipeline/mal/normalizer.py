"""
MAL record normalizer.

Reads raw JSONL from discovery, deduplicates by MAL ID,
normalizes fields into the common schema, and writes Parquet.
"""

import json
import logging
from typing import Any

import pandas as pd

from pipeline.config import RAW_MAL_DIR, STAGING_DIR

logger = logging.getLogger(__name__)


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw MAL record to the normalized schema."""

    # --- Title ---
    title = raw.get("title", "")
    alt_data = raw.get("alternative_titles", {})
    if isinstance(alt_data, dict):
        original_title = alt_data.get("ja", "") or title
        alt_titles: list[str] = []
        if alt_data.get("en"):
            alt_titles.append(alt_data["en"])
        alt_titles.extend(alt_data.get("synonyms", []))
    else:
        original_title = title
        alt_titles = []

    # --- Media type ---
    mal_media_type = raw.get("media_type", "unknown")
    media_type_map = {
        "tv": "anime_tv",
        "movie": "anime_movie",
        "ova": "ova",
        "ona": "ona",
        "special": "special",
        "music": "music",
        "unknown": "unknown",
    }
    media_type = media_type_map.get(mal_media_type, mal_media_type)

    # --- Dates ---
    start_date = raw.get("start_date", "") or ""
    end_date = raw.get("end_date", "") or ""

    release_year: int | None = None
    if start_date and len(start_date) >= 4:
        try:
            release_year = int(start_date[:4])
        except ValueError:
            pass

    # Fallback to start_season
    if release_year is None:
        season = raw.get("start_season")
        if isinstance(season, dict) and season.get("year"):
            try:
                release_year = int(season["year"])
            except (ValueError, TypeError):
                pass

    # --- Genres ---
    genres_raw = raw.get("genres", [])
    genres = [
        g.get("name", "")
        for g in genres_raw
        if isinstance(g, dict) and g.get("name")
    ]

    # --- Studios ---
    studios_raw = raw.get("studios", [])
    studios = [
        s.get("name", "")
        for s in studios_raw
        if isinstance(s, dict) and s.get("name")
    ]

    # --- Discovery provenance ---
    discovered_from = []
    strategy = raw.get("_strategy")
    if strategy:
        discovered_from.append(strategy)

    return {
        "source": "mal",
        "source_id": raw.get("id"),
        "title": title,
        "original_title": original_title,
        "alternative_titles": alt_titles,
        "media_type": media_type,
        "source_media_type": mal_media_type,
        "release_date": start_date or None,
        "release_year": release_year,
        "end_date": end_date or None,
        "genres": genres,
        "original_language": "ja",
        "origin_country": ["JP"],
        "overview": raw.get("synopsis", "") or "",
        "rating": raw.get("mean"),
        "vote_count": raw.get("num_scoring_users"),
        "popularity": raw.get("popularity"),
        "rank": raw.get("rank"),
        "favorites": raw.get("num_favorites"),
        "num_list_users": raw.get("num_list_users"),
        "studios": studios,
        "production_companies": [],
        "production_countries": ["Japan"],
        "status": raw.get("status"),
        "runtime": raw.get("average_episode_duration"),
        "num_episodes": raw.get("num_episodes"),
        "source_material": raw.get("source"),
        "discovered_from": discovered_from,
    }


def normalize_mal() -> pd.DataFrame:
    """
    Read raw MAL JSONL, normalize, deduplicate by source_id,
    and write to staging Parquet.
    """
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGING_DIR / "mal_normalized.parquet"

    raw_path = RAW_MAL_DIR / "discovery.jsonl"
    if not raw_path.exists():
        logger.warning("No MAL discovery JSONL found.")
        df = pd.DataFrame()
        df.to_parquet(output_path, index=False)
        return df

    # Read, dedup, merge strategies
    by_id: dict[int, dict[str, Any]] = {}
    strategies_by_id: dict[int, list[str]] = {}

    logger.info("Reading MAL discovery JSONL...")
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            sid = raw.get("id")
            if sid is None:
                continue

            strategy = raw.get("_strategy", "unknown")
            if sid not in by_id:
                by_id[sid] = raw
                strategies_by_id[sid] = [strategy]
            else:
                if strategy not in strategies_by_id[sid]:
                    strategies_by_id[sid].append(strategy)

    logger.info("  %d unique MAL IDs", len(by_id))

    # Normalize
    records = []
    for sid, raw in by_id.items():
        record = _normalize_record(raw)
        record["discovered_from"] = strategies_by_id.get(sid, [])
        records.append(record)

    if not records:
        logger.warning("No MAL records after normalization.")
        df = pd.DataFrame()
        df.to_parquet(output_path, index=False)
        return df

    df = pd.DataFrame(records)

    # Final dedup (should already be unique)
    before = len(df)
    df = df.drop_duplicates(subset=["source", "source_id"])
    after = len(df)
    if before != after:
        logger.info(
            "Removed %d duplicate source_ids", before - after,
        )

    df.to_parquet(output_path, index=False)
    logger.info(
        "MAL normalized: %d records → %s",
        len(df), output_path,
    )

    return df
