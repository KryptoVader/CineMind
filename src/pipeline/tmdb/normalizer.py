"""
TMDB record normalizer.

Reads raw JSONL from discovery, deduplicates by source_id,
maps genre_ids → genre names, and writes a unified Parquet file.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import RAW_TMDB_DIR, STAGING_DIR
from pipeline.tmdb.client import TMDBClient

logger = logging.getLogger(__name__)


def _normalize_record(
    raw: dict[str, Any],
    genre_map: dict[int, str],
) -> dict[str, Any]:
    """Convert one raw discover record to the normalized schema."""
    media_type = raw.get("_media_type", "movie")

    # Title handling differs between movie / TV
    if media_type == "movie":
        title = raw.get("title", "")
        original_title = raw.get("original_title", "")
        release_date = raw.get("release_date", "")
    else:
        title = raw.get("name", "")
        original_title = raw.get("original_name", "")
        release_date = raw.get("first_air_date", "")

    # Map genre IDs → names
    genre_ids = raw.get("genre_ids", [])
    genres = [
        genre_map.get(gid, f"Unknown({gid})")
        for gid in genre_ids
        if gid is not None
    ]

    # Extract year
    release_year: int | None = None
    if release_date and len(release_date) >= 4:
        try:
            release_year = int(release_date[:4])
        except ValueError:
            pass

    return {
        "source": "tmdb",
        "source_id": raw.get("id"),
        "title": title,
        "original_title": original_title,
        "alternative_titles": [],
        "media_type": media_type,
        "source_media_type": media_type,
        "release_date": release_date or None,
        "release_year": release_year,
        "end_date": None,
        "genres": genres,
        "original_language": raw.get("original_language"),
        "origin_country": raw.get("origin_country", []),
        "overview": raw.get("overview", ""),
        "rating": raw.get("vote_average"),
        "vote_count": raw.get("vote_count"),
        "popularity": raw.get("popularity"),
        "rank": None,
        "favorites": None,
        "num_list_users": None,
        "studios": [],
        "production_companies": [],
        "production_countries": [],
        "status": None,
        "runtime": None,
        "num_episodes": None,
        "source_material": None,
        "discovered_from": [raw.get("_strategy", "unknown")],
    }


def _read_jsonl_dedup(
    path: Path,
    genre_map: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Read a JSONL file, deduplicate by id, and normalize.

    If an ID appears multiple times (from different strategies),
    the first occurrence is kept and all strategy names are merged.
    """
    by_id: dict[int, dict[str, Any]] = {}
    strategies_by_id: dict[int, list[str]] = {}

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as fh:
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

    # Normalize
    records = []
    for sid, raw in by_id.items():
        record = _normalize_record(raw, genre_map)
        record["discovered_from"] = strategies_by_id.get(sid, [])
        records.append(record)

    return records


def normalize_tmdb() -> pd.DataFrame:
    """
    Read all raw TMDB discovery JSONL, normalize, deduplicate,
    and write to staging Parquet.

    Returns the resulting DataFrame.
    """
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGING_DIR / "tmdb_normalized.parquet"

    # Load genre map
    client = TMDBClient()
    genre_map = client.load_genre_map()
    logger.info("Genre map: %d genres", len(genre_map))

    # Read and normalize movies
    movies_path = RAW_TMDB_DIR / "discovery_movies.jsonl"
    tv_path = RAW_TMDB_DIR / "discovery_tv.jsonl"

    logger.info("Reading TMDB movies JSONL...")
    movie_records = _read_jsonl_dedup(movies_path, genre_map)
    logger.info("  %d unique movies", len(movie_records))

    logger.info("Reading TMDB TV JSONL...")
    tv_records = _read_jsonl_dedup(tv_path, genre_map)
    logger.info("  %d unique TV shows", len(tv_records))

    all_records = movie_records + tv_records

    if not all_records:
        logger.warning("No TMDB records found.")
        df = pd.DataFrame()
        df.to_parquet(output_path, index=False)
        return df

    df = pd.DataFrame(all_records)

    # Final dedup by source_id (should already be unique)
    before = len(df)
    df = df.drop_duplicates(subset=["source", "source_id"])
    after = len(df)
    if before != after:
        logger.info(
            "Removed %d duplicate source_ids", before - after
        )

    df.to_parquet(output_path, index=False)
    logger.info(
        "TMDB normalized: %d records → %s",
        len(df), output_path,
    )

    return df
