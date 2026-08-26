"""
Canonical entity builder.

Merges TMDB and MAL normalized datasets into a unified
canonical universe (canonical_entities.parquet) with deterministic
cinemind_id, tmdb_id, and mal_id fields.
"""

import logging
import numpy as np
import pandas as pd

from pipeline.config import CANONICAL_DIR, STAGING_DIR

logger = logging.getLogger(__name__)


def build_canonical_dataset() -> pd.DataFrame:
    """
    Build unified canonical entities from TMDB and MAL normalized datasets,
    applying verified cross-source identity links if present.
    """
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    canonical_output_path = CANONICAL_DIR / "canonical_entities.parquet"
    candidates_output_path = CANONICAL_DIR / "candidates.parquet"
    links_path = CANONICAL_DIR / "entity_links.parquet"

    # ---- Load normalized datasets ----
    tmdb_path = STAGING_DIR / "tmdb_normalized.parquet"
    mal_path = STAGING_DIR / "mal_normalized.parquet"

    tmdb_df = pd.read_parquet(tmdb_path) if tmdb_path.exists() else pd.DataFrame()
    mal_df = pd.read_parquet(mal_path) if mal_path.exists() else pd.DataFrame()

    if tmdb_df.empty and mal_df.empty:
        logger.error("No normalized data to build canonical dataset.")
        empty = pd.DataFrame()
        empty.to_parquet(canonical_output_path, index=False)
        empty.to_parquet(candidates_output_path, index=False)
        return empty

    # Initialize tmdb_id and mal_id columns
    if not tmdb_df.empty:
        tmdb_df["tmdb_id"] = tmdb_df["source_id"].astype("Int64")
        tmdb_df["mal_id"] = pd.Series(dtype="Int64")
    if not mal_df.empty:
        mal_df["mal_id"] = mal_df["source_id"].astype("Int64")
        mal_df["tmdb_id"] = pd.Series(dtype="Int64")

    # Load verified cross-source links if available
    linked_tmdb_ids: set[int] = set()
    linked_mal_ids: set[int] = set()
    merged_records: list[dict] = []

    if links_path.exists():
        links_df = pd.read_parquet(links_path)
        if not links_df.empty:
            logger.info("Applying %d verified cross-source links...", len(links_df))
            for _, link in links_df.iterrows():
                t_id = int(link["tmdb_id"])
                m_id = int(link["mal_id"])

                tmdb_match = tmdb_df[tmdb_df["source_id"] == t_id]
                mal_match = mal_df[mal_df["source_id"] == m_id]

                if not tmdb_match.empty and not mal_match.empty:
                    t_row = tmdb_match.iloc[0].to_dict()
                    m_row = mal_match.iloc[0].to_dict()

                    linked_tmdb_ids.add(t_id)
                    linked_mal_ids.add(m_id)

                    # Create merged canonical entity (preserving MAL anime media_type & TMDB details)
                    merged = dict(t_row)
                    merged["cinemind_id"] = f"cinemind_tmdb_{t_id}_mal_{m_id}"
                    merged["tmdb_id"] = t_id
                    merged["mal_id"] = m_id
                    merged["source"] = "tmdb+mal"
                    merged["source_presence"] = ["tmdb", "mal"]
                    merged["match_confidence"] = link.get("match_confidence", "high")

                    # Prefer MAL title/details if available
                    if m_row.get("media_type"):
                        merged["media_type"] = m_row["media_type"]
                    
                    merged_records.append(merged)

    # Filter unmerged TMDB and MAL records
    unmerged_tmdb = tmdb_df[~tmdb_df["source_id"].isin(linked_tmdb_ids)].copy() if not tmdb_df.empty else pd.DataFrame()
    unmerged_mal = mal_df[~mal_df["source_id"].isin(linked_mal_ids)].copy() if not mal_df.empty else pd.DataFrame()

    if not unmerged_tmdb.empty:
        unmerged_tmdb["cinemind_id"] = "cinemind_tmdb_" + unmerged_tmdb["source_id"].astype(str)
        unmerged_tmdb["source_presence"] = [["tmdb"]] * len(unmerged_tmdb)
        unmerged_tmdb["match_confidence"] = "unmatched"

    if not unmerged_mal.empty:
        unmerged_mal["cinemind_id"] = "cinemind_mal_" + unmerged_mal["source_id"].astype(str)
        unmerged_mal["source_presence"] = [["mal"]] * len(unmerged_mal)
        unmerged_mal["match_confidence"] = "unmatched"

    merged_df = pd.DataFrame(merged_records) if merged_records else pd.DataFrame()

    all_frames = [f for f in [unmerged_tmdb, unmerged_mal, merged_df] if not f.empty]
    canonical_df = pd.concat(all_frames, ignore_index=True)

    # Verify uniqueness
    dupes = canonical_df["cinemind_id"].duplicated().sum()
    if dupes > 0:
        logger.warning("Dropping %d duplicate cinemind_ids in canonical entity table", dupes)
        canonical_df = canonical_df.drop_duplicates(subset=["cinemind_id"])

    # Apply Genre Taxonomy Unification
    from pipeline.canonical.genre_map import unify_genres
    if "genres" in canonical_df.columns:
        canonical_df["genres"] = canonical_df["genres"].apply(unify_genres)

    # Ensure list columns default to empty lists rather than NaNs
    list_cols = ["keywords", "themes", "demographics", "relations", "alternative_titles", "production_companies", "production_countries"]
    for col in list_cols:
        if col in canonical_df.columns:
            canonical_df[col] = canonical_df[col].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])

    # INVARIANT ASSERTION: Row count must match expectation
    expected_len = len(tmdb_df) + len(mal_df) - len(linked_tmdb_ids)
    assert len(canonical_df) == expected_len, f"Canonical row count mismatch: expected {expected_len}, got {len(canonical_df)}"

    # Save canonical entity table and candidates parquet
    try:
        canonical_df.to_parquet(canonical_output_path, index=False)
        canonical_df.to_parquet(candidates_output_path, index=False)
    except Exception:
        canonical_df.to_parquet(canonical_output_path, index=False, engine="fastparquet")
        canonical_df.to_parquet(candidates_output_path, index=False, engine="fastparquet")

    logger.info(
        "Canonical Universe built: %d entities (%d TMDB-only, %d MAL-only, %d merged) → %s",
        len(canonical_df), len(unmerged_tmdb), len(unmerged_mal), len(merged_records), canonical_output_path,
    )

    return canonical_df

