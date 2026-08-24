"""
Cross-source entity resolution (TMDB ↔ MAL).

Multi-signal matching pipeline with anti-overmerging safeguards:
  1. Exact normalized title + compatible year + compatible media type → HIGH
  2. Alternate / Japanese title match + compatible year               → MEDIUM
  3. Fuzzy title match (similarity > 0.85) + compatible year          → LOW (saved to match_candidates, NOT merged)

Anti-overmerging safeguards prevent merging sequels, seasons, remakes, movies, OVAs, and specials.
"""

import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import CANONICAL_DIR, STAGING_DIR

logger = logging.getLogger(__name__)


# =============================================================
# Helper Utilities & Normalization
# =============================================================

def normalize_genres(val: Any) -> list[str]:
    """
    Robustly convert any container/scalar genre representation into a clean list of strings.
    Handles: list, tuple, np.ndarray, pd.Series, json-string, scalar string, None/NaN.
    """
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        res = []
        for item in val:
            if item and isinstance(item, str) and item.strip():
                res.append(item.strip())
            elif item and not isinstance(item, (list, tuple, dict)) and pd.notna(item):
                res.append(str(item).strip())
        return res
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except Exception:
                pass
        return [val]
    return []


def normalize_title(title: str | None) -> str:
    """
    Normalize a title for comparison.
    - Unicode NFKC normalization
    - Lowercase
    - Replace punctuation with spaces (preserving digits)
    - Collapse whitespace
    """
    if not title:
        return ""
    title = unicodedata.normalize("NFKC", title)
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_title_modifiers(title: str) -> set[str]:
    """
    Extract title modifiers like season numbers, part numbers, movie/ova tags
    to prevent over-merging franchise entries.
    """
    t_norm = normalize_title(title)
    modifiers = set()
    patterns = [
        (r"\bseason\s*(\d+)\b", "season_{}"),
        (r"\bpart\s*(\d+)\b", "part_{}"),
        (r"\b(\d+)(st|nd|rd|th)\s*season\b", "season_{}"),
        (r"\bmovie\b", "movie"),
        (r"\bova\b", "ova"),
        (r"\bona\b", "ona"),
        (r"\bspecial\b", "special"),
        (r"\bbrotherhood\b", "brotherhood"),
        (r"\bshippuden\b", "shippuden"),
        (r"\bfinal\b", "final"),
        (r"\b(\d+)\b", "num_{}"),
    ]
    for pattern, tag in patterns:
        matches = re.findall(pattern, t_norm)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            modifiers.add(tag.format(m) if "{}" in tag else tag)
    return modifiers


def years_compatible(y1: Any, y2: Any, tolerance: int = 1) -> bool:
    """Check if two release years are within tolerance."""
    if y1 is None or y2 is None:
        return False
    try:
        return abs(int(y1) - int(y2)) <= tolerance
    except (ValueError, TypeError):
        return False


def media_types_compatible(tmdb_mt: str, mal_mt: str) -> bool:
    """Check if TMDB and MAL media types could refer to the same work."""
    compat = {
        ("movie", "anime_movie"),
        ("tv", "anime_tv"),
        ("tv", "ona"),
        ("tv", "special"),
        ("tv", "ova"),
    }
    return (tmdb_mt, mal_mt) in compat


def _get_all_titles(row: pd.Series) -> list[str]:
    """Collect all title variants for a row."""
    titles = []
    for col in ["title", "original_title"]:
        val = row.get(col)
        if val and isinstance(val, str) and val.strip():
            titles.append(val)
    alt = row.get("alternative_titles")
    if isinstance(alt, (list, tuple, np.ndarray)):
        for t in alt:
            if isinstance(t, str) and t.strip():
                titles.append(t)
    return titles


# =============================================================
# Core Resolution Pipeline
# =============================================================

def run_cross_source_matching() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match TMDB animation/Japanese titles against MAL titles.

    Returns:
        (matches_df, unresolved_df)
    """
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = CANONICAL_DIR / "entity_links.parquet"
    unresolved_path = CANONICAL_DIR / "match_candidates.parquet"

    tmdb_path = STAGING_DIR / "tmdb_normalized.parquet"
    mal_path = STAGING_DIR / "mal_normalized.parquet"

    if not tmdb_path.exists() or not mal_path.exists():
        logger.warning("Need both TMDB and MAL normalized files.")
        empty = pd.DataFrame()
        empty.to_parquet(matches_path, index=False)
        empty.to_parquet(unresolved_path, index=False)
        return empty, empty

    tmdb_df = pd.read_parquet(tmdb_path)
    mal_df = pd.read_parquet(mal_path)

    # --- Filter TMDB to Animation or Japanese content ---
    def is_anim_candidate(row: pd.Series) -> bool:
        genres = normalize_genres(row.get("genres"))
        if "Animation" in genres:
            return True
        lang = str(row.get("original_language", "")).lower()
        if lang == "ja":
            return True
        countries = row.get("origin_country")
        if isinstance(countries, (list, tuple, np.ndarray)) and "JP" in countries:
            return True
        return False

    tmdb_anim = tmdb_df[tmdb_df.apply(is_anim_candidate, axis=1)].copy()

    logger.info(
        "Cross-source matching: %d TMDB candidates ↔ %d MAL candidates",
        len(tmdb_anim), len(mal_df),
    )

    if tmdb_anim.empty or mal_df.empty:
        empty = pd.DataFrame()
        empty.to_parquet(matches_path, index=False)
        empty.to_parquet(unresolved_path, index=False)
        return empty, empty

    # Index MAL titles
    mal_title_index: dict[str, list[int]] = {}
    mal_norm_titles: dict[int, list[str]] = {}

    for idx in mal_df.index:
        row = mal_df.loc[idx]
        all_titles = _get_all_titles(row)
        norm_titles = []
        for t in all_titles:
            nt = normalize_title(t)
            if nt:
                norm_titles.append(nt)
                mal_title_index.setdefault(nt, []).append(idx)
        mal_norm_titles[idx] = norm_titles

    logger.info(
        "MAL title index: %d unique normalized titles",
        len(mal_title_index),
    )

    matches: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    matched_mal_ids: set[int] = set()
    matched_tmdb_ids: set[int] = set()

    # Pre-extract MAL records for instant prefix+year candidate lookup
    mal_records_by_idx = {}
    mal_by_prefix_year: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for idx, row in mal_df.iterrows():
        y = row.get("release_year")
        try:
            y_int = int(y) if pd.notna(y) else None
        except (ValueError, TypeError):
            y_int = None

        rec = {
            "idx": idx,
            "id": int(row["source_id"]),
            "title": row.get("title", ""),
            "original_title": row.get("original_title", ""),
            "year": y_int,
            "media_type": row.get("media_type", ""),
            "norm_titles": mal_norm_titles.get(idx, []),
        }
        mal_records_by_idx[idx] = rec

        if y_int is not None:
            for nt in rec["norm_titles"]:
                if len(nt) >= 3:
                    prefix = nt[:3]
                    for yr in (y_int - 1, y_int, y_int + 1):
                        mal_by_prefix_year.setdefault((prefix, yr), []).append(rec)

    for _, tmdb_row in tmdb_anim.iterrows():
        tmdb_id = tmdb_row["source_id"]
        if tmdb_id in matched_tmdb_ids:
            continue

        tmdb_titles = _get_all_titles(tmdb_row)
        tmdb_norms = [normalize_title(t) for t in tmdb_titles if normalize_title(t)]
        tmdb_year = tmdb_row.get("release_year")
        tmdb_mt = tmdb_row.get("media_type", "")

        best_match: dict[str, Any] | None = None
        best_confidence = ""

        for tn in tmdb_norms:
            if tn not in mal_title_index:
                continue

            for mal_idx in mal_title_index[tn]:
                mal_rec = mal_records_by_idx[mal_idx]
                mal_id = mal_rec["id"]
                mal_year = mal_rec["year"]
                mal_mt = mal_rec["media_type"]

                if mal_id in matched_mal_ids:
                    continue

                # Anti-overmerging safeguard
                t_mods = extract_title_modifiers(tmdb_row.get("title", ""))
                m_mods = extract_title_modifiers(mal_rec["title"])
                if t_mods != m_mods:
                    confidence = "low"
                else:
                    yr_ok = years_compatible(tmdb_year, mal_year, tolerance=1)
                    mt_ok = media_types_compatible(tmdb_mt, mal_mt)

                    if yr_ok and mt_ok:
                        confidence = "high"
                    elif yr_ok:
                        confidence = "medium"
                    else:
                        confidence = "low"

                match_info = {
                    "tmdb_id": int(tmdb_id),
                    "mal_id": int(mal_id),
                    "tmdb_title": tmdb_row.get("title", ""),
                    "mal_title": mal_rec["title"],
                    "tmdb_original_title": tmdb_row.get("original_title", ""),
                    "mal_original_title": mal_rec["original_title"],
                    "tmdb_year": tmdb_year,
                    "mal_year": mal_year,
                    "tmdb_media_type": tmdb_mt,
                    "mal_media_type": mal_mt,
                    "match_confidence": confidence,
                    "match_method": "exact_title",
                    "matched_title": tn,
                    "similarity": 1.0,
                }

                conf_rank = {"high": 3, "medium": 2, "low": 1}
                if (
                    best_match is None
                    or conf_rank.get(confidence, 0) > conf_rank.get(best_confidence, 0)
                ):
                    best_match = match_info
                    best_confidence = confidence

        if best_match:
            if best_confidence in ("high", "medium"):
                matches.append(best_match)
                matched_tmdb_ids.add(tmdb_id)
                matched_mal_ids.add(best_match["mal_id"])
            else:
                unresolved.append(best_match)

    logger.info(
        "Exact title matching: %d matches (HIGH/MEDIUM), %d candidates (LOW)",
        len(matches), len(unresolved),
    )

    # --- Fast Prefix+Year Blocked Fuzzy matching ---
    unmatched_tmdb = tmdb_anim[~tmdb_anim["source_id"].isin(matched_tmdb_ids)]
    unmatched_with_year = unmatched_tmdb[unmatched_tmdb["release_year"].notna()]

    logger.info(
        "Fuzzy matching: checking %d remaining TMDB candidates",
        len(unmatched_with_year),
    )

    fuzzy_count = 0
    for _, tmdb_row in unmatched_with_year.iterrows():
        tmdb_id = tmdb_row["source_id"]
        tmdb_norms = [normalize_title(t) for t in _get_all_titles(tmdb_row) if normalize_title(t)]
        try:
            tmdb_year = int(tmdb_row.get("release_year")) if pd.notna(tmdb_row.get("release_year")) else None
        except (ValueError, TypeError):
            tmdb_year = None
        tmdb_mt = tmdb_row.get("media_type", "")

        if not tmdb_norms or tmdb_year is None:
            continue

        best_sim = 0.0
        best_info: dict[str, Any] | None = None

        candidate_mal_records: list[dict[str, Any]] = []
        for tn in tmdb_norms:
            if len(tn) >= 3:
                p3 = tn[:3]
                candidate_mal_records.extend(mal_by_prefix_year.get((p3, tmdb_year), []))

        for mal_rec in candidate_mal_records:
            mal_id = mal_rec["id"]
            if mal_id in matched_mal_ids:
                continue

            for tn in tmdb_norms:
                for mn in mal_rec["norm_titles"]:
                    if abs(len(tn) - len(mn)) > 5:
                        continue
                    sim = SequenceMatcher(None, tn, mn).ratio()
                    if sim > best_sim and sim > 0.88:
                        best_sim = sim
                        mal_mt = mal_rec["media_type"]
                        mt_ok = media_types_compatible(tmdb_mt, mal_mt)

                        t_mods = extract_title_modifiers(tmdb_row.get("title", ""))
                        m_mods = extract_title_modifiers(mal_rec["title"])
                        
                        conf = "medium" if (mt_ok and t_mods == m_mods) else "low"

                        best_info = {
                            "tmdb_id": int(tmdb_id),
                            "mal_id": int(mal_id),
                            "tmdb_title": tmdb_row.get("title", ""),
                            "mal_title": mal_rec["title"],
                            "tmdb_original_title": tmdb_row.get("original_title", ""),
                            "mal_original_title": mal_rec["original_title"],
                            "tmdb_year": tmdb_year,
                            "mal_year": mal_rec["year"],
                            "tmdb_media_type": tmdb_mt,
                            "mal_media_type": mal_mt,
                            "match_confidence": conf,
                            "match_method": "fuzzy_title",
                            "matched_title": f"{tn} ↔ {mn}",
                            "similarity": round(sim, 4),
                        }

        if best_info:
            fuzzy_count += 1
            if best_info["match_confidence"] == "medium":
                matches.append(best_info)
                matched_tmdb_ids.add(tmdb_id)
                matched_mal_ids.add(best_info["mal_id"])
            else:
                unresolved.append(best_info)

    logger.info(
        "Fuzzy matching: %d additional candidates found", fuzzy_count,
    )

    matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()
    unresolved_df = pd.DataFrame(unresolved) if unresolved else pd.DataFrame()

    matches_df.to_parquet(matches_path, index=False)
    unresolved_df.to_parquet(unresolved_path, index=False)

    logger.info(
        "Cross-source matching complete: %d verified links → %s, %d LOW candidates → %s",
        len(matches_df), matches_path, len(unresolved_df), unresolved_path,
    )

    return matches_df, unresolved_df

