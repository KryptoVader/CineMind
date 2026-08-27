"""
CineMind V2 — Structured Feature Predicates
Pure predicate functions for evaluating entity features deterministically.
"""
from dataclasses import dataclass
from typing import Callable, Any, Optional
import pandas as pd
import numpy as np
from cinemind_v2.knowledge.entity_store import EntityStore


@dataclass
class FeaturePredicate:
    feature_id: str
    category: str
    description: str
    eval_fn: Callable[[pd.DataFrame], np.ndarray]  # Returns boolean array of shape (N,)
    unknown_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None  # Optional boolean array where feature is unknown


def build_structured_predicates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build structured feature definitions over DataFrame.
    Returns list of dicts: {'feature_id': str, 'category': str, 'description': str, 'mask': np.ndarray, 'unknown_mask': np.ndarray}
    """
    predicates = []
    num_entities = len(df)

    # 1. Media Type Predicates
    media_types = [
        ("media:movie", "movie", "Is it a movie?"),
        ("media:tv", "tv", "Is it a TV series?"),
        ("media:anime", "anime", "Is it an anime series or anime movie?"),
    ]
    if "media_type" in df.columns:
        media_series = df["media_type"].astype(str)
    else:
        media_series = pd.Series([""] * num_entities)

    for fid, key, desc in media_types:
        if key == "anime":
            mask = media_series.str.contains("anime|ova|ona", case=False, na=False).values
        else:
            mask = media_series.str.contains(key, case=False, na=False).values
        predicates.append({
            "feature_id": fid,
            "category": "media_type",
            "description": desc,
            "mask": mask.astype(bool),
            "unknown_mask": np.zeros(num_entities, dtype=bool),
        })

    # 2. Release Decade Predicates
    decades = [
        ("decade:2020s", 2020, 2029, "Was it released in the 2020s (2020 or later)?"),
        ("decade:2010s", 2010, 2019, "Was it released in the 2010s (2010–2019)?"),
        ("decade:2000s", 2000, 2009, "Was it released in the 2000s (2000–2009)?"),
        ("decade:1990s", 1990, 1999, "Was it released in the 1990s (1990–1999)?"),
        ("decade:1980s", 1980, 1989, "Was it released in the 1980s (1980–1989)?"),
        ("decade:pre_1980", 1800, 1979, "Was it released before 1980?"),
    ]
    if "release_year" in df.columns:
        years = pd.to_numeric(df["release_year"], errors="coerce").values
    else:
        years = np.full(num_entities, np.nan)
    year_unknown = np.isnan(years)

    for fid, start_yr, end_yr, desc in decades:
        mask = (years >= start_yr) & (years <= end_yr)
        mask = np.where(year_unknown, False, mask)
        predicates.append({
            "feature_id": fid,
            "category": "decade",
            "description": desc,
            "mask": mask.astype(bool),
            "unknown_mask": year_unknown.astype(bool),
        })

    # 3. Original Language Predicates
    languages = [
        ("language:english", "en", "Is the original language English?"),
        ("language:japanese", "ja", "Is the original language Japanese?"),
        ("language:korean", "ko", "Is the original language Korean?"),
        ("language:chinese", "zh", "Is the original language Chinese?"),
        ("language:french", "fr", "Is the original language French?"),
        ("language:spanish", "es", "Is the original language Spanish?"),
        ("language:german", "de", "Is the original language German?"),
        ("language:hindi", "hi", "Is the original language Hindi?"),
    ]
    if "original_language" in df.columns:
        langs = df["original_language"].astype(str).str.lower().values
        lang_unknown = df["original_language"].isna().values
    else:
        langs = np.full(num_entities, "", dtype=object)
        lang_unknown = np.ones(num_entities, dtype=bool)

    for fid, code, desc in languages:
        mask = (langs == code)
        predicates.append({
            "feature_id": fid,
            "category": "language",
            "description": desc,
            "mask": mask.astype(bool),
            "unknown_mask": lang_unknown.astype(bool),
        })

    # 4. Rating Tier Predicates
    if "rating" in df.columns:
        ratings = pd.to_numeric(df["rating"], errors="coerce").values
    else:
        ratings = np.full(num_entities, np.nan)
    rating_unknown = np.isnan(ratings)

    rating_buckets = [
        ("rating:masterpiece", 8.5, 10.0, "Is it critically acclaimed (rated 8.5+)?"),
        ("rating:good", 7.5, 8.49, "Is it well-regarded (rated 7.5 to 8.4)?"),
        ("rating:average", 6.0, 7.49, "Is it average-rated (rated 6.0 to 7.4)?"),
        ("rating:poor", 0.0, 5.99, "Is it lower-rated (rated below 6.0)?"),
    ]
    for fid, min_r, max_r, desc in rating_buckets:
        mask = (ratings >= min_r) & (ratings <= max_r)
        mask = np.where(rating_unknown, False, mask)
        predicates.append({
            "feature_id": fid,
            "category": "rating",
            "description": desc,
            "mask": mask.astype(bool),
            "unknown_mask": rating_unknown.astype(bool),
        })

    # 5. Genre Predicates (Structured metadata genres)
    top_genres = [
        ("genre:action", ["action"]),
        ("genre:adventure", ["adventure"]),
        ("genre:animation", ["animation", "anime"]),
        ("genre:comedy", ["comedy"]),
        ("genre:crime", ["crime"]),
        ("genre:documentary", ["documentary"]),
        ("genre:drama", ["drama"]),
        ("genre:family", ["family", "children"]),
        ("genre:fantasy", ["fantasy"]),
        ("genre:history", ["history", "historical"]),
        ("genre:horror", ["horror"]),
        ("genre:music", ["music"]),
        ("genre:mystery", ["mystery"]),
        ("genre:romance", ["romance", "romantic"]),
        ("genre:science_fiction", ["science fiction", "sci-fi", "scifi"]),
        ("genre:thriller", ["thriller", "suspense"]),
        ("genre:war", ["war"]),
        ("genre:western", ["western"]),
    ]

    def _check_genre(row_genres: Any, targets: list[str]) -> bool:
        if isinstance(row_genres, (list, np.ndarray, tuple)):
            g_strs = [str(g).lower() for g in row_genres]
        elif isinstance(row_genres, str):
            g_strs = [row_genres.lower()]
        else:
            return False
        return any(any(t in g for t in targets) for g in g_strs)

    if "genres" in df.columns:
        genre_series = df["genres"]
        genre_unknown = genre_series.isna().values
    else:
        genre_series = pd.Series([None] * num_entities)
        genre_unknown = np.ones(num_entities, dtype=bool)

    for fid, targets in top_genres:
        g_name = targets[0].capitalize()
        mask = genre_series.apply(lambda x: _check_genre(x, targets)).values.astype(bool)
        predicates.append({
            "feature_id": fid,
            "category": "genre",
            "description": f"Does it belong to the {g_name} genre?",
            "mask": mask,
            "unknown_mask": genre_unknown.astype(bool),
        })

    return predicates
