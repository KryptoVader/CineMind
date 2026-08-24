"""
Multi-strategy TMDB discovery.

Strategies:
  1. Year-by-year — discover/movie + discover/tv for each year
  2. Language segmentation — repeat for high-volume years with language filters
  3. Genre × decade — genre-based discovery across decades
  4. Low-popularity sweep — sort_by=vote_count.asc for obscure titles

All strategies write append-only JSONL and track provenance.
Checkpoints make every strategy safely resumable.
"""

import json
import logging
from typing import Any

from pipeline.checkpoint import CheckpointManager, GracefulShutdown
from pipeline.config import (
    CHECKPOINT_SAVE_INTERVAL,
    DISCOVERY_END_YEAR,
    DISCOVERY_START_YEAR,
    HIGH_VOLUME_PAGE_THRESHOLD,
    RAW_TMDB_DIR,
    TMDB_MAX_PAGES,
    TMDB_MAX_PAGES_PER_YEAR_MOVIE,
    TMDB_MAX_PAGES_PER_YEAR_TV,
    TMDB_SEGMENT_LANGUAGES,
)
from pipeline.tmdb.client import TMDBClient

logger = logging.getLogger(__name__)


class TMDBDiscovery:
    """Multi-strategy TMDB title discovery."""

    def __init__(self) -> None:
        self.client = TMDBClient()
        self.checkpoint = CheckpointManager("tmdb_discovery")

        # Raw output files (append-only JSONL)
        RAW_TMDB_DIR.mkdir(parents=True, exist_ok=True)
        self.movies_path = RAW_TMDB_DIR / "discovery_movies.jsonl"
        self.tv_path = RAW_TMDB_DIR / "discovery_tv.jsonl"

        # In-memory dedup sets (rebuilt from JSONL on init)
        self.seen_movie_ids: set[int] = set()
        self.seen_tv_ids: set[int] = set()

        # Per-strategy contribution counters
        self.strategy_stats: dict[str, dict[str, int]] = {}

        self._load_existing_ids()

        # High-volume years (loaded from checkpoint or discovered)
        self.high_volume_years: dict[str, set[int]] = {
            "movie": set(self.checkpoint.get_stat("hv_movie", [])),
            "tv": set(self.checkpoint.get_stat("hv_tv", [])),
        }

        # Genre map (cached)
        self.genre_map = self.client.load_genre_map()

    # ==========================================================
    # Initialisation helpers
    # ==========================================================

    def _load_existing_ids(self) -> None:
        """Rebuild seen-ID sets from existing JSONL files."""
        for path, id_set, label in [
            (self.movies_path, self.seen_movie_ids, "movie"),
            (self.tv_path, self.seen_tv_ids, "TV"),
        ]:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        id_set.add(record["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info(
                "Loaded %d existing TMDB %s IDs from JSONL",
                len(id_set), label,
            )

    # ==========================================================
    # Record writing
    # ==========================================================

    def _write_records(
        self,
        records: list[dict[str, Any]],
        media_type: str,
        strategy: str,
        **metadata: Any,
    ) -> int:
        """Append new records to JSONL, deduplicating by ID."""
        id_set = (
            self.seen_movie_ids
            if media_type == "movie"
            else self.seen_tv_ids
        )
        path = (
            self.movies_path
            if media_type == "movie"
            else self.tv_path
        )

        new_count = 0
        strategy_key = f"{strategy}:{media_type}"

        with open(path, "a", encoding="utf-8") as fh:
            for record in records:
                rid = record.get("id")
                if rid is None or rid in id_set:
                    continue

                id_set.add(rid)
                # Attach provenance metadata
                record["_media_type"] = media_type
                record["_strategy"] = strategy
                for k, v in metadata.items():
                    record[f"_{k}"] = v

                fh.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                new_count += 1

        # Track strategy contribution
        stats = self.strategy_stats.setdefault(
            strategy_key, {"total": 0, "new": 0}
        )
        stats["total"] += len(records)
        stats["new"] += new_count

        return new_count

    # ==========================================================
    # Core paginated discovery loop
    # ==========================================================

    def _discover_paginated(
        self,
        media_type: str,
        task_key: str,
        strategy: str,
        api_params: dict[str, Any],
        max_pages: int = TMDB_MAX_PAGES,
        **metadata: Any,
    ) -> tuple[int, int]:
        """
        Paginate a single discover query up to max_pages.

        Returns (new_ids_found, total_pages_reported).
        """
        if self.checkpoint.is_complete(task_key):
            return 0, 0

        start_page = self.checkpoint.get_progress(task_key) + 1
        if start_page > max_pages:
            self.checkpoint.complete_task(task_key)
            return 0, 0

        discover_fn = (
            self.client.discover_movies
            if media_type == "movie"
            else self.client.discover_tv
        )

        total_new = 0
        total_pages = 0

        for page in range(start_page, max_pages + 1):
            if GracefulShutdown.is_requested():
                self.checkpoint.update_progress(task_key, page - 1)
                self.checkpoint.save()
                return total_new, total_pages

            try:
                data = discover_fn(page=page, **api_params)
            except Exception as exc:
                logger.error(
                    "Error on %s page %d: %s",
                    task_key, page, exc,
                )
                self.checkpoint.update_progress(
                    task_key, max(0, page - 1)
                )
                self.checkpoint.save()
                return total_new, total_pages

            results = data.get("results", [])
            reported_pages = min(
                data.get("total_pages", 0), TMDB_MAX_PAGES
            )
            if page == start_page:
                total_pages = reported_pages

            if not results:
                break

            new = self._write_records(
                results, media_type, strategy, **metadata
            )
            total_new += new

            # Periodic checkpoint
            if page % CHECKPOINT_SAVE_INTERVAL == 0:
                self.checkpoint.update_progress(task_key, page)
                self.checkpoint.save()

            if page >= reported_pages:
                break

        self.checkpoint.complete_task(task_key)
        return total_new, total_pages

    # ==========================================================
    # Strategy 1: Year-by-year
    # ==========================================================

    def run_year_discovery(
        self,
        years: list[int] | None = None,
    ) -> None:
        """Discover movies & TV year-by-year."""
        if years is None:
            years = list(
                range(DISCOVERY_START_YEAR, DISCOVERY_END_YEAR + 1)
            )

        logger.info(
            "TMDB year-by-year discovery: %d years (%d–%d)",
            len(years), min(years), max(years),
        )

        for year in years:
            for media_type in ["movie", "tv"]:
                if GracefulShutdown.is_requested():
                    return

                task_key = f"year:{media_type}:{year}"
                if self.checkpoint.is_complete(task_key):
                    continue

                logger.info("Discovering %s %d ...", media_type, year)

                if media_type == "movie":
                    params = {"primary_release_year": year}
                    page_cap = TMDB_MAX_PAGES_PER_YEAR_MOVIE
                else:
                    params = {"first_air_date_year": year}
                    page_cap = TMDB_MAX_PAGES_PER_YEAR_TV

                new, pages = self._discover_paginated(
                    media_type, task_key, "year",
                    params, max_pages=page_cap, year=year,
                )

                # Flag for language segmentation
                if pages >= HIGH_VOLUME_PAGE_THRESHOLD:
                    self.high_volume_years[media_type].add(year)
                    hv_key = f"hv_{media_type}"
                    self.checkpoint.update_stat(
                        hv_key,
                        sorted(self.high_volume_years[media_type]),
                    )
                    self.checkpoint.save()
                    logger.info(
                        "  ⚠ High-volume: %s %d (%d pages) "
                        "→ flagged for language segmentation",
                        media_type, year, pages,
                    )

                logger.info(
                    "  %s %d: %d pages, %d new  |  "
                    "Running: %d movies, %d TV",
                    media_type, year, pages, new,
                    len(self.seen_movie_ids),
                    len(self.seen_tv_ids),
                )

    # ==========================================================
    # Strategy 2: Language segmentation
    # ==========================================================

    def run_language_segmentation(
        self,
        years: list[int] | None = None,
        languages: list[str] | None = None,
    ) -> None:
        """Re-discover high-volume years filtered by language."""
        if languages is None:
            languages = TMDB_SEGMENT_LANGUAGES

        target: dict[str, list[int]] = {}
        for mt in ["movie", "tv"]:
            hv = self.high_volume_years[mt]
            if years:
                target[mt] = sorted(y for y in years if y in hv)
            else:
                target[mt] = sorted(hv)

        combos = sum(
            len(yrs) * len(languages) for yrs in target.values()
        )
        if combos == 0:
            logger.info(
                "No high-volume years for language segmentation."
            )
            return

        logger.info(
            "Language segmentation: %d year×lang combos", combos,
        )

        for media_type, year_list in target.items():
            for year in year_list:
                for lang in languages:
                    if GracefulShutdown.is_requested():
                        return

                    task_key = f"lang:{media_type}:{year}:{lang}"
                    if self.checkpoint.is_complete(task_key):
                        continue

                    if media_type == "movie":
                        params = {
                            "primary_release_year": year,
                            "with_original_language": lang,
                        }
                    else:
                        params = {
                            "first_air_date_year": year,
                            "with_original_language": lang,
                        }

                    new, _ = self._discover_paginated(
                        media_type, task_key, "year_lang",
                        params, year=year, language=lang,
                    )

                    if new > 0:
                        logger.info(
                            "  %s %d [%s]: +%d new",
                            media_type, year, lang, new,
                        )

    # ==========================================================
    # Strategy 3: Genre × decade
    # ==========================================================

    def run_genre_decade_discovery(
        self,
        decades: list[tuple[int, int]] | None = None,
    ) -> None:
        """Discover by genre within each decade."""
        if not self.genre_map:
            logger.warning(
                "No genre map available. Skipping genre discovery."
            )
            return

        if decades is None:
            decades = []
            for start in range(
                DISCOVERY_START_YEAR, DISCOVERY_END_YEAR + 1, 10
            ):
                end = min(start + 9, DISCOVERY_END_YEAR)
                decades.append((start, end))

        logger.info(
            "Genre × decade: %d genres × %d decades",
            len(self.genre_map), len(decades),
        )

        for genre_id, genre_name in sorted(self.genre_map.items()):
            for ds, de in decades:
                for media_type in ["movie", "tv"]:
                    if GracefulShutdown.is_requested():
                        return

                    task_key = (
                        f"genre:{media_type}:{genre_id}:{ds}_{de}"
                    )
                    if self.checkpoint.is_complete(task_key):
                        continue

                    if media_type == "movie":
                        params = {
                            "with_genres": str(genre_id),
                            "primary_release_date.gte": f"{ds}-01-01",
                            "primary_release_date.lte": f"{de}-12-31",
                        }
                    else:
                        params = {
                            "with_genres": str(genre_id),
                            "first_air_date.gte": f"{ds}-01-01",
                            "first_air_date.lte": f"{de}-12-31",
                        }

                    new, _ = self._discover_paginated(
                        media_type, task_key, "genre_decade",
                        params,
                        genre_id=genre_id,
                        genre_name=genre_name,
                        decade=f"{ds}-{de}",
                    )

                    if new > 0:
                        logger.info(
                            "  %s %s %d–%d: +%d new",
                            media_type, genre_name, ds, de, new,
                        )

    # ==========================================================
    # Strategy 4: Low-popularity sweep
    # ==========================================================

    def run_low_popularity_discovery(
        self,
        years: list[int] | None = None,
    ) -> None:
        """Find obscure titles by sorting by fewest votes."""
        if years is None:
            years = list(
                range(DISCOVERY_START_YEAR, DISCOVERY_END_YEAR + 1)
            )

        logger.info("Low-popularity sweep: %d years", len(years))

        for year in years:
            for media_type in ["movie", "tv"]:
                if GracefulShutdown.is_requested():
                    return

                task_key = f"lowpop:{media_type}:{year}"
                if self.checkpoint.is_complete(task_key):
                    continue

                if media_type == "movie":
                    params = {
                        "primary_release_year": year,
                        "sort_by": "vote_count.asc",
                        "vote_count.gte": 1,
                    }
                else:
                    params = {
                        "first_air_date_year": year,
                        "sort_by": "vote_count.asc",
                        "vote_count.gte": 1,
                    }

                new, _ = self._discover_paginated(
                    media_type, task_key, "low_popularity",
                    params, year=year,
                )

                if new > 0:
                    logger.info(
                        "  %s %d low-pop: +%d new",
                        media_type, year, new,
                    )

    # ==========================================================
    # Orchestration
    # ==========================================================

    def run_all(
        self,
        years: list[int] | None = None,
    ) -> None:
        """Run all four strategies in sequence."""
        logger.info("=" * 60)
        logger.info("TMDB FULL DISCOVERY")
        logger.info("=" * 60)

        GracefulShutdown.install()

        self.run_year_discovery(years)
        if GracefulShutdown.is_requested():
            self._log_final_stats()
            return

        self.run_language_segmentation(years)
        if GracefulShutdown.is_requested():
            self._log_final_stats()
            return

        self.run_genre_decade_discovery()
        if GracefulShutdown.is_requested():
            self._log_final_stats()
            return

        self.run_low_popularity_discovery(years)
        self._log_final_stats()

    def _log_final_stats(self) -> None:
        """Log summary statistics."""
        total = len(self.seen_movie_ids) + len(self.seen_tv_ids)
        logger.info("=" * 60)
        logger.info("TMDB DISCOVERY SUMMARY")
        logger.info("  Movies:       %d", len(self.seen_movie_ids))
        logger.info("  TV:           %d", len(self.seen_tv_ids))
        logger.info("  Total:        %d", total)
        logger.info("  API requests: %d", self.client.request_count)
        for key, stats in sorted(self.strategy_stats.items()):
            logger.info(
                "  %-25s total=%d  new=%d",
                key, stats["total"], stats["new"],
            )
        logger.info("=" * 60)

    def get_status(self) -> dict[str, Any]:
        """Return structured status for CLI."""
        return {
            "movies": len(self.seen_movie_ids),
            "tv": len(self.seen_tv_ids),
            "total": (
                len(self.seen_movie_ids) + len(self.seen_tv_ids)
            ),
            "completed_tasks": self.checkpoint.completed_count,
            "api_requests": self.client.request_count,
            "strategy_stats": dict(self.strategy_stats),
            "high_volume_years": {
                k: sorted(v)
                for k, v in self.high_volume_years.items()
            },
        }
