"""
Multi-strategy MAL discovery.

Strategies:
  1. Ranking discovery — paginate all ranking types
  2. Systematic search — query with controlled vocabulary

All records write append-only JSONL with provenance.
Follows paging.next for pagination (not manual offset math).
"""

import json
import logging
from typing import Any

import requests

from pipeline.checkpoint import CheckpointManager, GracefulShutdown
from pipeline.config import (
    CHECKPOINT_SAVE_INTERVAL,
    MAL_MAX_RANKING_OFFSET,
    MAL_MAX_SEARCH_OFFSET,
    MAL_RANKING_TYPES,
    MAL_RESULTS_PER_PAGE,
    MAL_SEARCH_VOCABULARY,
    RAW_MAL_DIR,
)
from pipeline.mal.client import MALClient

logger = logging.getLogger(__name__)


class MALDiscovery:
    """Multi-strategy MAL anime discovery."""

    def __init__(self) -> None:
        self.client = MALClient()
        self.checkpoint = CheckpointManager("mal_discovery")

        RAW_MAL_DIR.mkdir(parents=True, exist_ok=True)
        self.raw_path = RAW_MAL_DIR / "discovery.jsonl"

        # In-memory dedup
        self.seen_ids: set[int] = set()

        # Per-strategy stats
        self.strategy_stats: dict[str, dict[str, int]] = {}

        # Per-query marginal contribution (for the report)
        self.query_contributions: dict[str, int] = {}

        self._load_existing_ids()

    # ==========================================================
    # Init helpers
    # ==========================================================

    def _load_existing_ids(self) -> None:
        """Rebuild seen-ID set from existing JSONL."""
        if not self.raw_path.exists():
            return
        with open(self.raw_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.seen_ids.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(
            "Loaded %d existing MAL IDs from JSONL",
            len(self.seen_ids),
        )

    # ==========================================================
    # Record writing
    # ==========================================================

    def _write_records(
        self,
        records: list[dict[str, Any]],
        strategy: str,
        **metadata: Any,
    ) -> int:
        """Append new records to JSONL, deduplicating by ID."""
        new_count = 0

        with open(self.raw_path, "a", encoding="utf-8") as fh:
            for record in records:
                rid = record.get("id")
                if rid is None or rid in self.seen_ids:
                    continue

                self.seen_ids.add(rid)
                record["_strategy"] = strategy
                for k, v in metadata.items():
                    record[f"_{k}"] = v

                fh.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                new_count += 1

        # Track stats
        stats = self.strategy_stats.setdefault(
            strategy, {"total": 0, "new": 0}
        )
        stats["total"] += len(records)
        stats["new"] += new_count

        return new_count

    # ==========================================================
    # Strategy 1: Ranking discovery
    # ==========================================================

    def run_ranking_discovery(
        self,
        ranking_types: list[str] | None = None,
    ) -> None:
        """Paginate all useful MAL ranking types."""
        if ranking_types is None:
            ranking_types = MAL_RANKING_TYPES

        logger.info(
            "MAL ranking discovery: %d types", len(ranking_types),
        )

        for ranking_type in ranking_types:
            if GracefulShutdown.is_requested():
                return

            task_key = f"ranking:{ranking_type}"
            if self.checkpoint.is_complete(task_key):
                logger.info(
                    "  ranking '%s': already complete", ranking_type,
                )
                continue

            offset = self.checkpoint.get_progress(task_key)
            logger.info(
                "  ranking '%s' from offset %d ...",
                ranking_type, offset,
            )

            batch_count = 0

            while True:
                if GracefulShutdown.is_requested():
                    self.checkpoint.update_progress(
                        task_key, offset,
                    )
                    self.checkpoint.save()
                    return

                try:
                    data = self.client.get_ranking(
                        ranking_type=ranking_type,
                        limit=MAL_RESULTS_PER_PAGE,
                        offset=offset,
                    )
                except Exception as exc:
                    logger.error(
                        "Error: ranking '%s' offset %d: %s",
                        ranking_type, offset, exc,
                    )
                    self.checkpoint.update_progress(
                        task_key, offset,
                    )
                    self.checkpoint.save()
                    break

                results = data.get("data", [])
                if not results:
                    break

                # Extract node + ranking info
                records = []
                for item in results:
                    node = item.get("node", {})
                    node["_ranking_info"] = item.get("ranking", {})
                    records.append(node)

                new = self._write_records(
                    records, "ranking",
                    ranking_type=ranking_type,
                )

                offset += len(results)
                batch_count += 1

                # Periodic checkpoint
                if batch_count % CHECKPOINT_SAVE_INTERVAL == 0:
                    self.checkpoint.update_progress(
                        task_key, offset,
                    )
                    self.checkpoint.save()

                # Follow paging.next
                paging = data.get("paging", {})
                if "next" not in paging or offset >= MAL_MAX_RANKING_OFFSET:
                    break

                # Log progress every ~1000 records
                if offset % 1000 < MAL_RESULTS_PER_PAGE:
                    logger.info(
                        "    '%s': offset %d, unique total %d",
                        ranking_type, offset, len(self.seen_ids),
                    )

            self.checkpoint.complete_task(task_key)
            logger.info(
                "  ranking '%s': done at offset %d  |  "
                "Total unique: %d",
                ranking_type, offset, len(self.seen_ids),
            )

    # ==========================================================
    # Strategy 2: Systematic search
    # ==========================================================

    def run_search_discovery(
        self,
        vocabulary: list[str] | None = None,
    ) -> None:
        """Search with systematic vocabulary to find titles
        missed by ranking endpoints."""
        if vocabulary is None:
            vocabulary = MAL_SEARCH_VOCABULARY

        # Filter to >= 3 characters (MAL requirement)
        vocabulary = [q for q in vocabulary if len(q) >= 3]

        logger.info(
            "MAL search discovery: %d queries", len(vocabulary),
        )

        for idx, query in enumerate(vocabulary, 1):
            if GracefulShutdown.is_requested():
                return

            task_key = f"search:{query}"
            if self.checkpoint.is_complete(task_key):
                continue

            offset = self.checkpoint.get_progress(task_key)
            query_new = 0

            while True:
                if GracefulShutdown.is_requested():
                    self.checkpoint.update_progress(
                        task_key, offset,
                    )
                    self.checkpoint.save()
                    return

                try:
                    data = self.client.search_anime(
                        q=query,
                        limit=MAL_RESULTS_PER_PAGE,
                        offset=offset,
                    )
                except requests.exceptions.HTTPError:
                    # Some queries may return 400 — skip them
                    logger.debug(
                        "  search '%s': HTTP error, skipping",
                        query,
                    )
                    break
                except Exception as exc:
                    logger.error(
                        "Error: search '%s' offset %d: %s",
                        query, offset, exc,
                    )
                    self.checkpoint.update_progress(
                        task_key, offset,
                    )
                    self.checkpoint.save()
                    break

                results = data.get("data", [])
                if not results:
                    break

                records = [item.get("node", {}) for item in results]
                new = self._write_records(
                    records, "search", query=query,
                )
                query_new += new

                offset += len(results)

                # Follow paging.next
                paging = data.get("paging", {})
                if "next" not in paging or offset >= MAL_MAX_SEARCH_OFFSET:
                    break

            self.checkpoint.complete_task(task_key)
            self.query_contributions[query] = query_new

            if query_new > 0 or idx % 20 == 0:
                logger.info(
                    "  [%d/%d] search '%s': +%d new  |  "
                    "Total unique: %d",
                    idx, len(vocabulary), query,
                    query_new, len(self.seen_ids),
                )

    # ==========================================================
    # Orchestration
    # ==========================================================

    def run_all(
        self,
        ranking_types: list[str] | None = None,
    ) -> None:
        """Run all MAL discovery strategies."""
        logger.info("=" * 60)
        logger.info("MAL FULL DISCOVERY")
        logger.info("=" * 60)

        GracefulShutdown.install()

        self.run_ranking_discovery(ranking_types)
        if GracefulShutdown.is_requested():
            self._log_final_stats()
            return

        self.run_search_discovery()
        self._log_final_stats()

    def _log_final_stats(self) -> None:
        total = len(self.seen_ids)
        logger.info("=" * 60)
        logger.info("MAL DISCOVERY SUMMARY")
        logger.info("  Total unique:  %d", total)
        logger.info("  API requests:  %d", self.client.request_count)
        for strat, stats in sorted(self.strategy_stats.items()):
            logger.info(
                "  %-20s total=%d  new=%d",
                strat, stats["total"], stats["new"],
            )
        if self.query_contributions:
            top = sorted(
                self.query_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            logger.info("  Top 10 search queries by new IDs:")
            for q, n in top:
                logger.info("    '%s': %d", q, n)
        logger.info("=" * 60)

    def get_status(self) -> dict[str, Any]:
        return {
            "total": len(self.seen_ids),
            "completed_tasks": self.checkpoint.completed_count,
            "api_requests": self.client.request_count,
            "strategy_stats": dict(self.strategy_stats),
            "query_contributions": dict(self.query_contributions),
        }
