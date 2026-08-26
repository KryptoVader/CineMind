"""
CineMind data pipeline CLI.

Commands:
  discover tmdb [--years Y1 Y2 ...]
  discover mal  [--ranking-types T1 T2 ...]
  discover all

  process       Normalize + deduplicate both sources
  match         Cross-source entity resolution
  audit         Generate all reports
  status        Show current pipeline state
  export        Export final candidate dataset summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.config import (
    CANONICAL_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    DISCOVERY_END_YEAR,
    DISCOVERY_START_YEAR,
    RAW_MAL_DIR,
    RAW_TMDB_DIR,
    REPORTS_DIR,
    STAGING_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================
# Command handlers
# =============================================================

def cmd_discover_tmdb(args: argparse.Namespace) -> None:
    """Run TMDB discovery."""
    from pipeline.tmdb.discovery import TMDBDiscovery

    disc = TMDBDiscovery()

    years = None
    if args.years:
        years = [int(y) for y in args.years]

    disc.run_all(years=years)


def cmd_discover_mal(args: argparse.Namespace) -> None:
    """Run MAL discovery."""
    from pipeline.mal.discovery import MALDiscovery

    disc = MALDiscovery()

    ranking_types = None
    if args.ranking_types:
        ranking_types = args.ranking_types

    disc.run_all(ranking_types=ranking_types)


def cmd_discover_all(args: argparse.Namespace) -> None:
    """Run both TMDB and MAL discovery."""
    from pipeline.checkpoint import GracefulShutdown

    logger.info("Running TMDB discovery...")
    cmd_discover_tmdb(args)

    if GracefulShutdown.is_requested():
        logger.info("Shutdown requested. Skipping MAL discovery.")
        return

    logger.info("Running MAL discovery...")
    cmd_discover_mal(args)


def cmd_process(args: argparse.Namespace) -> None:
    """Normalize and deduplicate both sources."""
    from pipeline.canonical.entity import build_canonical_dataset
    from pipeline.mal.normalizer import normalize_mal
    from pipeline.tmdb.normalizer import normalize_tmdb

    logger.info("=" * 60)
    logger.info("PROCESSING PIPELINE")
    logger.info("=" * 60)

    logger.info("Normalizing TMDB...")
    tmdb_df = normalize_tmdb()

    logger.info("Normalizing MAL...")
    mal_df = normalize_mal()

    logger.info("Building canonical dataset...")
    candidates = build_canonical_dataset()

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(
        "  TMDB:       %d records",
        len(tmdb_df) if tmdb_df is not None else 0,
    )
    logger.info(
        "  MAL:        %d records",
        len(mal_df) if mal_df is not None else 0,
    )
    logger.info(
        "  Candidates: %d records",
        len(candidates) if candidates is not None else 0,
    )
    logger.info("=" * 60)


def cmd_match(args: argparse.Namespace) -> None:
    """Run cross-source entity resolution."""
    from pipeline.canonical.entity import build_canonical_dataset
    from pipeline.canonical.matcher import run_cross_source_matching

    logger.info("=" * 60)
    logger.info("CROSS-SOURCE MATCHING")
    logger.info("=" * 60)

    matches, unresolved = run_cross_source_matching()

    logger.info("Rebuilding canonical universe with verified links...")
    canonical = build_canonical_dataset()

    logger.info("=" * 60)
    logger.info("MATCHING COMPLETE")
    logger.info("  Matches:    %d", len(matches))
    logger.info("  Unresolved: %d", len(unresolved))
    logger.info("  Canonical:  %d", len(canonical) if canonical is not None else 0)
    logger.info("=" * 60)


def cmd_audit(args: argparse.Namespace) -> None:
    """Generate all reports."""
    from pipeline.audit.reports import generate_all_reports

    generate_all_reports()

    logger.info(
        "Reports available in: %s", REPORTS_DIR,
    )


def cmd_analytics(args: argparse.Namespace) -> None:
    """Build development datasets, run validation, and generate DEVELOPMENT_DATASET_REPORT.md."""
    from pipeline.analytics.builder import build_analytics_views
    from pipeline.analytics.validator import validate_analytics_datasets
    from pipeline.analytics.report import generate_development_dataset_report

    logger.info("=" * 60)
    logger.info("BUILDING ANALYTICAL DATASETS & GUESSER READINESS LAYER")
    logger.info("=" * 60)

    views = build_analytics_views()
    val_res = validate_analytics_datasets(views)
    report_md = generate_development_dataset_report(views, val_res)

    print("\n" + "=" * 60)
    print("CINEMIND ANALYTICAL DATASETS & VALIDATION SUMMARY")
    print("=" * 60)
    for name, df in views.items():
        print(f"  {name:<32}: {len(df):>7,} records")
    print("-" * 60)
    print(f"  Validation Status               : {'ALL PASSED (12/12)' if val_res['all_passed'] else 'SOME FAILED'}")
    print("=" * 60 + "\n")


def cmd_sample(args: argparse.Namespace) -> None:
    """Run stratified diversity sampling."""
    from pipeline.canonical.sampler import run_diversity_sampling

    target = args.target if hasattr(args, "target") and args.target else 100000

    logger.info("=" * 60)
    logger.info("STRATIFIED DIVERSITY SAMPLING (Target: %d)", target)
    logger.info("=" * 60)

    df = run_diversity_sampling(target=target)

    logger.info("=" * 60)
    logger.info("SAMPLING COMPLETE: %d diverse records created", len(df))
    logger.info("=" * 60)


def cmd_status(args: argparse.Namespace) -> None:
    """Show current pipeline state."""
    print("=" * 60)
    print("CINEMIND PIPELINE STATUS")
    print("=" * 60)

    # Raw data
    print("\n--- Raw Data ---")
    for name, path in [
        ("TMDB movies JSONL", RAW_TMDB_DIR / "discovery_movies.jsonl"),
        ("TMDB TV JSONL", RAW_TMDB_DIR / "discovery_tv.jsonl"),
        ("MAL discovery JSONL", RAW_MAL_DIR / "discovery.jsonl"),
    ]:
        if path.exists():
            line_count = sum(
                1 for _ in open(path, encoding="utf-8")
            )
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {line_count:,} lines ({size_mb:.1f} MB)")
        else:
            print(f"  {name}: not found")

    # Staging data
    print("\n--- Staging ---")
    for name, path in [
        ("TMDB normalized", STAGING_DIR / "tmdb_normalized.parquet"),
        ("MAL normalized", STAGING_DIR / "mal_normalized.parquet"),
    ]:
        if path.exists():
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"  {name}: {len(df):,} records")
        else:
            print(f"  {name}: not found")

    # Canonical data
    print("\n--- Canonical ---")
    for name, path in [
        ("Candidates", CANONICAL_DIR / "candidates.parquet"),
        ("Cross-source matches", CANONICAL_DIR / "cross_source_matches.parquet"),
        ("Unresolved matches", CANONICAL_DIR / "unresolved_matches.parquet"),
    ]:
        if path.exists():
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"  {name}: {len(df):,} records")
        else:
            print(f"  {name}: not found")

    # Checkpoints
    print("\n--- Checkpoints ---")
    for cp_file in sorted(CHECKPOINT_DIR.glob("*_state.json")):
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            completed = len(data.get("completed", []))
            in_progress = len(data.get("progress", {}))
            print(
                f"  {cp_file.name}: "
                f"{completed} completed, "
                f"{in_progress} in progress"
            )
        except Exception:
            print(f"  {cp_file.name}: (error reading)")

    # Reports
    print("\n--- Reports ---")
    if REPORTS_DIR.exists():
        for rpt in sorted(REPORTS_DIR.glob("*.md")):
            size_kb = rpt.stat().st_size / 1024
            print(f"  {rpt.name} ({size_kb:.1f} KB)")
    else:
        print("  No reports generated yet.")

    print("\n" + "=" * 60)


def cmd_export(args: argparse.Namespace) -> None:
    """Export summary of the candidate dataset."""
    import pandas as pd

    candidates_path = CANONICAL_DIR / "candidates.parquet"
    if not candidates_path.exists():
        print("No candidate dataset found. Run 'process' first.")
        return

    df = pd.read_parquet(candidates_path)

    print("=" * 60)
    print("CINEMIND CANDIDATE UNIVERSE")
    print("=" * 60)
    print(f"\nTotal candidates: {len(df):,}")

    if "source" in df.columns:
        print("\nBy source:")
        for src, cnt in df["source"].value_counts().items():
            print(f"  {src}: {cnt:,}")

    if "media_type" in df.columns:
        print("\nBy media type:")
        for mt, cnt in df["media_type"].value_counts().items():
            print(f"  {mt}: {cnt:,}")

    if "release_year" in df.columns:
        years = df["release_year"].dropna()
        if len(years) > 0:
            print(f"\nYear range: {int(years.min())} – {int(years.max())}")

    if "original_language" in df.columns:
        print("\nTop 10 languages:")
        for lang, cnt in (
            df["original_language"].value_counts().head(10).items()
        ):
            print(f"  {lang}: {cnt:,}")

    print(f"\nCandidate file: {candidates_path}")
    print("=" * 60)


def cmd_enrich_tmdb(args: argparse.Namespace) -> None:
    """Run TMDB keyword enrichment crawler and merge back onto staging and canonical."""
    from pipeline.tmdb.enricher import TMDBKeywordEnricher, merge_tmdb_keywords
    from pipeline.canonical.entity import build_canonical_dataset

    max_items = getattr(args, "max_items", None)
    logger.info("=" * 60)
    logger.info("TMDB KEYWORD ENRICHMENT CRAWLER (max_items: %s)", max_items or "ALL")
    logger.info("=" * 60)

    enricher = TMDBKeywordEnricher()
    enricher.run(max_items=max_items)

    logger.info("Merging enriched keywords back into tmdb_normalized.parquet...")
    merged_tmdb = merge_tmdb_keywords()

    logger.info("Rebuilding canonical dataset with enriched TMDB keywords...")
    canonical = build_canonical_dataset()

    logger.info("=" * 60)
    logger.info("TMDB KEYWORD ENRICHMENT COMPLETE")
    logger.info("  TMDB staging entities:  %d", len(merged_tmdb))
    logger.info("  Canonical entities:     %d", len(canonical))
    logger.info("=" * 60)


def cmd_enrich_mal(args: argparse.Namespace) -> None:
    """Run official MAL v2 theme/demographic enrichment crawler and merge back onto staging and canonical."""
    from pipeline.mal.enricher import MALEnricher, merge_mal_enrichment
    from pipeline.canonical.entity import build_canonical_dataset

    max_items = getattr(args, "max_items", None)
    logger.info("=" * 60)
    logger.info("OFFICIAL MAL V2 THEME & DEMOGRAPHIC ENRICHMENT (max_items: %s)", max_items or "ALL")
    logger.info("=" * 60)

    enricher = MALEnricher()
    enricher.run(max_items=max_items)

    logger.info("Merging enriched themes/demographics back into mal_normalized.parquet...")
    merged_mal = merge_mal_enrichment()

    logger.info("Rebuilding canonical dataset with enriched MAL themes...")
    canonical = build_canonical_dataset()

    logger.info("=" * 60)
    logger.info("MAL THEME ENRICHMENT COMPLETE")
    logger.info("  MAL staging entities:   %d", len(merged_mal))
    logger.info("  Canonical entities:     %d", len(canonical))
    logger.info("=" * 60)


def cmd_normalize_genres(args: argparse.Namespace) -> None:
    """Rebuild canonical dataset applying genre taxonomy unification."""
    from pipeline.canonical.entity import build_canonical_dataset

    logger.info("=" * 60)
    logger.info("GENRE TAXONOMY UNIFICATION")
    logger.info("=" * 60)

    canonical = build_canonical_dataset()

    logger.info("=" * 60)
    logger.info("GENRE TAXONOMY UNIFICATION COMPLETE: %d canonical entities", len(canonical))
    logger.info("=" * 60)


# =============================================================
# Argument parser
# =============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="CineMind Data Acquisition & Processing Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- discover ---
    discover_parser = subparsers.add_parser(
        "discover", help="Run discovery for one or both sources",
    )
    discover_sub = discover_parser.add_subparsers(dest="source")

    # discover tmdb
    tmdb_p = discover_sub.add_parser("tmdb", help="TMDB discovery")
    tmdb_p.add_argument(
        "--years", nargs="+", type=int,
        help="Specific years to discover (default: all)",
    )

    # discover mal
    mal_p = discover_sub.add_parser("mal", help="MAL discovery")
    mal_p.add_argument(
        "--ranking-types", nargs="+",
        help="Specific ranking types (default: all)",
    )

    # discover all
    all_p = discover_sub.add_parser(
        "all", help="Run both TMDB and MAL discovery",
    )
    all_p.add_argument(
        "--years", nargs="+", type=int,
        help="Specific years for TMDB (default: all)",
    )
    all_p.add_argument(
        "--ranking-types", nargs="+",
        help="Specific ranking types for MAL (default: all)",
    )

    # --- enrich ---
    enrich_parser = subparsers.add_parser(
        "enrich", help="Run enrichment crawlers for keywords, themes, and demographics",
    )
    enrich_sub = enrich_parser.add_subparsers(dest="target")

    enrich_tmdb_p = enrich_sub.add_parser("tmdb-keywords", help="TMDB keywords enrichment")
    enrich_tmdb_p.add_argument("--max-items", type=int, help="Limit number of items to process in this run")

    enrich_mal_p = enrich_sub.add_parser("mal-themes", help="MAL themes/demographics enrichment")
    enrich_mal_p.add_argument("--max-items", type=int, help="Limit number of items to process in this run")

    # --- normalize-genres ---
    subparsers.add_parser(
        "normalize-genres", help="Rebuild canonical dataset applying genre taxonomy unification",
    )

    # --- process ---
    subparsers.add_parser(
        "process",
        help="Normalize + deduplicate + build canonical dataset",
    )

    # --- match ---
    subparsers.add_parser(
        "match", help="Cross-source entity resolution",
    )

    # --- audit ---
    subparsers.add_parser(
        "audit", help="Generate all audit reports",
    )

    # --- sample ---
    sample_p = subparsers.add_parser(
        "sample", help="Run stratified diversity sampling down to target size",
    )
    sample_p.add_argument(
        "--target", type=int, default=100000,
        help="Target total sample size (default: 100000)",
    )

    # --- status ---
    subparsers.add_parser(
        "status", help="Show current pipeline state",
    )

    # --- export ---
    subparsers.add_parser(
        "export", help="Export candidate dataset summary",
    )

    # --- analytics ---
    subparsers.add_parser(
        "analytics", help="Build development datasets, run validation, and generate report",
    )

    return parser


# =============================================================
# Main
# =============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "process": cmd_process,
        "match": cmd_match,
        "sample": cmd_sample,
        "audit": cmd_audit,
        "analytics": cmd_analytics,
        "status": cmd_status,
        "export": cmd_export,
        "normalize-genres": cmd_normalize_genres,
    }

    if args.command == "discover":
        if not hasattr(args, "source") or args.source is None:
            parser.parse_args(["discover", "--help"])
            sys.exit(1)

        source_handlers = {
            "tmdb": cmd_discover_tmdb,
            "mal": cmd_discover_mal,
            "all": cmd_discover_all,
        }
        handler = source_handlers.get(args.source)
        if handler:
            handler(args)
        else:
            print(f"Unknown source: {args.source}")
            sys.exit(1)

    elif args.command == "enrich":
        if not hasattr(args, "target") or args.target is None:
            parser.parse_args(["enrich", "--help"])
            sys.exit(1)

        enrich_handlers = {
            "tmdb-keywords": cmd_enrich_tmdb,
            "mal-themes": cmd_enrich_mal,
        }
        handler = enrich_handlers.get(args.target)
        if handler:
            handler(args)
        else:
            print(f"Unknown enrich target: {args.target}")
            sys.exit(1)

    elif args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

