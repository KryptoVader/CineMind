"""
Stratified Diversity Sampler.

Creates a balanced, diverse 100k candidate universe from the full
acquired pool. Zero popularity bias: sampling is UNIFORM across all
discovered candidates within each (Media Type × Decade × Language) stratum,
ensuring obscure 1980s movies have the exact same chance of selection as
massive 2026 blockbusters.

Outputs:
  - data/canonical/diverse_100k.parquet
  - data/reports/sampling_report.md
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import CANONICAL_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def _decade_label(year: Any) -> str:
    """Convert release_year to decade string (e.g. '1980s')."""
    if pd.isna(year):
        return "Unknown"
    try:
        y = int(year)
        if y < 1900:
            return "Pre-1900"
        base = (y // 10) * 10
        return f"{base}s"
    except (ValueError, TypeError):
        return "Unknown"


def _coarse_media_group(mt: str) -> str:
    """Group fine-grained media types into main categories."""
    if mt in ("anime_tv", "anime_movie", "ova", "ona", "special", "music"):
        return "anime"
    elif mt == "tv":
        return "tv"
    elif mt == "movie":
        return "movie"
    return "other"


def _coarse_lang_group(lang: Any) -> str:
    """Group languages into categories to ensure geographic diversity."""
    if pd.isna(lang) or not lang:
        return "other"
    lang = str(lang).lower().strip()
    major_languages = {
        "en", "ja", "es", "fr", "de", "ko", "hi", "zh", "it", "ru", "pt", "tr", "sv", "nl"
    }
    if lang in major_languages:
        return lang
    return "other_lang"


def sample_diverse_dataset(
    df: pd.DataFrame,
    target_total: int = 100000,
    media_alloc: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Perform multi-dimensional stratified sampling with zero popularity bias.

    Parameters:
        df: Input DataFrame (candidates)
        target_total: Desired total number of records (e.g. 100,000)
        media_alloc: Target proportions per media group (movie, tv, anime, other)
        seed: Random seed for reproducibility

    Returns:
        (sampled_df, metrics_dict)
    """
    if len(df) <= target_total:
        logger.info(
            "Input dataset size (%d) <= target (%d). Keeping all records.",
            len(df), target_total,
        )
        metrics = {
            "sampling_applied": False,
            "original_total": len(df),
            "sampled_total": len(df),
            "target_total": target_total,
            "quotas": df["media_type"].apply(_coarse_media_group).value_counts().to_dict(),
            "media_breakdown": df["media_type"].value_counts().to_dict(),
            "language_count": df["original_language"].nunique(),
            "year_min": int(df["release_year"].dropna().min()) if df["release_year"].notna().any() else None,
            "year_max": int(df["release_year"].dropna().max()) if df["release_year"].notna().any() else None,
        }
        return df.copy(), metrics

    np.random.seed(seed)
    df = df.copy()

    # Precompute strata features
    df["_decade"] = df["release_year"].apply(_decade_label)
    df["_media_group"] = df["media_type"].apply(_coarse_media_group)
    df["_lang_group"] = df["original_language"].apply(_coarse_lang_group)

    if media_alloc is None:
        media_alloc = {
            "movie": 0.40,
            "tv": 0.25,
            "anime": 0.30,
            "other": 0.05,
        }

    # Step 1: Compute initial quotas per media group
    media_counts = df["_media_group"].value_counts().to_dict()
    quotas: dict[str, int] = {}
    unallocated = target_total

    remaining_groups = []
    for mg, prop in media_alloc.items():
        avail = media_counts.get(mg, 0)
        desired = int(target_total * prop)
        actual = min(avail, desired)
        quotas[mg] = actual
        unallocated -= actual
        if avail > actual:
            remaining_groups.append(mg)

    # Redistribute any unallocated quota to groups with remaining capacity
    while unallocated > 0 and remaining_groups:
        add_per_group = unallocated // len(remaining_groups)
        if add_per_group == 0:
            add_per_group = 1

        for mg in list(remaining_groups):
            avail = media_counts.get(mg, 0)
            current = quotas[mg]
            capacity = avail - current
            if capacity <= 0:
                remaining_groups.remove(mg)
                continue

            to_add = min(capacity, add_per_group, unallocated)
            quotas[mg] += to_add
            unallocated -= to_add
            if unallocated <= 0:
                break

    logger.info("Sampling target allocations by media group: %s", quotas)

    # Step 2: Pure uniform random sampling within each (Decade × Language) stratum
    sampled_indices: list[Any] = []

    for mg, mg_target in quotas.items():
        mg_df = df[df["_media_group"] == mg]
        if len(mg_df) <= mg_target:
            sampled_indices.extend(mg_df.index.tolist())
            continue

        # Group by (_decade, _lang_group)
        strata = mg_df.groupby(["_decade", "_lang_group"], observed=False)
        stratum_sizes = strata.size()

        # Square-root smoothing boosts historical decades & minority languages
        smoothed_weights = np.sqrt(stratum_sizes)
        smoothed_weights = smoothed_weights / smoothed_weights.sum()

        stratum_targets = (smoothed_weights * mg_target).astype(int)

        diff = mg_target - stratum_targets.sum()
        if diff > 0:
            top_strata = stratum_sizes.nlargest(diff).index
            for s_key in top_strata:
                stratum_targets[s_key] += 1

        # Pure UNIFORM random selection within each stratum — NO POPULARITY WEIGHTING
        for stratum_key, group in strata:
            s_target = stratum_targets.get(stratum_key, 0)
            if s_target <= 0:
                continue

            n_samples = min(len(group), s_target)
            # Random uniform choice without popularity bias
            chosen = group.sample(n=n_samples, random_state=seed).index.tolist()
            sampled_indices.extend(chosen)

    sampled_df = df.loc[sampled_indices].copy()

    # Clean up temporary internal columns
    internal_cols = ["_decade", "_media_group", "_lang_group"]
    sampled_df = sampled_df.drop(columns=internal_cols, errors="ignore")

    metrics = {
        "original_total": len(df),
        "sampled_total": len(sampled_df),
        "target_total": target_total,
        "quotas": quotas,
        "media_breakdown": sampled_df["media_type"].value_counts().to_dict(),
        "language_count": sampled_df["original_language"].nunique(),
        "year_min": int(sampled_df["release_year"].dropna().min()) if sampled_df["release_year"].notna().any() else None,
        "year_max": int(sampled_df["release_year"].dropna().max()) if sampled_df["release_year"].notna().any() else None,
    }

    return sampled_df, metrics


def run_diversity_sampling(target: int = 100000) -> pd.DataFrame:
    """
    Run sampling pipeline: reads candidate pool, applies diversity sampling,
    saves diverse_100k.parquet, and writes sampling report.
    """
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    canonical_path = CANONICAL_DIR / "canonical_entities.parquet"
    candidates_path = CANONICAL_DIR / "candidates.parquet"
    input_path = canonical_path if canonical_path.exists() else candidates_path
    output_path = CANONICAL_DIR / "diverse_100k.parquet"
    report_path = REPORTS_DIR / "sampling_report.md"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found at {input_path}. Run 'process' / 'match' first."
        )

    logger.info("Loading canonical dataset from %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d candidate records.", len(df))

    sampled_df, metrics = sample_diverse_dataset(df, target_total=target)

    # Save sampled dataset
    sampled_df.to_parquet(output_path, index=False)
    logger.info(
        "Diverse dataset saved: %d records → %s",
        len(sampled_df), output_path,
    )

    # Generate sampling report
    report_lines = [
        "# CineMind Stratified Diversity Sampling Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
        f"- **Original Candidate Pool:** {metrics['original_total']:,}",
        f"- **Target Sample Size:** {metrics['target_total']:,}",
        f"- **Final Sampled Titles:** {metrics['sampled_total']:,}",
        f"- **Release Year Range:** {metrics.get('year_min')} – {metrics.get('year_max')}",
        f"- **Unique Languages Represented:** {metrics.get('language_count')}",
        "",
        "## Media Type Quotas & Allocations",
        "",
        "| Media Category | Candidate Pool | Target Allocation | Sampled Count | % of Sample |",
        "|:---------------|---------------:|------------------:|--------------:|------------:|",
    ]

    orig_mg = df["media_type"].apply(_coarse_media_group).value_counts()
    samp_mg = sampled_df["media_type"].apply(_coarse_media_group).value_counts()

    for mg in ["movie", "tv", "anime", "other"]:
        orig_c = orig_mg.get(mg, 0)
        target_c = metrics.get("quotas", {}).get(mg, 0)
        samp_c = samp_mg.get(mg, 0)
        pct = (samp_c / len(sampled_df) * 100) if len(sampled_df) > 0 else 0
        report_lines.append(
            f"| {mg.capitalize()} | {orig_c:,} | {target_c:,} | {samp_c:,} | {pct:.1f}% |"
        )

    report_lines.extend([
        "",
        "## Decade Distribution (Before vs After)",
        "",
        "| Decade | Original Count | Sampled Count | Sample Share |",
        "|:-------|---------------:|--------------:|-------------:|",
    ])

    orig_dec = df["release_year"].apply(_decade_label).value_counts()
    samp_dec = sampled_df["release_year"].apply(_decade_label).value_counts()

    all_decades = sorted(set(orig_dec.index) | set(samp_dec.index))
    for dec in all_decades:
        c1 = orig_dec.get(dec, 0)
        c2 = samp_dec.get(dec, 0)
        pct = (c2 / len(sampled_df) * 100) if len(sampled_df) > 0 else 0
        report_lines.append(f"| {dec} | {c1:,} | {c2:,} | {pct:.1f}% |")

    report_lines.extend([
        "",
        "## Top 20 Languages in Sample",
        "",
        "| Language | Count | Share % |",
        "|:---------|------:|--------:|",
    ])

    top_langs = sampled_df["original_language"].value_counts().head(20)
    for lang, cnt in top_langs.items():
        pct = (cnt / len(sampled_df) * 100) if len(sampled_df) > 0 else 0
        report_lines.append(f"| {lang} | {cnt:,} | {pct:.1f}% |")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Wrote sampling report to %s", report_path)

    return sampled_df
