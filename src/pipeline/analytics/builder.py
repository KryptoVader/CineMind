"""
Analytics View Builder.

Constructs source-separated analytical parquet datasets from diverse_100k.parquet:
1. development_entities.parquet (Complete ~98k universe)
2. development_tmdb.parquet (TMDB-side entities: tmdb + tmdb+mal)
3. development_mal.parquet (MAL-side entities: mal + tmdb+mal)
4. development_shared.parquet (Verified cross-source entities: tmdb+mal)
"""

import logging
from pathlib import Path
import pandas as pd

from pipeline.config import DATA_DIR, CANONICAL_DIR

logger = logging.getLogger(__name__)

ANALYTICS_DIR = DATA_DIR / "analytics"
ROOT_ANALYTICS_DIR = DATA_DIR.parent.parent / "data" / "analytics"


def build_analytics_views() -> dict[str, pd.DataFrame]:
    """Load development sample and export 4 source-separated analytical views."""
    for d in [ANALYTICS_DIR, ROOT_ANALYTICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    input_path = CANONICAL_DIR / "diverse_100k.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Development sample dataset not found at {input_path}")

    dev_df = pd.read_parquet(input_path)
    logger.info("Loaded base development sample: %d records from %s", len(dev_df), input_path)

    # 1. Full Development Universe
    entities_df = dev_df.copy()

    # 2. TMDB View (tmdb or tmdb+mal)
    tmdb_df = dev_df[dev_df["source"].isin(["tmdb", "tmdb+mal"])].copy()

    # 3. MAL View (mal or tmdb+mal)
    mal_df = dev_df[dev_df["source"].isin(["mal", "tmdb+mal"])].copy()

    # 4. Shared View (tmdb+mal only)
    shared_df = dev_df[dev_df["source"] == "tmdb+mal"].copy()

    views = {
        "development_entities.parquet": entities_df,
        "development_tmdb.parquet": tmdb_df,
        "development_mal.parquet": mal_df,
        "development_shared.parquet": shared_df,
    }

    for filename, df in views.items():
        p1 = ANALYTICS_DIR / filename
        p2 = ROOT_ANALYTICS_DIR / filename
        df.to_parquet(p1, index=False)
        df.to_parquet(p2, index=False)
        logger.info("Exported analytical view %s (%d records) to %s and %s", filename, len(df), ANALYTICS_DIR, ROOT_ANALYTICS_DIR)

    return views
