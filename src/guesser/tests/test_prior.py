"""
Unit test for source-aware popularity prior normalization.
Asserts that famous entities from both TMDB and MAL land in comparable high prior percentile ranges.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from guesser.belief import BeliefTracker


class MockKnowledgeBase:
    """Lightweight KnowledgeBase stub for prior distribution testing."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_entities = len(df)


def test_prior_normalization_comparability():
    # Load dataset
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    dev_path = data_dir / "analytics" / "development_entities.parquet"
    if not dev_path.exists():
        dev_path = data_dir / "canonical" / "canonical_entities.parquet"

    try:
        df = pd.read_parquet(dev_path)
    except Exception:
        df = pd.read_parquet(dev_path, engine="fastparquet")

    kb = MockKnowledgeBase(df)
    tracker = BeliefTracker(kb)

    # Compute prior percentiles across all entities
    priors = np.exp(tracker.log_priors)
    prior_series = pd.Series(priors)
    pct_ranks = prior_series.rank(pct=True).values

    # Check 3 multi-tier cross-source pairs across TMDB and MAL
    pairs = [
        ("High Popularity Tier", "Inception", "Death Note", 0.15),
        ("Mid Popularity Tier", "Coherence", "Serial Experiments Lain", 0.15),
        ("Niche / Cult Tier", "Primer", "Kaiba", 0.15),
    ]

    print("\n=== CROSS-SOURCE PRIOR NORMALIZATION VERIFICATION ===")
    print(f"{'Tier':<22s} | {'TMDB Title':<24s} (Pct) | {'MAL Title':<25s} (Pct) | Diff")
    print("-" * 90)

    for tier, tmdb_title, mal_title, max_diff in pairs:
        t_match = df[(df["source"].isin(["tmdb", "tmdb+mal"])) & (df["title"].str.lower() == tmdb_title.lower())]
        if t_match.empty:
            t_match = df[(df["source"].isin(["tmdb", "tmdb+mal"])) & (df["title"].str.contains(tmdb_title, case=False, na=False))]

        m_match = df[(df["source"].isin(["mal", "tmdb+mal"])) & (df["title"].str.lower() == mal_title.lower())]
        if m_match.empty:
            m_match = df[(df["source"].isin(["mal", "tmdb+mal"])) & (df["title"].str.contains(mal_title, case=False, na=False))]

        assert not t_match.empty, f"TMDB entity '{tmdb_title}' not found"
        assert not m_match.empty, f"MAL entity '{mal_title}' not found"

        t_idx = t_match.index[0]
        m_idx = m_match.index[0]

        t_pct = pct_ranks[t_idx]
        m_pct = pct_ranks[m_idx]

        diff = abs(t_pct - m_pct)
        t_name = t_match.iloc[0]["title"][:22]
        m_name = m_match.iloc[0]["title"][:22]

        print(f"{tier:<22s} | {t_name:<24s} ({t_pct*100:6.2f}%) | {m_name:<25s} ({m_pct*100:6.2f}%) | {diff*100:5.2f}%")
        assert diff <= max_diff, f"Cross-source prior imbalance for {tier}: TMDB={t_pct:.4f}, MAL={m_pct:.4f}, diff={diff:.4f}"

    print("=" * 90)
    print("Prior normalization multi-tier test PASSED!")


if __name__ == "__main__":
    test_prior_normalization_comparability()
    print("Prior normalization test PASSED!")
