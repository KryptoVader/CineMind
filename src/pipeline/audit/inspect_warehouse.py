import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path("c:/cinemind/src")
DATA_DIR = SRC_DIR / "data"
CANONICAL_DIR = DATA_DIR / "canonical"

def inspect_warehouse():
    canonical_file = CANONICAL_DIR / "canonical_entities.parquet"
    sampled_file = CANONICAL_DIR / "diverse_100k.parquet"
    links_file = CANONICAL_DIR / "entity_links.parquet"
    match_cand_file = CANONICAL_DIR / "match_candidates.parquet"

    print("=== DATASET FILE EXISTENCE ===")
    print(f"canonical_entities.parquet: {canonical_file.exists()}")
    print(f"diverse_100k.parquet:       {sampled_file.exists()}")
    print(f"entity_links.parquet:       {links_file.exists()}")
    print(f"match_candidates.parquet:   {match_cand_file.exists()}")

    c_df = pd.read_parquet(canonical_file) if canonical_file.exists() else pd.DataFrame()
    s_df = pd.read_parquet(sampled_file) if sampled_file.exists() else pd.DataFrame()
    l_df = pd.read_parquet(links_file) if links_file.exists() else pd.DataFrame()
    mc_df = pd.read_parquet(match_cand_file) if match_cand_file.exists() else pd.DataFrame()

    print(f"\nCanonical Row Count: {len(c_df):,}")
    print(f"Sampled Row Count:   {len(s_df):,}")
    print(f"Links Row Count:     {len(l_df):,}")
    print(f"Match Candidates Count: {len(mc_df):,}")

    if not c_df.empty:
        print("\n--- Canonical Source Presence ---")
        if "source_presence" in c_df.columns:
            sp_str = c_df["source_presence"].apply(lambda x: "+".join(x) if isinstance(x, (list, np.ndarray, tuple)) else str(x))
            print(sp_str.value_counts())
        else:
            print(c_df["source"].value_counts())

        print("\n--- Canonical Media Type ---")
        print(c_df["media_type"].value_counts(dropna=False))

        print("\n--- Canonical Languages (Top 15) ---")
        lang_col = "original_language" if "original_language" in c_df.columns else "language"
        print(c_df[lang_col].value_counts(dropna=False).head(15))

        def decade_label(yr):
            if pd.isna(yr):
                return "Unknown"
            try:
                y = int(yr)
                if y < 1900:
                    return "Pre-1900"
                return f"{(y // 10) * 10}s"
            except (ValueError, TypeError):
                return "Unknown"

        print("\n--- Canonical Decades ---")
        print(c_df["release_year"].apply(decade_label).value_counts())

        print("\n--- Missingness Summary ---")
        print(c_df.isna().sum())

        print("\n--- Available Columns ---")
        print(c_df.columns.tolist())

if __name__ == "__main__":
    inspect_warehouse()
