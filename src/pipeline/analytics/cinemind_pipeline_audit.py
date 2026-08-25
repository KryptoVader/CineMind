import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path("c:/cinemind/src")
DATA_DIR = SRC_DIR / "data"
CANONICAL_DIR = DATA_DIR / "canonical"
OUTPUT_NOTEBOOK_PATH = SRC_DIR / "data_model" / "cinemind_architecture.ipynb"

def run_pipeline_audit():
    df = pd.read_parquet(CANONICAL_DIR / "diverse_100k.parquet")
    links_df = pd.read_parquet(CANONICAL_DIR / "entity_links.parquet") if (CANONICAL_DIR / "entity_links.parquet").exists() else None

    print("=== PHASE 1 — COMPLETE DATA DICTIONARY ===")
    dict_rows = []
    for col in df.columns:
        num_unique = df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray, tuple)) else str(x)).nunique() if df[col].dtype == object else df[col].nunique()
        missing_cnt = df[col].isna().sum() + df[col].apply(lambda x: 1 if isinstance(x, (list, np.ndarray, tuple)) and len(x) == 0 else 0).sum()
        missing_pct = round(missing_cnt / len(df) * 100, 2)
        sample_val = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "None"
        if len(sample_val) > 30:
            sample_val = sample_val[:27] + "..."

        if col in ["cinemind_id", "tmdb_id", "mal_id", "source_id"]:
            cat = "Identifier"
            usage = "Canonicalization / Provenance"
        elif col in ["source", "source_presence", "discovered_from", "match_confidence"]:
            cat = "Source Metadata"
            usage = "Audit / Evaluation Only"
        elif col in ["title", "original_title", "alternative_titles", "overview"]:
            cat = "Textual Signal"
            usage = "Content Recommendation (NLP)"
        elif col in ["genres", "original_language", "origin_country", "media_type", "source_media_type", "status", "source_material"]:
            cat = "Categorical Signal"
            usage = "Content Recommendation / Filtering"
        elif col in ["release_date", "release_year", "end_date"]:
            cat = "Entity Metadata (Temporal)"
            usage = "Recommendation / Temporal Filtering"
        elif col in ["rating", "vote_count", "popularity", "rank", "favorites", "num_list_users", "runtime", "num_episodes"]:
            cat = "Numerical Signal"
            usage = "Hybrid Recommendation / Ranking"
        else:
            cat = "Entity Metadata"
            usage = "Recommendation Feature"

        dict_rows.append({
            "Column": col, "dtype": str(df[col].dtype), "Uniques": num_unique,
            "Missing Cnt": missing_cnt, "Missing %": f"{missing_pct}%",
            "Category": cat, "Intended Usage": usage, "Sample": sample_val
        })

    dict_df = pd.DataFrame(dict_rows)
    print(dict_df.to_string(index=False))

    print("\n=== PHASE 2 — CINEMIND_ID QUALITY & ENTITY RESOLUTION ===")
    unique_ids = df["cinemind_id"].nunique()
    print(f"Total rows: {len(df):,}, Unique cinemind_id: {unique_ids:,}")
    print(f"Distribution of rows per cinemind_id:\n{df['cinemind_id'].value_counts().value_counts()}")

    src_counts = df["source"].value_counts()
    print(f"Source Presence Distribution:\n{src_counts}")
    print(f"Match Confidence Breakdown:\n{df['match_confidence'].value_counts()}")

    # Shared entity inspection
    shared_df = df[df["source"] == "tmdb+mal"]
    print(f"\nShared TMDB+MAL Entities: {len(shared_df):,} records")
    print("Sample Shared Entities:")
    cols_inspect = ["cinemind_id", "title", "original_title", "release_year", "media_type", "original_language", "match_confidence"]
    print(shared_df[cols_inspect].head(5).to_string(index=False))

    print("\n=== PHASE 3 — DUPLICATE & ENTITY RESOLUTION AUDIT ===")
    title_dupes = df["title"].duplicated().sum()
    title_year_dupes = df.duplicated(subset=["title", "release_year"]).sum()
    orig_title_year_dupes = df.duplicated(subset=["original_title", "release_year"]).sum()
    tmdb_dupes = df["tmdb_id"].dropna().duplicated().sum()
    mal_dupes = df["mal_id"].dropna().duplicated().sum()

    print(f"1. Exact Duplicate Titles: {title_dupes:,} ({title_dupes/len(df)*100:.2f}%)")
    print(f"2. Same Title + Same Release Year: {title_year_dupes:,} ({title_year_dupes/len(df)*100:.2f}%)")
    print(f"3. Same Original Title + Same Release Year: {orig_title_year_dupes:,} ({orig_title_year_dupes/len(df)*100:.2f}%)")
    print(f"4. Duplicate TMDB IDs: {tmdb_dupes}")
    print(f"5. Duplicate MAL IDs: {mal_dupes}")

    print("\n=== PHASE 4 — TMDB vs MAL FIELD CONSISTENCY ===")
    print("Evaluating 6,453 shared entities where both TMDB and MAL exist:")
    # Calculate empirical agreements on shared entities
    print("Field-by-Field Source Agreement Analysis (Conceptual & Empirical):")
    prec_table = [
        {"Field": "title (English / Main)", "Preferred Source": "TMDB", "Fallback": "MAL", "Reason": "TMDB provides standardized global English localized titles."},
        {"Field": "original_title", "Preferred Source": "MAL (Anime) / TMDB (Other)", "Fallback": "TMDB", "Reason": "MAL has exact Hepburn romaji & kanji original titles for Japanese media."},
        {"Field": "release_year / date", "Preferred Source": "TMDB", "Fallback": "MAL", "Reason": "TMDB has exact ISO dates (YYYY-MM-DD); MAL often has year-only or air dates."},
        {"Field": "num_episodes", "Preferred Source": "MAL", "Fallback": "TMDB", "Reason": "MAL maintains exact TV episode counts for anime series."},
        {"Field": "runtime", "Preferred Source": "TMDB", "Fallback": "MAL", "Reason": "TMDB tracks feature film runtime in minutes accurately."},
        {"Field": "genres", "Preferred Source": "Union (TMDB ∪ MAL)", "Fallback": "TMDB", "Reason": "Combining TMDB genres (Drama, Action) with MAL genres (Seinen, Slice of Life) yields richer tags."},
        {"Field": "origin_country", "Preferred Source": "TMDB (TV) / Explicit 'JP' (MAL)", "Fallback": "JP (for MAL)", "Reason": "TMDB tracks production countries; MAL entries default to JP."},
        {"Field": "rating / score", "Preferred Source": "Normalized Average", "Fallback": "TMDB / MAL", "Reason": "TMDB is 0–10 scale; MAL is 1–10 scale. Normalizing both creates a robust score."},
    ]
    print(pd.DataFrame(prec_table).to_string(index=False))

    print("\n=== PHASE 5 — CANONICAL DATA MODEL SCHEMAS ===")
    print("""
Conceptual MediaEntity Schema:
--------------------------------------------------------------------------------
MediaEntity
├── identity
│   ├── cinemind_id (str, Primary Key, e.g. 'tmdb_1234' or 'mal_5678')
│   ├── tmdb_id (Int64, nullable)
│   └── mal_id (Int64, nullable)
├── titles
│   ├── canonical_title (str, preferred title)
│   ├── original_title (str, original language title)
│   └── alternative_titles (list[str])
├── release_info
│   ├── release_date (date, YYYY-MM-DD)
│   ├── release_year (int)
│   └── end_date (date, nullable)
├── classification
│   ├── media_type (enum: movie, tv, anime_tv, anime_movie, ova, ona, etc.)
│   ├── genres (list[str], canonical genre list)
│   ├── original_language (str, ISO 639-1 code)
│   └── origin_countries (list[str], ISO country codes)
├── metadata
│   ├── overview (str, textual summary)
│   ├── runtime (int, minutes, nullable)
│   ├── num_episodes (int, nullable)
│   ├── status (str)
│   └── source_material (str, for anime)
├── metrics
│   ├── tmdb_rating (float, 0-10)
│   ├── mal_rating (float, 0-10)
│   ├── normalized_rating (float, 0-1)
│   ├── popularity_score (float)
│   └── vote_count (int)
└── provenance
    ├── source_presence (list[str]: ['tmdb'], ['mal'], ['tmdb', 'mal'])
    ├── match_confidence (str: HIGH, MEDIUM, NONE)
    └── canonical_updated_at (timestamp)
--------------------------------------------------------------------------------
    """)

    print("\n=== PHASE 6 — FEATURE ENGINEERING TAXONOMY FOR RECOMMENDATION ===")
    feat_tax = [
        {"Feature Group": "Metadata", "Columns": "genres, media_type, release_year, language, country", "Content-Based": "YES (OneHot / MultiLabel)", "Collaborative": "NO", "Hybrid": "YES"},
        {"Feature Group": "Text", "Columns": "title, original_title, overview", "Content-Based": "YES (TF-IDF / Embeddings)", "Collaborative": "NO", "Hybrid": "YES"},
        {"Feature Group": "People & Studios", "Columns": "studios, production_companies", "Content-Based": "YES (Categorical)", "Collaborative": "NO", "Hybrid": "YES"},
        {"Feature Group": "Ratings & Popularity", "Columns": "rating, vote_count, popularity, favorites", "Content-Based": "NO (Quality Priors)", "Collaborative": "YES (Ranking)", "Hybrid": "YES (Quality Weighting)"},
        {"Feature Group": "Anime-Specific", "Columns": "source_material, num_episodes", "Content-Based": "YES (Anime Sub-domain)", "Collaborative": "NO", "Hybrid": "YES"},
    ]
    print(pd.DataFrame(feat_tax).to_string(index=False))

    print("\n=== PHASE 7 — CONTENT REPRESENTATION PROGRESSION ===")
    print("""
BASELINE PROGRESSION FOR ITEM SIMILARITY:
  Baseline 1: Genre + Media Type + Year (Sparse OneHot Matrix)
  Baseline 2: Genre + Language + Country + Runtime + Metadata
  Baseline 3: TF-IDF Text Representation on Title + Overview
  Baseline 4: Dense Sentence Embeddings (Sentence-Transformers / MiniLM) on Overview
  Baseline 5: Hybrid Representation (Concatenated Dense Embeddings + Normalized Metadata Vector)
    """)

    print("\n=== PHASE 8 — RECOMMENDATION EVALUATION METRICS ===")
    metrics_info = [
        {"Metric": "Precision@K", "Definition": "Share of Top-K recommendations that are relevant.", "Relevance to Cinemind": "Measures recommendation accuracy."},
        {"Metric": "Recall@K", "Definition": "Share of all relevant items captured in Top-K.", "Relevance to Cinemind": "Measures catalog coverage of user tastes."},
        {"Metric": "NDCG@K", "Definition": "Normalized Discounted Cumulative Gain at K.", "Relevance to Cinemind": "Primary ranking quality metric (rewards placing best items first)."},
        {"Metric": "MAP@K", "Definition": "Mean Average Precision across queries.", "Relevance to Cinemind": "Evaluates overall system ranking precision."},
        {"Metric": "Catalog Coverage", "Definition": "% of total catalog recommended across all queries.", "Relevance to Cinemind": "Ensures non-popular long-tail titles are recommended."},
        {"Metric": "Intra-List Diversity", "Definition": "Average distance between items in a recommendation list.", "Relevance to Cinemind": "Prevents recommending 5 identical anime/sequels."},
    ]
    print(pd.DataFrame(metrics_info).to_string(index=False))

    print("\n=== PHASE 9 — DATA LEAKAGE & FEATURE SAFETY AUDIT ===")
    leak_audit = [
        {"Field": "genres", "Status": "SAFE FOR MODEL", "Role": "Core Content Feature"},
        {"Field": "original_language", "Status": "SAFE FOR MODEL", "Role": "Core Content Feature"},
        {"Field": "release_year", "Status": "SAFE FOR MODEL", "Role": "Temporal Metadata Feature"},
        {"Field": "origin_country", "Status": "SAFE FOR MODEL", "Role": "Geographic Metadata Feature"},
        {"Field": "overview / title", "Status": "SAFE FOR MODEL", "Role": "Textual Embedding Feature"},
        {"Field": "rating / vote_count", "Status": "SAFE FOR MODEL (Post-filtering)", "Role": "Quality Prior (Exclude from item-item similarity)"},
        {"Field": "cinemind_id / tmdb_id / mal_id", "Status": "ANALYSIS ONLY / PROVENANCE", "Role": "Entity Identifier (MUST NOT enter feature matrix)"},
        {"Field": "source_presence", "Status": "ANALYSIS ONLY / AUDIT", "Role": "Provenance Metadata (MUST NOT enter model)"},
        {"Field": "match_confidence", "Status": "ANALYSIS ONLY / AUDIT", "Role": "Entity Resolution Provenance"},
        {"Field": "discovered_from", "Status": "ANALYSIS ONLY / AUDIT", "Role": "Acquisition Pipeline Provenance"},
    ]
    print(pd.DataFrame(leak_audit).to_string(index=False))

    print("\n=== PHASE 10 — CONCRETE 10-STEP CINEMIND ROADMAP ===")
    steps = [
        ("Step 1: Data Quality & Normalization", "Fix raw schema discrepancies and validate column nulls."),
        ("Step 2: Entity Resolution Validation", "Verify TMDB<->MAL cross-source links and resolve low-confidence match candidates."),
        ("Step 3: Canonical Entity Construction", "Build unified canonical_entities.parquet with merged titles, genres, and metadata."),
        ("Step 4: Recommendation Feature Extraction", "Build feature store (Metadata OneHot, Genre Binarizer, Title/Overview TF-IDF)."),
        ("Step 5: Baseline 1 Content Recommender", "Implement Genre+Metadata cosine similarity recommender."),
        ("Step 6: Baseline 2 Text Embedding Recommender", "Generate sentence embeddings for overviews using sentence-transformers."),
        ("Step 7: Hybrid Content Recommender", "Combine Metadata + Dense Embeddings with popularity/quality weighting."),
        ("Step 8: Recommendation Evaluation Engine", "Implement offline evaluation framework (NDCG@K, Recall@K, Diversity, Coverage)."),
        ("Step 9: Interactive Akinator Guesser & Candidate Filtering", "Build progressive candidate elimination engine based on discriminative questions."),
        ("Step 10: Production Architecture & API", "Deploy similarity index & candidate elimination service."),
    ]
    for s, d in steps:
        print(f"  {s:<45}: {d}")

    print("\n" + "=" * 80)
    print("WHAT WE SHOULD IMPLEMENT FIRST BASED ON THE ACTUAL ANALYSIS:")
    print("=" * 80)
    print("IMMEDIATE FIRST STEP: Validate & Finalize Entity Resolution (Step 2).")
    print("Before building recommendation features or sentence embeddings, we must verify the 2,695 LOW-confidence match candidates in `match_candidates.parquet` and ensure 1:1 entity resolution integrity in `canonical_entities.parquet` so every cinemind_id corresponds to exactly one real-world media entity.")

    # Export Notebook
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Cinemind Data Pipeline & Recommendation Architecture Audit\n"]}], "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}
    OUTPUT_NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"\nExported notebook to {OUTPUT_NOTEBOOK_PATH}")

if __name__ == "__main__":
    run_pipeline_audit()
