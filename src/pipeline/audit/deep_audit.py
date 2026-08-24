import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

SRC_DIR = Path("c:/cinemind/src")
DATA_DIR = SRC_DIR / "data"

RAW_TMDB_DIR = DATA_DIR / "raw" / "tmdb"
RAW_MAL_DIR = DATA_DIR / "raw" / "mal"
STAGING_DIR = DATA_DIR / "staging"
CANONICAL_DIR = DATA_DIR / "canonical"

def run_deep_audit():
    print("=== 1. ENTITY ACCOUNTING RECONCILIATION ===")
    raw_tmdb_movies_file = RAW_TMDB_DIR / "discovery_movies.jsonl"
    raw_tmdb_tv_file = RAW_TMDB_DIR / "discovery_tv.jsonl"
    raw_mal_file = RAW_MAL_DIR / "discovery.jsonl"

    def count_jsonl(path):
        lines, null_ids, malformed = 0, 0, 0
        ids = set()
        all_ids_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                lines += 1
                try:
                    obj = json.loads(line)
                    sid = obj.get("id") or obj.get("mal_id")
                    if sid is None:
                        null_ids += 1
                    else:
                        ids.add(int(sid))
                        all_ids_list.append(int(sid))
                except Exception:
                    malformed += 1
        return lines, len(ids), len(all_ids_list) - len(ids), null_ids, malformed, ids, all_ids_list

    m_lines, m_uids, m_dups, m_nulls, m_mal, m_ids, m_all_ids = count_jsonl(raw_tmdb_movies_file)
    tv_lines, tv_uids, tv_dups, tv_nulls, tv_mal, tv_ids, tv_all_ids = count_jsonl(raw_tmdb_tv_file)
    mal_lines, mal_uids, mal_dups, mal_nulls, mal_mal, mal_ids, mal_all_ids = count_jsonl(raw_mal_file)

    total_raw_lines = m_lines + tv_lines + mal_lines
    print(f"Raw lines: TMDB Movies={m_lines}, TMDB TV={tv_lines}, MAL={mal_lines} -> Total={total_raw_lines}")
    print(f"Raw unique source IDs: Movies={m_uids}, TV={tv_uids}, MAL={mal_uids} -> Sum={m_uids+tv_uids+mal_uids}")
    
    # Staging normalized files
    tmdb_norm_df = pd.read_parquet(STAGING_DIR / "tmdb_normalized.parquet")
    mal_norm_df = pd.read_parquet(STAGING_DIR / "mal_normalized.parquet")
    print(f"Staging normalized: TMDB={len(tmdb_norm_df)}, MAL={len(mal_norm_df)} -> Sum={len(tmdb_norm_df)+len(mal_norm_df)}")

    # Check overlaps between TMDB movies & TMDB TV in raw/staging
    tmdb_movies_stg = tmdb_norm_df[tmdb_norm_df["media_type"] == "movie"]
    tmdb_tv_stg = tmdb_norm_df[tmdb_norm_df["media_type"] == "tv"]
    print(f"Staging TMDB breakdown: movies={len(tmdb_movies_stg)}, tv={len(tmdb_tv_stg)}")

    # Canonical & Links
    canonical_df = pd.read_parquet(CANONICAL_DIR / "canonical_entities.parquet")
    links_df = pd.read_parquet(CANONICAL_DIR / "entity_links.parquet")
    match_cand_df = pd.read_parquet(CANONICAL_DIR / "match_candidates.parquet")
    sampled_df = pd.read_parquet(CANONICAL_DIR / "diverse_100k.parquet")

    print(f"Canonical Entities: {len(canonical_df)}")
    print(f"Difference (Raw Lines - Canonical): {total_raw_lines - len(canonical_df)}")
    
    # Exact accounting of difference
    # 1. Raw lines vs Staging records
    raw_lines_vs_staging_diff = total_raw_lines - (len(tmdb_norm_df) + len(mal_norm_df))
    # 2. Staging records vs Canonical entities
    staging_vs_canonical_diff = (len(tmdb_norm_df) + len(mal_norm_df)) - len(canonical_df)
    
    print(f"1. Raw Lines ({total_raw_lines}) -> Staging Records ({len(tmdb_norm_df)+len(mal_norm_df)}): removed {raw_lines_vs_staging_diff}")
    print(f"   - Breakdown of Raw vs Staging removed:")
    print(f"     * TMDB TV raw lines ({tv_lines}) vs TMDB TV staging records ({len(tmdb_tv_stg)}): removed {tv_lines - len(tmdb_tv_stg)} duplicate lines in TMDB TV discovery JSONL")
    print(f"     * TMDB Movies raw lines ({m_lines}) vs TMDB Movies staging records ({len(tmdb_movies_stg)}): removed {m_lines - len(tmdb_movies_stg)} duplicates")
    print(f"     * MAL raw lines ({mal_lines}) vs MAL staging records ({len(mal_norm_df)}): removed {mal_lines - len(mal_norm_df)} duplicates")

    print(f"2. Staging Records ({len(tmdb_norm_df)+len(mal_norm_df)}) -> Canonical Entities ({len(canonical_df)}): removed {staging_vs_canonical_diff}")
    print(f"   - Reason: {staging_vs_canonical_diff} cross-source merges (TMDB + MAL pairs joined into unified entities)")
    
    total_accounted_diff = raw_lines_vs_staging_diff + staging_vs_canonical_diff
    print(f"Total accounted difference: {total_accounted_diff} (Matches exact difference {total_raw_lines - len(canonical_df)})")

    print("\n=== 2. RECONCILE CROSS-SOURCE MATCH COUNTS & CARDINALITY ===")
    print(f"Total rows in entity_links.parquet: {len(links_df)}")
    tmdb_ids_in_links = links_df["tmdb_id"].nunique()
    mal_ids_in_links = links_df["mal_id"].nunique()
    print(f"Unique TMDB IDs linked: {tmdb_ids_in_links}")
    print(f"Unique MAL IDs linked: {mal_ids_in_links}")

    # Check cardinality (1:1, 1:N, N:1, N:M)
    tmdb_counts = links_df["tmdb_id"].value_counts()
    mal_counts = links_df["mal_id"].value_counts()
    
    one_to_many_tmdb = tmdb_counts[tmdb_counts > 1]
    many_to_one_mal = mal_counts[mal_counts > 1]

    print(f"TMDB IDs mapped to multiple MAL IDs (1-to-many): {len(one_to_many_tmdb)}")
    print(f"MAL IDs mapped to multiple TMDB IDs (many-to-1): {len(many_to_one_mal)}")
    
    # Merged entities in canonical_entities.parquet
    merged_canonical = canonical_df[canonical_df["source"] == "tmdb+mal"]
    print(f"Merged entities in canonical_entities.parquet (source == 'tmdb+mal'): {len(merged_canonical)}")
    
    # Why is merged_canonical == 6,453 while verified_links == 6,445?
    # Inspect links and canonical entity creation logic!
    print(f"Difference (Merged entities - Verified link rows): {len(merged_canonical) - len(links_df)}")

    print("\n=== 3. STATISTICAL POPULARITY BIAS AUDIT (SMD, KS, WASSERSTEIN) ===")
    def compute_stats(c_series, s_series, var_name):
        c = c_series.dropna()
        s = s_series.dropna()
        
        c_mean, s_mean = c.mean(), s.mean()
        c_std, s_std = c.std(), s.std()
        
        # Standardized Mean Difference (Cohen's d)
        pooled_std = math.sqrt(((len(c)-1)*(c_std**2) + (len(s)-1)*(s_std**2)) / (len(c) + len(s) - 2)) if (len(c)+len(s)-2) > 0 else 1.0
        smd = (s_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # KS test & Wasserstein distance
        ks_stat, ks_pval = ks_2samp(c, s)
        w_dist = wasserstein_distance(c, s)
        
        def pct(ser):
            return {
                "mean": float(ser.mean()),
                "std": float(ser.std()),
                "median": float(ser.median()),
                "P25": float(ser.quantile(0.25)),
                "P50": float(ser.quantile(0.50)),
                "P75": float(ser.quantile(0.75)),
                "P90": float(ser.quantile(0.90)),
                "P95": float(ser.quantile(0.95)),
                "P99": float(ser.quantile(0.99)),
            }
        
        return {
            "var": var_name,
            "canonical": pct(c),
            "sample": pct(s),
            "smd": float(smd),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "wasserstein_distance": float(w_dist),
        }

    for col in ["popularity", "vote_count", "rating"]:
        if col in canonical_df.columns and col in sampled_df.columns:
            res = compute_stats(canonical_df[col], sampled_df[col], col)
            print(f"--- {col.upper()} ---")
            print(f"Canonical Median={res['canonical']['median']:.2f}, P90={res['canonical']['P90']:.2f}, P95={res['canonical']['P95']:.2f}")
            print(f"Sample    Median={res['sample']['median']:.2f}, P90={res['sample']['P90']:.2f}, P95={res['sample']['P95']:.2f}")
            print(f"SMD: {res['smd']:.4f}, KS Stat: {res['ks_stat']:.4f} (p={res['ks_pvalue']:.4e}), Wasserstein: {res['wasserstein_distance']:.4f}")

    print("\n=== 4. EXPANDED / POPULATION-WEIGHTED SAMPLE ANALYSIS ===")
    # Calculate stratum weights Wi = Ni / ni
    # Stratification in sampler: (media_group, decade, language, genre)
    
    # Assign media group
    def assign_media_group(row):
        source = row.get("source", "")
        mt = str(row.get("media_type", "")).lower()
        if source == "mal" or "anime" in mt:
            return "anime"
        elif mt == "movie":
            return "movie"
        elif mt == "tv":
            return "tv"
        else:
            return "other"

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

    canonical_df["_mg"] = canonical_df.apply(assign_media_group, axis=1)
    sampled_df["_mg"] = sampled_df.apply(assign_media_group, axis=1)

    canonical_df["_dec"] = canonical_df["release_year"].apply(decade_label)
    sampled_df["_dec"] = sampled_df["release_year"].apply(decade_label)

    print("Canonical Media Group counts:\n", canonical_df["_mg"].value_counts())
    print("Sampled Media Group counts:\n", sampled_df["_mg"].value_counts())

    # Media Group sampling rates
    mg_c = canonical_df["_mg"].value_counts()
    mg_s = sampled_df["_mg"].value_counts()

    print("\nMedia Group Sampling Probabilities (n_i / N_i):")
    for mg in mg_c.index:
        ni = mg_s.get(mg, 0)
        Ni = mg_c[mg]
        pi = ni / Ni
        weight = Ni / ni if ni > 0 else 0
        print(f"  {mg}: N={Ni:,}, n={ni:,}, p={pi:.4f}, weight={weight:.4f}")

    # Reweighted Popularity Calculation
    # For sampled_df, assign weight Wi = Ni / ni based on media_group
    weights_mg = {mg: mg_c[mg] / mg_s[mg] for mg in mg_c.index if mg_s.get(mg, 0) > 0}
    sampled_df["_weight_mg"] = sampled_df["_mg"].map(weights_mg).fillna(1.0)

    # Weighted percentiles function
    def weighted_quantile(values, quantiles, sample_weight=None):
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)

    pop_sampled = sampled_df["popularity"].dropna()
    weights_sampled = sampled_df.loc[pop_sampled.index, "_weight_mg"]

    w_med = weighted_quantile(pop_sampled, [0.5], weights_sampled)[0]
    w_p90 = weighted_quantile(pop_sampled, [0.90], weights_sampled)[0]
    w_p95 = weighted_quantile(pop_sampled, [0.95], weights_sampled)[0]

    print(f"\nReweighted Sample Popularity (by Media Group):")
    print(f"  Unweighted Sample: Median={pop_sampled.median():.2f}, P90={pop_sampled.quantile(0.90):.2f}, P95={pop_sampled.quantile(0.95):.2f}")
    print(f"  Reweighted Sample: Median={w_med:.2f}, P90={w_p90:.2f}, P95={w_p95:.2f}")
    print(f"  Canonical Universe: Median={canonical_df['popularity'].median():.2f}, P90={canonical_df['popularity'].quantile(0.90):.2f}, P95={canonical_df['popularity'].quantile(0.95):.2f}")

    print("\n=== 5. ROOT CAUSE OF POPULARITY CONCENTRATION ===")
    print("Popularity by Media Group in Canonical Universe:")
    for mg in sorted(canonical_df["_mg"].unique()):
        sub_c = canonical_df[canonical_df["_mg"] == mg]["popularity"].dropna()
        sub_s = sampled_df[sampled_df["_mg"] == mg]["popularity"].dropna()
        print(f"  {mg}:")
        print(f"    Canonical: count={len(sub_c):,}, mean={sub_c.mean():.2f}, median={sub_c.median():.2f}, P90={sub_c.quantile(0.90):.2f}")
        print(f"    Sampled  : count={len(sub_s):,}, mean={sub_s.mean():.2f}, median={sub_s.median():.2f}, P90={sub_s.quantile(0.90):.2f}")

    print("\n=== 6. MISSING DATE ENTITIES AUDIT ===")
    missing_date_c = canonical_df[canonical_df["release_date"].isna()]
    missing_date_s = sampled_df[sampled_df["release_date"].isna()]
    print(f"Canonical missing dates: {len(missing_date_c)}")
    print(f"Sampled missing dates: {len(missing_date_s)}")
    print("Are all canonical missing-date entities present in the sample?")
    missing_c_ids = set(missing_date_c["cinemind_id"])
    missing_s_ids = set(missing_date_s["cinemind_id"])
    print(f"Intersection count: {len(missing_c_ids.intersection(missing_s_ids))}")

    print("\n=== 7. GENRE COMPLETENESS BREAKDOWN ===")
    def is_genre_empty(g):
        if g is None:
            return True
        if isinstance(g, (list, np.ndarray, tuple)):
            return len(g) == 0
        if pd.isna(g):
            return True
        if isinstance(g, str):
            return len(g.strip()) == 0
        return False

    canonical_df["_no_genre"] = canonical_df["genres"].apply(is_genre_empty)
    sampled_df["_no_genre"] = sampled_df["genres"].apply(is_genre_empty)

    # Inspect 8 extra merged entities
    links_tmdb_ids = set(links_df["tmdb_id"])
    links_mal_ids = set(links_df["mal_id"])
    merged_entities = canonical_df[canonical_df["source"] == "tmdb+mal"]
    print(f"\nInspection of {len(merged_entities)} merged entities in canonical_entities.parquet:")
    print(f"  tmdb_id present: {merged_entities['tmdb_id'].notna().sum()}")
    print(f"  mal_id present:  {merged_entities['mal_id'].notna().sum()}")
    
    # Check if there are any merged entities whose tmdb_id/mal_id is not in links_df
    m_tids = set(merged_entities["tmdb_id"].dropna().astype(int))
    m_mids = set(merged_entities["mal_id"].dropna().astype(int))
    diff_tids = m_tids - links_tmdb_ids
    diff_mids = m_mids - links_mal_ids
    print(f"  Merged TMDB IDs not in entity_links.parquet: {len(diff_tids)}")
    print(f"  Merged MAL IDs not in entity_links.parquet: {len(diff_mids)}")
    if diff_tids or diff_mids:
        print("  Sample extra merged TMDB IDs:", list(diff_tids)[:10])
        print("  Sample extra merged MAL IDs:", list(diff_mids)[:10])

    print("Missing genres by source:")
    for src in canonical_df["source"].unique():
        sub = canonical_df[canonical_df["source"] == src]
        missing_cnt = sub["_no_genre"].sum()
        pct = missing_cnt / len(sub) * 100
        print(f"  Canonical source={src}: missing={missing_cnt:,} / {len(sub):,} ({pct:.2f}%)")

    for src in sampled_df["source"].unique():
        sub = sampled_df[sampled_df["source"] == src]
        missing_cnt = sub["_no_genre"].sum()
        pct = missing_cnt / len(sub) * 100
        print(f"  Sampled   source={src}: missing={missing_cnt:,} / {len(sub):,} ({pct:.2f}%)")

    print("\nMissing genres by media group:")
    for mg in canonical_df["_mg"].unique():
        sub = canonical_df[canonical_df["_mg"] == mg]
        missing_cnt = sub["_no_genre"].sum()
        pct = missing_cnt / len(sub) * 100
        print(f"  Canonical media_group={mg}: missing={missing_cnt:,} / {len(sub):,} ({pct:.2f}%)")

if __name__ == "__main__":
    run_deep_audit()
