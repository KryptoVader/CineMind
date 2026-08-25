"""
CineMind Incremental Experiments Suite (Phase 6).

Runs 3 incremental experiments:
Experiment 1: Metadata-only questions (55+ expanded definitions)
Experiment 2: Metadata + TF-IDF Concept Cluster questions (200 clusters)
Experiment 3: Full Architecture (Metadata + Concept Clusters + Contrastive Keywords)

Measures Accuracy@5, Accuracy@10, Accuracy@15, Accuracy@20, Mean Questions, and Median Questions.
"""

import argparse
from typing import Any
import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR, CANONICAL_DIR
from guesser.knowledge import KnowledgeBase
from guesser.belief import BeliefTracker
from guesser.generators import MetadataGenerator, ConceptClusterGenerator, ContrastiveGenerator, CandidateQuestion
from guesser.engine import CineMindGuesserEngine

ANALYTICS_DIR = DATA_DIR / "analytics"


def load_dataset() -> pd.DataFrame:
    """Load development entities dataset."""
    p1 = ANALYTICS_DIR / "development_entities.parquet"
    p2 = CANONICAL_DIR / "canonical_entities.parquet"
    if p1.exists():
        return pd.read_parquet(p1)
    elif p2.exists():
        return pd.read_parquet(p2)
    else:
        raise FileNotFoundError("Development dataset not found.")


def run_experiment(engine: CineMindGuesserEngine, df: pd.DataFrame, exp_name: str, enabled_generators: list[str], sample_indices: np.ndarray) -> dict[str, Any]:
    """Run a single experimental configuration using the shared engine KnowledgeBase."""
    print(f"\n" + "=" * 64)
    print(f"RUNNING {exp_name.upper()}")
    print("=" * 64)

    correct_cnt = 0
    q_counts = []

    acc_at_5 = 0
    acc_at_10 = 0
    acc_at_15 = 0
    acc_at_20 = 0

    for game_num, target_idx in enumerate(sample_indices, 1):
        target = df.iloc[target_idx].to_dict()
        engine.reset()
        q_count = 0

        hit_at_5 = False
        hit_at_10 = False
        hit_at_15 = False
        hit_at_20 = False

        while not engine.should_guess(max_questions=30):
            candidate_qs: list[CandidateQuestion] = []

            if "metadata" in enabled_generators:
                candidate_qs.extend(engine.meta_gen.generate_questions(engine.asked_q_ids))
            if "concept" in enabled_generators:
                candidate_qs.extend(engine.concept_gen.generate_questions(engine.asked_q_ids))
            if "contrastive" in enabled_generators:
                top_k = engine.tracker.get_top_candidates(k=2)
                if top_k[0][1] >= 0.005 or len(engine.tracker.history) >= 6:
                    candidate_qs.extend(engine.contrast_gen.generate_questions(engine.asked_q_ids, top_candidates_k=5))

            if not candidate_qs:
                break

            best_q = engine.select_best_question(candidate_qs)
            if best_q is None:
                break

            p_yes_vec = best_q.get_p_yes_fn()
            is_yes = float(p_yes_vec[target_idx]) >= 0.30
            ans_str = "yes" if is_yes else "no"

            engine.answer_question(best_q, ans_str)
            q_count += 1

            # Check milestone accuracy
            top_c = engine.get_top_candidates(k=1)[0][0]
            if top_c["cinemind_id"] == target["cinemind_id"]:
                if q_count <= 5: hit_at_5 = True
                if q_count <= 10: hit_at_10 = True
                if q_count <= 15: hit_at_15 = True
                if q_count <= 20: hit_at_20 = True

        if hit_at_5: acc_at_5 += 1
        if hit_at_10: acc_at_10 += 1
        if hit_at_15: acc_at_15 += 1
        if hit_at_20: acc_at_20 += 1

        top_cand, top_p = engine.get_top_candidates(k=1)[0]
        is_correct = top_cand["cinemind_id"] == target["cinemind_id"]
        if is_correct:
            correct_cnt += 1

        # Requirement 7: Wire feedback logging into experiments
        from guesser.feedback import log_game_feedback
        log_game_feedback(engine.tracker.history, top_cand["cinemind_id"], is_correct, target["cinemind_id"])

        q_counts.append(q_count)

    total_g = len(sample_indices)
    metrics = {
        "exp_name": exp_name,
        "total": total_g,
        "accuracy": correct_cnt / total_g * 100,
        "acc_at_5": acc_at_5 / total_g * 100,
        "acc_at_10": acc_at_10 / total_g * 100,
        "acc_at_15": acc_at_15 / total_g * 100,
        "acc_at_20": acc_at_20 / total_g * 100,
        "mean_q": float(np.mean(q_counts)),
        "median_q": float(np.median(q_counts)),
    }

    print(f"  Accuracy           : {metrics['accuracy']:.1f}% ({correct_cnt}/{total_g})")
    print(f"  Accuracy @ 5 Qs    : {metrics['acc_at_5']:.1f}%")
    print(f"  Accuracy @ 10 Qs   : {metrics['acc_at_10']:.1f}%")
    print(f"  Accuracy @ 15 Qs   : {metrics['acc_at_15']:.1f}%")
    print(f"  Accuracy @ 20 Qs   : {metrics['acc_at_20']:.1f}%")
    print(f"  Mean Questions     : {metrics['mean_q']:.1f}")
    print(f"  Median Questions   : {metrics['median_q']:.1f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CineMind Incremental Experiments")
    parser.add_argument("--samples", type=int, default=100, help="Number of benchmark samples (default: 100)")
    args = parser.parse_args()

    df = load_dataset()
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=min(args.samples, len(df)), replace=False)

    print(f"Initializing CineMind Shared Engine across {len(df):,} entities...")
    engine = CineMindGuesserEngine(df, num_concepts=200)

    print(f"Starting CineMind Incremental Experiments on {len(sample_indices)} samples...")

    m1 = run_experiment(engine, df, "Experiment 1 (Metadata Only)", ["metadata"], sample_indices)
    m2 = run_experiment(engine, df, "Experiment 2 (Metadata + Concept Clusters)", ["metadata", "concept"], sample_indices)
    m3 = run_experiment(engine, df, "Experiment 3 (Full Architecture: Metadata + Concept + Contrastive)", ["metadata", "concept", "contrastive"], sample_indices)

    print("\n" + "=" * 70)
    print("INCREMENTAL EXPERIMENTAL SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Exp 1 (Metadata)':<16} | {'Exp 2 (+Concept)':<14} | {'Exp 3 (Full)':<14}")
    print("-" * 70)
    print(f"{'Overall Accuracy':<25} | {m1['accuracy']:15.1f}% | {m2['accuracy']:13.1f}% | {m3['accuracy']:13.1f}%")
    print(f"{'Accuracy @ 5 Qs':<25} | {m1['acc_at_5']:15.1f}% | {m2['acc_at_5']:13.1f}% | {m3['acc_at_5']:13.1f}%")
    print(f"{'Accuracy @ 10 Qs':<25} | {m1['acc_at_10']:15.1f}% | {m2['acc_at_10']:13.1f}% | {m3['acc_at_10']:13.1f}%")
    print(f"{'Accuracy @ 15 Qs':<25} | {m1['acc_at_15']:15.1f}% | {m2['acc_at_15']:13.1f}% | {m3['acc_at_15']:13.1f}%")
    print(f"{'Accuracy @ 20 Qs':<25} | {m1['acc_at_20']:15.1f}% | {m2['acc_at_20']:13.1f}% | {m3['acc_at_20']:13.1f}%")
    print(f"{'Mean Questions':<25} | {m1['mean_q']:16.1f} | {m2['mean_q']:14.1f} | {m3['mean_q']:14.1f}")
    print(f"{'Median Questions':<25} | {m1['median_q']:16.1f} | {m2['median_q']:14.1f} | {m3['median_q']:14.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

