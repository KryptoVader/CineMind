"""
CineMind Simulator & Benchmark Suite (Phase 5).

Usage:
  python -m guesser.simulator              # Play interactive live game
  python -m guesser.simulator --test-auto  # Run automated bot benchmark
  python -m guesser.simulator --samples 30 # Benchmark sample count
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from pipeline.config import DATA_DIR, CANONICAL_DIR
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


def run_automated_benchmark(samples: int = 20) -> None:
    """Run automated bot benchmark measuring Accuracy@K and Mean/Median questions."""
    df = load_dataset()
    print(f"Loaded dataset: {len(df):,} records.")
    print("Initializing CineMind Generative Guesser Engine (Knowledge + Belief + Hierarchical Generators)...")

    engine = CineMindGuesserEngine(df, num_concepts=200)

    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=min(samples, len(df)), replace=False)

    print(f"Running automated benchmark on {len(sample_indices)} target entities...\n")

    correct_cnt = 0
    q_counts = []

    # Accuracy@K metrics
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
            gen_q = engine.get_best_question()
            if gen_q is None:
                break

            p_yes_vec = gen_q.get_p_yes_fn()
            # Bot answers based on empirical P(YES) > 0.40
            is_yes = float(p_yes_vec[target_idx]) >= 0.30
            ans_str = "yes" if is_yes else "no"

            engine.answer_question(gen_q, ans_str)
            q_count += 1

            # Check top candidate at question milestones
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

        # Requirement 7: Wire feedback logging into benchmark
        from guesser.feedback import log_game_feedback
        log_game_feedback(engine.tracker.history, top_cand["cinemind_id"], is_correct, target["cinemind_id"])

        q_counts.append(q_count)
        status = "CORRECT" if is_correct else "WRONG"
        print(f"Game {game_num:2d}/{len(sample_indices)}: [{status}] Target: '{target['title']}' | Guessed: '{top_cand['title']}' in {q_count} Qs (Prob: {top_p*100:.1f}%)")

    total_g = len(sample_indices)
    print("\n" + "=" * 64)
    print("CINEMIND GENERATIVE ENGINE BENCHMARK RESULTS")
    print("=" * 64)
    print(f"  Total Games Played   : {total_g}")
    print(f"  Overall Accuracy     : {correct_cnt/total_g*100:.1f}% ({correct_cnt}/{total_g})")
    print(f"  Accuracy @ 5 Qs      : {acc_at_5/total_g*100:.1f}% ({acc_at_5}/{total_g})")
    print(f"  Accuracy @ 10 Qs     : {acc_at_10/total_g*100:.1f}% ({acc_at_10}/{total_g})")
    print(f"  Accuracy @ 15 Qs     : {acc_at_15/total_g*100:.1f}% ({acc_at_15}/{total_g})")
    print(f"  Accuracy @ 20 Qs     : {acc_at_20/total_g*100:.1f}% ({acc_at_20}/{total_g})")
    print(f"  Mean Questions/Game  : {np.mean(q_counts):.1f} questions")
    print(f"  Median Questions     : {np.median(q_counts):.1f} questions")
    print(f"  Min / Max Questions  : {min(q_counts)} / {max(q_counts)} questions")
    print("=" * 64)


def run_interactive_game() -> None:
    """Run interactive terminal guesser game."""
    df = load_dataset()
    print("Initializing CineMind Generative Engine...")
    engine = CineMindGuesserEngine(df, num_concepts=200)

    print("\n" + "=" * 60)
    print("WELCOME TO CINEMIND — FULLY GENERATIVE AKINATOR GUESSER")
    print("=" * 60)
    print("Think of any movie, TV series, or anime in your mind!")
    print("Answer: yes (y) / no (n) / dont know (k) / quit (q)")
    print("=" * 60 + "\n")

    q_count = 0
    while not engine.should_guess(max_questions=35):
        gen_q = engine.get_best_question()
        if gen_q is None:
            break

        q_count += 1
        top_cand, top_p = engine.get_top_candidates(k=1)[0]

        print(f"Question #{q_count} [{gen_q.generator_type.upper()}] Top Guess: '{top_cand['title']}' ({top_p*100:.1f}%)")
        print(f"  -> {gen_q.text}")

        while True:
            user_input = input("Your answer (y/n/k/q): ").strip().lower()
            if user_input in ["q", "quit", "exit"]:
                print("Game exited.")
                return
            if user_input in ["y", "yes"]:
                ans = "yes"
                break
            elif user_input in ["n", "no"]:
                ans = "no"
                break
            elif user_input in ["k", "dont_know", "dk", "idk"]:
                ans = "dont_know"
                break
            else:
                print("  [!] Invalid. Type y / n / k / q")

        engine.answer_question(gen_q, ans)
        print()

    top5 = engine.get_top_candidates(k=5)
    top_cand, top_p = top5[0]

    print("\n" + "=" * 60)
    print("CINEMIND GUESS RESULT")
    print("=" * 60)
    print(f"\n  Are you thinking of:")
    print(f"  ★ {top_cand['title'].upper()} ({top_cand.get('release_year', 'N/A')})")
    print(f"    Media: {top_cand.get('media_type')} | Language: {top_cand.get('original_language')}")
    print(f"    Confidence: {top_p*100:.1f}% (in {q_count} questions)\n")

    if len(top5) > 1:
        print("  Other candidates:")
        for i, (cand, p) in enumerate(top5[1:], 2):
            print(f"    {i}. {cand['title']} ({cand.get('release_year', '?')}) — {p*100:.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CineMind Generative Guesser Simulator")
    parser.add_argument("--test-auto", action="store_true", help="Run automated bot benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of benchmark games")
    args = parser.parse_args()

    if args.test_auto:
        run_automated_benchmark(samples=args.samples)
    else:
        run_interactive_game()


if __name__ == "__main__":
    main()
