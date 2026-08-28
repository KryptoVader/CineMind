"""
Benchmark Runner CLI for CineMind Milestone 2A: Offline Guessing Simulator.

Runs reproducible simulated games using actual entity dataset and question catalog,
prints statistical summary, and exports results to JSON.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import json
import numpy as np

from cinemind.data.loader import DataLoader, DEFAULT_CANONICAL_PATH
from cinemind.questions.catalog import QuestionCatalog, DEFAULT_CATALOG_PARQUET_PATH
from cinemind.evaluation.simulator import GameSimulator, TargetSampler, SimulationResult
from cinemind.evaluation.metrics import EvaluationMetrics

DEFAULT_BENCHMARK_OUTPUT_PATH = Path("data/model/baseline_benchmark.json")


class BenchmarkRunner:
    """Orchestrates multi-game simulation benchmark runs."""

    def __init__(
        self,
        dataset_path: Path = DEFAULT_CANONICAL_PATH,
        catalog_path: Path = DEFAULT_CATALOG_PARQUET_PATH,
        sample_size: Optional[int] = 20000,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.catalog_path = Path(catalog_path)
        self.sample_size = sample_size

        self.entities = []
        self.catalog = None

    def initialize() -> None:
        pass

    def initialize(self) -> None:
        """Load dataset and question catalog."""
        loader = DataLoader(filepath=self.dataset_path)
        self.entities = loader.load_entities(sample_size=self.sample_size)

        if self.catalog_path.exists():
            self.catalog = QuestionCatalog.load(parquet_path=self.catalog_path)
        else:
            # Build catalog on the fly if not yet compiled
            self.catalog = QuestionCatalog.build_catalog(self.entities)

    def run_benchmark(
        self,
        num_games: int = 1000,
        seed: int = 42,
        max_questions: int = 25,
        sampling_mode: str = "uniform_entity",
    ) -> EvaluationMetrics:
        """Run benchmark experiment across num_games."""
        if not self.entities or not self.catalog:
            self.initialize()

        rng = np.random.RandomState(seed)
        simulator = GameSimulator(
            candidate_universe=self.entities,
            question_catalog=self.catalog,
            max_questions=max_questions,
        )

        results: List[SimulationResult] = []

        for i in range(num_games):
            # Sample hidden target strictly in evaluation runner
            if sampling_mode == "popularity_weighted":
                target = TargetSampler.sample_popularity_weighted(self.entities, rng=rng)
            else:
                target = TargetSampler.sample_uniform(self.entities, rng=rng)

            # Game seed derived deterministically per game
            game_seed = int(rng.randint(0, 1000000))
            res = simulator.run_game(target_entity=target, seed=game_seed)
            results.append(res)

        metrics = EvaluationMetrics(results=results, initial_universe_size=len(self.entities))
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CineMind Milestone 2A Offline Benchmark Runner")
    parser.add_argument("--games", type=int, default=1000, help="Number of simulated games to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-questions", type=int, default=25, help="Maximum question budget per game")
    parser.add_argument("--sampling-mode", type=str, default="uniform_entity", choices=["uniform_entity", "popularity_weighted"])
    parser.add_argument("--sample-size", type=int, default=20000, help="Entity population sample size")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_CANONICAL_PATH))
    parser.add_argument("--catalog-path", type=str, default=str(DEFAULT_CATALOG_PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_BENCHMARK_OUTPUT_PATH))

    args = parser.parse_args()

    print("============================================================", flush=True)
    print("Initializing CineMind Milestone 2A Offline Benchmark...", flush=True)
    print(f"Target Games:     {args.games}", flush=True)
    print(f"Seed:             {args.seed}", flush=True)
    print(f"Max Questions:    {args.max_questions}", flush=True)
    print(f"Sampling Mode:    {args.sampling_mode}", flush=True)
    print(f"Sample Size:      {args.sample_size}", flush=True)
    print("============================================================", flush=True)

    runner = BenchmarkRunner(
        dataset_path=Path(args.dataset_path),
        catalog_path=Path(args.catalog_path),
        sample_size=args.sample_size,
    )
    runner.initialize()
    print(f"Loaded {len(runner.entities)} entities and {len(runner.catalog)} questions in catalog.", flush=True)

    metrics = runner.run_benchmark(
        num_games=args.games,
        seed=args.seed,
        max_questions=args.max_questions,
        sampling_mode=args.sampling_mode,
    )

    report_text = metrics.summary_report()
    print("\n" + report_text + "\n")

    # Export machine readable benchmark results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"Benchmark results successfully exported to: {out_path}")


if __name__ == "__main__":
    main()
