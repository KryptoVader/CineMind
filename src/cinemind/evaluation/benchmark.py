"""
Benchmark Runner CLI for CineMind Milestone 2A & 2B.

Runs multi-policy, multi-seed offline simulations comparing Random, EntropySplit,
and ExpectedCandidateReduction policies.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

from cinemind.data.loader import DataLoader, DEFAULT_CANONICAL_PATH
from cinemind.questions.catalog import QuestionCatalog, DEFAULT_CATALOG_PARQUET_PATH
from cinemind.evaluation.simulator import GameSimulator, TargetSampler, SimulationResult
from cinemind.evaluation.adaptive_policy import EntropySplitPolicy, ExpectedCandidateReductionPolicy, inspect_top_questions
from cinemind.evaluation.metrics import EvaluationMetrics, MultiSeedComparison

DEFAULT_ADAPTIVE_BENCHMARK_OUTPUT_PATH = Path("data/model/adaptive_benchmark.json")


class BenchmarkRunner:
    """Orchestrates multi-game, multi-policy, multi-seed simulation benchmarks."""

    def __init__(
        self,
        dataset_path: Path = DEFAULT_CANONICAL_PATH,
        catalog_path: Path = DEFAULT_CATALOG_PARQUET_PATH,
        sample_size: Optional[int] = 10000,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.catalog_path = Path(catalog_path)
        self.sample_size = sample_size

        self.entities = []
        self.catalog = None

    def initialize(self) -> None:
        """Load dataset and question catalog."""
        loader = DataLoader(filepath=self.dataset_path)
        self.entities = loader.load_entities(sample_size=self.sample_size)

        if self.catalog_path.exists():
            self.catalog = QuestionCatalog.load(parquet_path=self.catalog_path)
        else:
            self.catalog = QuestionCatalog.build_catalog(self.entities)

    def run_policy_benchmark(
        self,
        policy_type: str = "random",
        num_games: int = 1000,
        seed: int = 42,
        max_questions: int = 25,
        sampling_mode: str = "uniform_entity",
    ) -> EvaluationMetrics:
        """Run single policy benchmark for a specific seed."""
        if not self.entities or not self.catalog:
            self.initialize()

        rng = np.random.RandomState(seed)
        simulator = GameSimulator(
            candidate_universe=self.entities,
            question_catalog=self.catalog,
            max_questions=max_questions,
            policy_type=policy_type,
        )

        results: List[SimulationResult] = []

        for _ in range(num_games):
            if sampling_mode == "popularity_weighted":
                target = TargetSampler.sample_popularity_weighted(self.entities, rng=rng)
            else:
                target = TargetSampler.sample_uniform(self.entities, rng=rng)

            game_seed = int(rng.randint(0, 1000000))
            res = simulator.run_game(target_entity=target, seed=game_seed)
            results.append(res)

        return EvaluationMetrics(results=results, initial_universe_size=len(self.entities))


def main() -> None:
    parser = argparse.ArgumentParser(description="CineMind Milestone 2B Adaptive Benchmark Runner")
    parser.add_argument("--games", type=int, default=1000, help="Number of simulated games per seed")
    parser.add_argument("--seeds", type=str, default="42,123,999", help="Comma-separated random seeds")
    parser.add_argument("--policies", type=str, default="random,entropy,expected_reduction", help="Comma-separated policies")
    parser.add_argument("--max-questions", type=int, default=25, help="Maximum question budget per game")
    parser.add_argument("--sampling-mode", type=str, default="uniform_entity", choices=["uniform_entity", "popularity_weighted"])
    parser.add_argument("--sample-size", type=int, default=10000, help="Entity population sample size")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_CANONICAL_PATH))
    parser.add_argument("--catalog-path", type=str, default=str(DEFAULT_CATALOG_PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_ADAPTIVE_BENCHMARK_OUTPUT_PATH))
    parser.add_argument("--inspect", action="store_true", help="Inspect top 20 questions for initial population and exit")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        dataset_path=Path(args.dataset_path),
        catalog_path=Path(args.catalog_path),
        sample_size=args.sample_size,
    )
    runner.initialize()

    if args.inspect:
        inspect_top_questions(candidates=runner.entities, catalog=runner.catalog, top_n=20)
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    print("============================================================", flush=True)
    print("Initializing CineMind Milestone 2B Multi-Policy Benchmark...", flush=True)
    print(f"Target Games per Seed: {args.games}", flush=True)
    print(f"Seeds:                 {seeds}", flush=True)
    print(f"Policies:              {policies}", flush=True)
    print(f"Max Questions:         {args.max_questions}", flush=True)
    print(f"Sampling Mode:         {args.sampling_mode}", flush=True)
    print(f"Universe Sample Size:  {len(runner.entities)} entities", flush=True)
    print("============================================================", flush=True)

    multi_policy_results: Dict[str, List[EvaluationMetrics]] = {}

    for pol in policies:
        print(f"\n---> Benchmarking Policy: '{pol}' across seeds {seeds}...", flush=True)
        pol_metrics: List[EvaluationMetrics] = []
        for s in seeds:
            print(f"     Running seed={s} ({args.games} games)...", flush=True)
            m = runner.run_policy_benchmark(
                policy_type=pol,
                num_games=args.games,
                seed=s,
                max_questions=args.max_questions,
                sampling_mode=args.sampling_mode,
            )
            pol_metrics.append(m)
        multi_policy_results[pol] = pol_metrics

    # Print summary reports per policy
    for pol, m_list in multi_policy_results.items():
        print("\n" + m_list[0].summary_report(), flush=True)

    # Print side-by-side comparison table
    comparison = MultiSeedComparison(policy_metrics_map=multi_policy_results)
    comp_text = comparison.summary_table()
    print("\n" + comp_text + "\n", flush=True)

    # Export machine readable output
    out_dict: Dict[str, Any] = {
        "games_per_seed": args.games,
        "seeds": seeds,
        "universe_size": len(runner.entities),
        "policies": {},
    }
    for pol, m_list in multi_policy_results.items():
        out_dict["policies"][pol] = [m.to_dict() for m in m_list]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"Adaptive benchmark results successfully exported to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
