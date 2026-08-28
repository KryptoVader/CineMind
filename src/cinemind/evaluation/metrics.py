"""
Evaluation Metrics Analyzer for CineMind Milestone 2A & 2B.

Computes accuracy, failure rate, zero-candidate rate, candidate reduction curves,
question-family breakdown, difficult target diagnostics, and multi-seed/multi-policy side-by-side comparison tables.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np

from cinemind.evaluation.simulator import SimulationResult


class EvaluationMetrics:
    """Computes and aggregates statistical metrics over simulated game results for a single run."""

    def __init__(self, results: List[SimulationResult], initial_universe_size: int = 0) -> None:
        self.results = results
        self.initial_universe_size = initial_universe_size
        self._analyze()

    def _analyze(self) -> None:
        self.total_games = len(self.results)
        if self.total_games == 0:
            self.accuracy = 0.0
            self.failure_rate = 0.0
            self.zero_candidate_rate = 0.0
            self.questions_stats = {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
            self.candidate_reduction_curve: Dict[int, Dict[str, float]] = {}
            self.family_stats: Dict[str, Dict[str, Any]] = {}
            self.difficult_targets: List[Dict[str, Any]] = []
            self.zero_candidate_diagnostics: List[Dict[str, Any]] = []
            self.final_candidate_stats = {"mean": 0.0, "median": 0.0}
            return

        # 1. Correctness, Failure Rate, and Zero-Candidate Rate
        self.correct_guesses = sum(1 for r in self.results if r.correct)
        self.accuracy = self.correct_guesses / self.total_games

        self.unsuccessful_games = sum(1 for r in self.results if not r.correct)
        self.failure_rate = self.unsuccessful_games / self.total_games

        self.zero_candidate_games = sum(
            1 for r in self.results if r.terminated_reason == "empty_candidate_set" or r.remaining_candidates == 0
        )
        self.zero_candidate_rate = self.zero_candidate_games / self.total_games

        # 2. Questions asked stats
        q_counts = [r.questions_asked for r in self.results]
        self.questions_stats = {
            "mean": float(np.mean(q_counts)),
            "median": float(np.median(q_counts)),
            "min": int(np.min(q_counts)),
            "max": int(np.max(q_counts)),
        }

        # Final candidate pool size stats
        final_cand_counts = [r.remaining_candidates for r in self.results]
        self.final_candidate_stats = {
            "mean": float(np.mean(final_cand_counts)),
            "median": float(np.median(final_cand_counts)),
        }

        # 3. Candidate Reduction Curve across steps (step 0 to max_step)
        step_candidates: Dict[int, List[int]] = defaultdict(list)
        for r in self.results:
            step_candidates[0].append(self.initial_universe_size or (r.history[0]["remaining_candidates"] if r.history else 0))
            for h in r.history:
                step_candidates[h["step"]].append(h["remaining_candidates"])

        self.candidate_reduction_curve = {}
        for step in sorted(step_candidates.keys()):
            vals = step_candidates[step]
            self.candidate_reduction_curve[step] = {
                "mean_candidates": float(np.mean(vals)),
                "median_candidates": float(np.median(vals)),
                "sample_count": len(vals),
            }

        # 4. Question Family Breakdown
        family_counts: Dict[str, int] = defaultdict(int)
        family_reductions: Dict[str, List[float]] = defaultdict(list)

        for r in self.results:
            for i, h in enumerate(r.history):
                fam = h["question_family"]
                family_counts[fam] += 1
                prev_cnt = self.initial_universe_size if i == 0 else r.history[i - 1]["remaining_candidates"]
                curr_cnt = h["remaining_candidates"]
                if prev_cnt > 0:
                    reduction_ratio = (prev_cnt - curr_cnt) / prev_cnt
                    family_reductions[fam].append(reduction_ratio)

        self.family_stats = {}
        total_q_used = sum(family_counts.values()) or 1
        for fam, cnt in family_counts.items():
            reds = family_reductions[fam]
            self.family_stats[fam] = {
                "questions_used": cnt,
                "pct_of_total": float(cnt / total_q_used),
                "mean_candidate_reduction_ratio": float(np.mean(reds)) if reds else 0.0,
            }

        # 5. Difficult Targets
        self.difficult_targets = []
        for r in self.results:
            if not r.correct or r.questions_asked >= 20:
                self.difficult_targets.append({
                    "target_entity_id": r.target_entity_id,
                    "target_title": r.target_title,
                    "guessed_entity_id": r.guessed_entity_id,
                    "guessed_title": r.guessed_title,
                    "correct": r.correct,
                    "questions_asked": r.questions_asked,
                    "remaining_candidates": r.remaining_candidates,
                    "terminated_reason": r.terminated_reason,
                })

        # 6. Zero Candidate Diagnostics
        self.zero_candidate_diagnostics = []
        for r in self.results:
            if r.zero_candidate_info:
                diag = dict(r.zero_candidate_info)
                diag["target_entity_id"] = r.target_entity_id
                diag["target_title"] = r.target_title
                self.zero_candidate_diagnostics.append(diag)

    def summary_report(self) -> str:
        """Format human-readable text summary of benchmark results."""
        pol_name = self.results[0].policy_name if self.results else "Unknown"
        lines = [
            "============================================================",
            f"CineMind Benchmark Results — Policy: {pol_name}",
            "============================================================",
            f"Total Games Simulated:      {self.total_games}",
            f"Initial Candidate Universe: {self.initial_universe_size}",
            f"Correct Guesses:            {self.correct_guesses} / {self.total_games}",
            f"Accuracy:                   {self.accuracy * 100:.2f}%",
            f"Failure Rate:               {self.failure_rate * 100:.2f}% (unsuccessful games)",
            f"Zero-Candidate Rate:        {self.zero_candidate_rate * 100:.2f}% (pool collapsed to 0)",
            "------------------------------------------------------------",
            "Questions Per Game:",
            f"  Mean:   {self.questions_stats['mean']:.2f}",
            f"  Median: {self.questions_stats['median']:.1f}",
            f"  Min:    {self.questions_stats['min']}",
            f"  Max:    {self.questions_stats['max']}",
            "------------------------------------------------------------",
            "Final Remaining Candidates Per Game:",
            f"  Mean:   {self.final_candidate_stats['mean']:.2f}",
            f"  Median: {self.final_candidate_stats['median']:.1f}",
            "------------------------------------------------------------",
            "Candidate Reduction Curve (Mean Candidates Remaining):",
        ]

        for step in sorted(self.candidate_reduction_curve.keys())[:11]:
            stats = self.candidate_reduction_curve[step]
            lines.append(f"  Turn {step:2d}: {stats['mean_candidates']:10.1f} candidates (median: {stats['median_candidates']:.0f})")

        lines.append("------------------------------------------------------------")
        lines.append("Question Family Breakdown:")
        lines.append(f"  {'Family':<15} {'Used':<10} {'Usage %':<10} {'Avg Reduction Ratio':<20}")
        for fam, f_stat in sorted(self.family_stats.items(), key=lambda x: x[1]["questions_used"], reverse=True):
            lines.append(
                f"  {fam:<15} {f_stat['questions_used']:<10} {f_stat['pct_of_total']*100:6.1f}%    {f_stat['mean_candidate_reduction_ratio']*100:18.2f}%"
            )

        lines.append("------------------------------------------------------------")
        lines.append(f"Difficult Targets Logged:   {len(self.difficult_targets)}")
        lines.append(f"Zero-Candidate Traces Logged: {len(self.zero_candidate_diagnostics)}")
        lines.append("============================================================")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        pol_name = self.results[0].policy_name if self.results else "Unknown"
        return {
            "policy_name": pol_name,
            "total_games": self.total_games,
            "initial_universe_size": self.initial_universe_size,
            "correct_guesses": self.correct_guesses,
            "accuracy": float(self.accuracy),
            "failure_rate": float(self.failure_rate),
            "zero_candidate_rate": float(self.zero_candidate_rate),
            "questions_stats": self.questions_stats,
            "final_candidate_stats": self.final_candidate_stats,
            "candidate_reduction_curve": self.candidate_reduction_curve,
            "family_stats": self.family_stats,
            "difficult_targets_sample": self.difficult_targets[:20],
            "zero_candidate_diagnostics_sample": self.zero_candidate_diagnostics[:20],
        }


class MultiSeedComparison:
    """Aggregates EvaluationMetrics results across multiple seeds and policies for side-by-side comparison."""

    def __init__(self, policy_metrics_map: Dict[str, List[EvaluationMetrics]]) -> None:
        self.policy_metrics = policy_metrics_map

    def summary_table(self) -> str:
        """Format side-by-side multi-policy multi-seed summary table."""
        lines = [
            "==================================================================================================",
            "CineMind Milestone 2B — Multi-Policy Multi-Seed Comparison Report",
            "==================================================================================================",
            f"{'Policy':<25} {'Accuracy (Mean ± Std)':<24} {'Mean Q':<12} {'Median Q':<12} {'Final Candidates':<18} {'Failure Rate':<14}",
            "--------------------------------------------------------------------------------------------------",
        ]

        for pol_name, metrics_list in self.policy_metrics.items():
            accs = [m.accuracy * 100 for m in metrics_list]
            mean_qs = [m.questions_stats["mean"] for m in metrics_list]
            med_qs = [m.questions_stats["median"] for m in metrics_list]
            final_cands = [m.final_candidate_stats["mean"] for m in metrics_list]
            failures = [m.failure_rate * 100 for m in metrics_list]

            acc_str = f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}%"
            mq_str = f"{np.mean(mean_qs):.2f}"
            medq_str = f"{np.mean(med_qs):.1f}"
            fcand_str = f"{np.mean(final_cands):.1f} ± {np.std(final_cands):.1f}"
            fail_str = f"{np.mean(failures):.2f}%"

            lines.append(f"{pol_name:<25} {acc_str:<24} {mq_str:<12} {medq_str:<12} {fcand_str:<18} {fail_str:<14}")

        lines.append("==================================================================================================")
        return "\n".join(lines)
