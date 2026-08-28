"""
Evaluation subpackage for CineMind Milestone 2A & 2B: Offline Guessing Simulator & Adaptive Selection.
"""

from cinemind.evaluation.oracle import SimulationOracle
from cinemind.evaluation.simulator import (
    EliminationEngine,
    GameSimulator,
    RandomQuestionPolicy,
    SimulationResult,
    TargetSampler,
)
from cinemind.evaluation.adaptive_policy import (
    BaseAdaptivePolicy,
    EntropySplitPolicy,
    ExpectedCandidateReductionPolicy,
    QuestionSplitScore,
    inspect_top_questions,
)
from cinemind.evaluation.metrics import EvaluationMetrics
from cinemind.evaluation.benchmark import BenchmarkRunner

__all__ = [
    "SimulationOracle",
    "TargetSampler",
    "RandomQuestionPolicy",
    "EliminationEngine",
    "GameSimulator",
    "SimulationResult",
    "BaseAdaptivePolicy",
    "EntropySplitPolicy",
    "ExpectedCandidateReductionPolicy",
    "QuestionSplitScore",
    "inspect_top_questions",
    "EvaluationMetrics",
    "BenchmarkRunner",
]
