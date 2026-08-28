"""
Adaptive Question Selection Policies for CineMind Milestone 2B.

Provides EntropySplitPolicy (maximizing Answer Outcome Entropy H(A_q))
and ExpectedCandidateReductionPolicy (maximizing Expected Candidate Reduction R(q)).

MATHEMATICAL DISTINCTION:
- Answer Outcome Entropy H(A_q) = -sum_{a} p_a log_2(p_a) measures uncertainty over the
  question's answer outcomes (YES, NO, UNKNOWN). It is NOT entity belief entropy H(E) over
  a probabilistic prior/posterior distribution P(e), as Milestone 2B operates strictly
  over candidate-set splitting in a deterministic elimination framework.
- Expected Remaining Candidates E[|C'|] = p_Y * |C_Y| + p_N * |C_N| + p_U * |C|
  (because an UNKNOWN oracle answer retains the entire candidate pool of size |C|).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from cinemind.data.entity import Entity
from cinemind.questions.catalog import QuestionCatalog
from cinemind.questions.schema import PlayerAnswer, Question


@dataclass
class QuestionSplitScore:
    """Diagnostic details for a question evaluated against a candidate set."""
    question_id: str
    text: str
    question_family: str
    score: float
    yes_count: int
    no_count: int
    unknown_count: int
    total_candidates: int
    yes_fraction: float
    no_fraction: float
    unknown_fraction: float
    expected_remaining_candidates: float
    expected_reduction_ratio: float
    answer_entropy: float
    coverage: float
    missingness: float
    reliability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "question_family": self.question_family,
            "score": float(self.score),
            "yes_count": self.yes_count,
            "no_count": self.no_count,
            "unknown_count": self.unknown_count,
            "total_candidates": self.total_candidates,
            "yes_fraction": float(self.yes_fraction),
            "no_fraction": float(self.no_fraction),
            "unknown_fraction": float(self.unknown_fraction),
            "expected_remaining_candidates": float(self.expected_remaining_candidates),
            "expected_reduction_ratio": float(self.expected_reduction_ratio),
            "answer_entropy": float(self.answer_entropy),
            "coverage": float(self.coverage),
            "missingness": float(self.missingness),
            "reliability": float(self.reliability),
        }


def compute_question_split_score(
    question: Question,
    candidates: List[Entity],
    evaluator: Optional[Any] = None,
) -> QuestionSplitScore:
    """
    Computes answer outcome statistics, Answer Entropy H(A_q), and Expected Candidate Reduction R(q)
    for a question evaluated over a candidate population.
    """
    total = len(candidates)
    if total == 0:
        return QuestionSplitScore(
            question_id=question.question_id,
            text=question.text,
            question_family=question.question_family,
            score=0.0,
            yes_count=0,
            no_count=0,
            unknown_count=0,
            total_candidates=0,
            yes_fraction=0.0,
            no_fraction=0.0,
            unknown_fraction=0.0,
            expected_remaining_candidates=0.0,
            expected_reduction_ratio=0.0,
            answer_entropy=0.0,
            coverage=question.coverage,
            missingness=question.missingness,
            reliability=question.reliability,
        )

    yes_c = 0
    no_c = 0
    unk_c = 0

    if evaluator and hasattr(evaluator, "evaluate"):
        eval_fn = evaluator.evaluate
        for e in candidates:
            ans = eval_fn(question, e)
            if ans == PlayerAnswer.YES:
                yes_c += 1
            elif ans == PlayerAnswer.NO:
                no_c += 1
            else:
                unk_c += 1
    else:
        for e in candidates:
            ans = question.evaluate(e)
            if ans == PlayerAnswer.YES:
                yes_c += 1
            elif ans == PlayerAnswer.NO:
                no_c += 1
            else:
                unk_c += 1

    p_y = yes_c / total
    p_n = no_c / total
    p_u = unk_c / total

    h_a = 0.0
    for p in (p_y, p_n, p_u):
        if p > 0:
            h_a -= p * np.log2(p)

    exp_remaining = (p_y * yes_c) + (p_n * no_c) + (p_u * total)
    exp_reduction = 1.0 - (exp_remaining / total)

    return QuestionSplitScore(
        question_id=question.question_id,
        text=question.text,
        question_family=question.question_family,
        score=0.0,
        yes_count=yes_c,
        no_count=no_c,
        unknown_count=unk_c,
        total_candidates=total,
        yes_fraction=float(p_y),
        no_fraction=float(p_n),
        unknown_fraction=float(p_u),
        expected_remaining_candidates=float(exp_remaining),
        expected_reduction_ratio=float(exp_reduction),
        answer_entropy=float(h_a),
        coverage=float(question.coverage),
        missingness=float(question.missingness),
        reliability=float(question.reliability),
    )


class VectorizedQuestionEvaluator:
    """Precomputed uint8 numpy matrix for ultra-fast vectorized question split scoring."""

    def __init__(self, catalog: QuestionCatalog, entities: List[Entity]) -> None:
        self.questions = catalog.list_questions()
        self.entities = entities

        self.q_id_to_idx: Dict[str, int] = {q.question_id: i for i, q in enumerate(self.questions)}
        self.e_id_to_idx: Dict[str, int] = {e.cinemind_id: j for j, e in enumerate(self.entities)}

        num_q = len(self.questions)
        num_e = len(self.entities)

        # 0 = UNKNOWN, 1 = YES, 2 = NO
        self.ans_matrix = np.zeros((num_q, num_e), dtype=np.int8)

        for i, q in enumerate(self.questions):
            for j, e in enumerate(self.entities):
                ans = q.evaluate(e)
                if ans == PlayerAnswer.YES:
                    self.ans_matrix[i, j] = 1
                elif ans == PlayerAnswer.NO:
                    self.ans_matrix[i, j] = 2

    def evaluate(self, question: Question, entity: Entity) -> PlayerAnswer:
        q_idx = self.q_id_to_idx.get(question.question_id)
        e_idx = self.e_id_to_idx.get(entity.cinemind_id)
        if q_idx is not None and e_idx is not None:
            val = self.ans_matrix[q_idx, e_idx]
            if val == 1:
                return PlayerAnswer.YES
            elif val == 2:
                return PlayerAnswer.NO
            else:
                return PlayerAnswer.UNKNOWN
        return question.evaluate(entity)

    def compute_all_splits_vectorized(
        self,
        cand_indices: np.ndarray,
        asked_question_ids: Set[str],
    ) -> List[QuestionSplitScore]:
        total = len(cand_indices)
        if total == 0:
            return []

        sub_matrix = self.ans_matrix[:, cand_indices]  # shape (num_q, total)

        yes_counts = (sub_matrix == 1).sum(axis=1)
        no_counts = (sub_matrix == 2).sum(axis=1)
        unk_counts = (sub_matrix == 0).sum(axis=1)

        p_y = yes_counts / total
        p_n = no_counts / total
        p_u = unk_counts / total

        with np.errstate(divide="ignore", invalid="ignore"):
            log_py = np.where(p_y > 0, np.log2(p_y), 0.0)
            log_pn = np.where(p_n > 0, np.log2(p_n), 0.0)
            log_pu = np.where(p_u > 0, np.log2(p_u), 0.0)

        h_arr = -(p_y * log_py + p_n * log_pn + p_u * log_pu)

        exp_remaining_arr = (p_y * yes_counts) + (p_n * no_counts) + (p_u * total)
        exp_reduction_arr = 1.0 - (exp_remaining_arr / total)

        scores: List[QuestionSplitScore] = []
        for i, q in enumerate(self.questions):
            if q.question_id in asked_question_ids:
                continue
            scores.append(
                QuestionSplitScore(
                    question_id=q.question_id,
                    text=q.text,
                    question_family=q.question_family,
                    score=0.0,
                    yes_count=int(yes_counts[i]),
                    no_count=int(no_counts[i]),
                    unknown_count=int(unk_counts[i]),
                    total_candidates=total,
                    yes_fraction=float(p_y[i]),
                    no_fraction=float(p_n[i]),
                    unknown_fraction=float(p_u[i]),
                    expected_remaining_candidates=float(exp_remaining_arr[i]),
                    expected_reduction_ratio=float(exp_reduction_arr[i]),
                    answer_entropy=float(h_arr[i]),
                    coverage=float(q.coverage),
                    missingness=float(q.missingness),
                    reliability=float(q.reliability),
                )
            )

        return scores


class BaseAdaptivePolicy(ABC):
    """Abstract base class for adaptive question selection policies."""

    def __init__(self, catalog: QuestionCatalog) -> None:
        self.catalog = catalog
        self.questions = catalog.list_questions()

    @abstractmethod
    def score_question(self, split_score: QuestionSplitScore) -> float:
        """Assign policy score to evaluated question split."""
        pass

    def evaluate_candidates(
        self,
        candidates: List[Entity],
        asked_question_ids: Set[str],
        evaluator: Optional[Any] = None,
        cand_indices: Optional[np.ndarray] = None,
    ) -> List[QuestionSplitScore]:
        """Evaluate and score all available unused questions over current candidate set."""
        if evaluator and hasattr(evaluator, "compute_all_splits_vectorized") and cand_indices is not None:
            scores = evaluator.compute_all_splits_vectorized(cand_indices, asked_question_ids)
            for s in scores:
                s.score = self.score_question(s)
        else:
            scores = []
            for q in self.questions:
                if q.question_id in asked_question_ids:
                    continue
                split = compute_question_split_score(q, candidates, evaluator=evaluator)
                split.score = self.score_question(split)
                scores.append(split)

        # Deterministic Tie-Breaking Sort:
        # 1. Higher score
        # 2. Higher coverage
        # 3. Lower missingness
        # 4. Higher reliability
        # 5. Deterministic question_id ascending
        scores.sort(
            key=lambda s: (
                s.score,
                s.coverage,
                -s.missingness,
                s.reliability,
                -ord(s.question_id[0]) if s.question_id else 0,
            ),
            reverse=True,
        )

        return scores

    def select_next_question(
        self,
        candidates: List[Entity],
        asked_question_ids: Set[str],
        evaluator: Optional[Any] = None,
        cand_indices: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Question], Optional[QuestionSplitScore]]:
        """Select best next question according to policy score and tie-breaking rules."""
        scores = self.evaluate_candidates(candidates, asked_question_ids, evaluator=evaluator, cand_indices=cand_indices)
        if not scores:
            return None, None

        top_score = scores[0]
        question = self.catalog.get_by_id(top_score.question_id)
        return question, top_score


class EntropySplitPolicy(BaseAdaptivePolicy):
    """
    Adaptive Policy maximizing Answer Outcome Entropy H(A_q).
    """

    def score_question(self, split_score: QuestionSplitScore) -> float:
        return split_score.answer_entropy


class ExpectedCandidateReductionPolicy(BaseAdaptivePolicy):
    """
    Adaptive Policy maximizing Expected Candidate Reduction Ratio R(q) = 1 - E[|C'|] / |C|.
    """

    def score_question(self, split_score: QuestionSplitScore) -> float:
        return split_score.expected_reduction_ratio


def inspect_top_questions(
    candidates: List[Entity],
    catalog: QuestionCatalog,
    policy: Optional[BaseAdaptivePolicy] = None,
    top_n: int = 20,
) -> List[QuestionSplitScore]:
    """
    Utility CLI function to inspect and print top N candidate questions for a candidate subset.
    """
    pol = policy or EntropySplitPolicy(catalog=catalog)
    evaluator = VectorizedQuestionEvaluator(catalog=catalog, entities=candidates)
    cand_indices = np.arange(len(candidates), dtype=np.int32)
    scores = pol.evaluate_candidates(candidates, asked_question_ids=set(), evaluator=evaluator, cand_indices=cand_indices)

    print("==========================================================================================", flush=True)
    print(f"Top {min(top_n, len(scores))} Questions Scored by {pol.__class__.__name__}", flush=True)
    print("==========================================================================================", flush=True)
    print(
        f"{'Rank':<5} {'Question Text':<35} {'Family':<10} {'Score':<8} {'YES%':<6} {'NO%':<6} {'UNK%':<6} {'ExpRem':<8} {'ExpRed%':<8}",
        flush=True,
    )
    print("-" * 90, flush=True)

    for rank, s in enumerate(scores[:top_n], 1):
        print(
            f"{rank:<5} {s.text[:34]:<35} {s.question_family:<10} {s.score:<8.4f} {s.yes_fraction*100:<6.1f} {s.no_fraction*100:<6.1f} {s.unknown_fraction*100:<6.1f} {s.expected_remaining_candidates:<8.1f} {s.expected_reduction_ratio*100:<8.2f}%",
            flush=True,
        )
    print("==========================================================================================", flush=True)

    return scores[:top_n]
