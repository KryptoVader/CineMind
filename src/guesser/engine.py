"""
CineMind Engine Orchestrator (Phase 4).

Integrates KnowledgeBase, BeliefTracker, and 3 Hierarchical Question Generators.
Ranks proposed candidate questions using Two-Tier Information Gain IG(q):
- Phase 1: Global IG over 100,000 entities (early game broad search)
- Phase 2: Local IG over Top-50 candidates (late game candidate discrimination)
"""

import math
import logging
from typing import Optional, Any
import numpy as np
import pandas as pd

from guesser.knowledge import KnowledgeBase
from guesser.belief import BeliefTracker
from guesser.generators import CandidateQuestion, MetadataGenerator, ConceptClusterGenerator, ContrastiveGenerator

logger = logging.getLogger(__name__)


class CineMindGuesserEngine:
    """Main CineMind Generative Active Learning Guesser Engine."""

    def __init__(self, df: pd.DataFrame, num_concepts: int = 200, debug_ig_logging: bool = False):
        self.df = df
        self.kb = KnowledgeBase(df, num_concepts=num_concepts)
        self.tracker = BeliefTracker(self.kb)
        self.debug_ig_logging = debug_ig_logging

        # Instantiate Generators
        self.meta_gen = MetadataGenerator(self.kb, self.tracker)
        self.concept_gen = ConceptClusterGenerator(self.kb, self.tracker)
        self.contrast_gen = ContrastiveGenerator(self.kb, self.tracker)

        self.asked_q_ids: set[str] = set()

    def reset(self) -> None:
        """Reset game state for a new interactive or bot session."""
        self.tracker.reset()
        self.asked_q_ids.clear()
        self._tier_logged = False

    def get_posterior_probs(self) -> np.ndarray:
        """Get current posterior probability vector."""
        return self.tracker.get_posterior_probs()

    def get_top_candidates(self, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Get top-K candidate entities."""
        return self.tracker.get_top_candidates(k=k)

    def select_best_question(self, candidate_qs: list[CandidateQuestion]) -> Optional[CandidateQuestion]:
        """
        Two-Tier Vectorized Information Gain calculation across candidate questions.
        Requirement 4: Top-5 IG debug logging tagged by generator type.
        """
        if not candidate_qs:
            return None

        num_qs = len(candidate_qs)
        probs = self.tracker.get_posterior_probs()  # shape (N,)
        turns = len(self.tracker.history)
        max_p = float(np.max(probs))
        entropy = self.tracker.get_entropy()

        # Gate purely on posterior concentration
        is_local = (max_p >= 0.02) or (entropy <= 12.0)

        if is_local:
            if not getattr(self, "_tier_logged", False):
                logger.info(f"  [Engine] Adaptive Tier Switch: Local-IG activated at turn {turns+1} (max_p={max_p*100:.2f}%, entropy={entropy:.2f} bits)")
                print(f"  [Engine] Adaptive Tier Switch: Local-IG activated at turn {turns+1} (max_p={max_p*100:.2f}%, entropy={entropy:.2f} bits)")
                self._tier_logged = True

            # Top-50 Local IG
            top_k = min(50, len(probs))
            sub_idx = np.argpartition(probs, -top_k)[-top_k:]
            sub_probs = probs[sub_idx]
            sum_sub = np.sum(sub_probs)
            if sum_sub > 0:
                sub_probs = sub_probs / sum_sub
            else:
                sub_probs = np.ones(top_k, dtype=np.float64) / top_k
            eval_probs = sub_probs
            eval_indices = sub_idx
        else:
            self._tier_logged = False
            eval_probs = probs
            eval_indices = None

        num_ent_eval = len(eval_probs)

        # Build 2D matrix of shape (Q, N_eval)
        p_yes_matrix = np.zeros((num_qs, num_ent_eval), dtype=np.float64)
        for i, q in enumerate(candidate_qs):
            full_p_yes = q.get_p_yes_fn()
            if eval_indices is not None:
                p_yes_matrix[i] = full_p_yes[eval_indices]
            else:
                p_yes_matrix[i] = full_p_yes

        np.clip(p_yes_matrix, 1e-4, 1.0 - 1e-4, out=p_yes_matrix)
        p_no_matrix = 1.0 - p_yes_matrix

        # P(YES) and P(NO) for each question across evaluation probability distribution
        p_y = p_yes_matrix @ eval_probs  # shape (Q,)
        p_n = p_no_matrix @ eval_probs   # shape (Q,)

        # Shannon Entropy H(E) over evaluation set
        nz_e = eval_probs[eval_probs > 1e-12]
        h_e = float(-np.sum(nz_e * np.log2(nz_e)))

        ig_scores = np.zeros(num_qs, dtype=np.float64)

        for i in range(num_qs):
            if p_y[i] < 1e-5 or p_n[i] < 1e-5:
                ig_scores[i] = -1.0
                continue

            unnorm_y = eval_probs * p_yes_matrix[i]
            norm_y = unnorm_y / p_y[i]
            nz_y = norm_y[norm_y > 1e-12]
            h_y = -np.sum(nz_y * np.log2(nz_y))

            unnorm_n = eval_probs * p_no_matrix[i]
            norm_n = unnorm_n / p_n[i]
            nz_n = norm_n[norm_n > 1e-12]
            h_n = -np.sum(nz_n * np.log2(nz_n))

            ig_scores[i] = h_e - (p_y[i] * h_y + p_n[i] * h_n)

        # Requirement 4: Debug Instrumentation — Log Top-5 IG candidate questions tagged by generator type
        if self.debug_ig_logging and num_qs > 0:
            top5_idx = np.argpartition(ig_scores, -min(5, num_qs))[-min(5, num_qs):]
            top5_idx = top5_idx[np.argsort(ig_scores[top5_idx])[::-1]]
            print(f"\n--- Turn {turns+1} IG Evaluation (Pool: {num_qs} questions) ---")
            for rank, q_i in enumerate(top5_idx, 1):
                cq = candidate_qs[q_i]
                print(f"  {rank}. [{cq.generator_type.upper():<15s}] '{cq.text}' -> IG = {ig_scores[q_i]:.4f} bits")
            print("-----------------------------------------------------------\n")

        best_idx = int(np.argmax(ig_scores))
        if ig_scores[best_idx] > 0.0:
            return candidate_qs[best_idx]
        return candidate_qs[0]

    def get_best_question(self) -> Optional[CandidateQuestion]:
        """Collect candidate questions from generators using Metadata Phasing, Speculative Guessing, and Two-Tier IG."""
        candidate_qs: list[CandidateQuestion] = []

        # Phased metadata cap: Only generate metadata questions if fewer than 7 metadata questions asked
        meta_asked = sum(1 for h in self.tracker.history if h["q_id"].startswith("m_"))
        if meta_asked < 7:
            candidate_qs.extend(self.meta_gen.generate_questions(self.asked_q_ids))

        # Always include concept cluster questions
        candidate_qs.extend(self.concept_gen.generate_questions(self.asked_q_ids))

        # Include contrastive keyword questions as shortlist narrows or after Q5
        top_k = self.tracker.get_top_candidates(k=2)
        if top_k[0][1] >= 0.005 or len(self.tracker.history) >= 5:
            candidate_qs.extend(self.contrast_gen.generate_questions(self.asked_q_ids, top_candidates_k=5))

        # Requirement 6: Speculative Early Guessing
        top1_cand, top1_prob = top_k[0]
        if top1_prob >= 0.20:
            cand_id = top1_cand["cinemind_id"]
            cand_title = top1_cand["title"]
            spec_q_id = f"speculative_guess_{cand_id}"

            if spec_q_id not in self.asked_q_ids:
                def make_spec_p_yes(target_cid: str):
                    return lambda: np.where(self.df["cinemind_id"] == target_cid, 0.95, 0.05)

                candidate_qs.append(CandidateQuestion(
                    q_id=spec_q_id,
                    text=f"Is the movie, show, or anime you're thinking of '{cand_title}'?",
                    generator_type="speculative",
                    get_p_yes_fn=make_spec_p_yes(cand_id)
                ))

        return self.select_best_question(candidate_qs)

    def answer_question(self, question: CandidateQuestion, answer: str) -> None:
        """Update Bayesian belief tracker with empirical likelihoods."""
        self.asked_q_ids.add(question.q_id)
        p_yes_vec = question.get_p_yes_fn()
        self.tracker.update_belief(p_yes_vec, answer, question.q_id)

    def should_guess(self, max_questions: int = 30) -> bool:
        """Check if convergence controller criteria are met."""
        return self.tracker.check_convergence(max_questions=max_questions)

