"""
CineMind V2 — Game Engine
Main orchestrator for the CineMind V2 Akinator-style active learning engine.
"""

from typing import Optional, Any
import numpy as np

from cinemind_v2.knowledge.entity_store import EntityStore, Entity
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.questions.question import Question
from cinemind_v2.questions.generator import QuestionGenerator
from cinemind_v2.selection.question_ranker import QuestionRanker
from cinemind_v2.inference.posterior import BayesianPosterior
from cinemind_v2.game.state import GameState, GameTurn


class GameEngine:
    """
    CineMind V2 Game Engine.
    Coordinates knowledge representations, question generation, Information Gain selection,
    Bayesian belief updates, optional Decision Tree signals, and configurable game-ending stopping conditions.
    """

    def __init__(
        self,
        entity_store: EntityStore,
        feature_store: FeatureStore,
        target_posterior: float = 0.80,
        target_margin: float = 0.50,
        target_entropy: float = 1.0,
        max_questions: int = 25,
        prior_probs: Optional[np.ndarray] = None,
        tree_model: Optional[Any] = None,
        use_tree_prioritization: bool = False,
    ):
        self.entity_store = entity_store
        self.feature_store = feature_store
        self.target_posterior = target_posterior
        self.target_margin = target_margin
        self.target_entropy = target_entropy
        self.max_questions = max_questions
        self.tree_model = tree_model
        self.use_tree_prioritization = use_tree_prioritization

        # Initialize Question Generator & Library
        self.generator = QuestionGenerator(self.feature_store)
        self.question_library = self.generator.generate_all_questions()

        # Initialize Question Ranker
        self.ranker = QuestionRanker(self.feature_store)

        # Initialize Bayesian Posterior Tracker
        self.posterior = BayesianPosterior(
            num_entities=self.entity_store.num_entities,
            cinemind_ids=self.entity_store.get_cinemind_ids(),
            prior_probs=prior_probs,
        )

        # Initialize Game State
        self.state = GameState(self.posterior)

    def start_new_game(self) -> None:
        """Start or reset game session."""
        self.state.reset()

    def get_next_question(self) -> Optional[tuple[Question, float, float]]:
        """
        Rank unasked questions and return top candidate question tuple: (Question, ig_score, tree_signal).
        """
        if self.state.is_over:
            return None

        ranked = self.ranker.rank_questions(
            questions=self.question_library,
            posterior=self.posterior,
            asked_ids=self.state.asked_question_ids,
            top_n=1,
            tree_model=self.tree_model,
            use_tree_prioritization=self.use_tree_prioritization,
        )

        if not ranked:
            return None

        return ranked[0]

    def answer_question(
        self,
        question: Question,
        answer: str,
        ig_score: float = 0.0,
        tree_signal: float = 0.0,
    ) -> dict[str, Any]:
        """
        Submit user answer ("YES", "NO", or "UNKNOWN"), update Bayesian posterior,
        check stopping condition, and return status dictionary.
        """
        if self.state.is_over:
            raise RuntimeError("Game is already over. Start a new game before answering.")

        # Get feature mask and unknown mask for question
        mask = self.feature_store.get_feature_values(question.feature_id)
        unknown_mask = self.feature_store.get_unknown_values(question.feature_id)

        # Update Bayesian Posterior
        self.posterior.update(
            question=question,
            answer=answer,
            feature_mask=mask,
            unknown_mask=unknown_mask,
        )

        # Record Turn
        top1_id, top1_prob = self.posterior.get_top_k(1)[0]
        entropy_after = self.posterior.get_entropy()

        self.state.record_turn(
            question=question,
            answer=answer,
            ig_score=ig_score,
            entropy_after=entropy_after,
            top_candidate_id=top1_id,
            top_candidate_prob=top1_prob,
        )

        # Check Stopping Condition
        is_finished, reason = self.check_stopping_condition()

        if is_finished:
            self.state.is_over = True
            top_candidate = self.entity_store.get_entity(top1_id)
            self.state.final_guess = {
                "entity_id": top1_id,
                "entity": top_candidate,
                "probability": top1_prob,
                "reason": reason,
                "turns_taken": len(self.state.turns),
            }

        return {
            "turn": len(self.state.turns),
            "is_over": self.state.is_over,
            "top_candidate_id": top1_id,
            "top_candidate_prob": top1_prob,
            "entropy": entropy_after,
            "final_guess": self.state.final_guess,
        }

    def check_stopping_condition(self) -> tuple[bool, str]:
        """
        Evaluate if stopping criteria are met:
        1. Top posterior probability >= target_posterior
        2. Top 1/2 probability margin >= target_margin
        3. Entropy <= target_entropy
        4. Max questions reached
        5. No more questions available
        """
        turns_count = len(self.state.turns)
        top2 = self.posterior.get_top_k(2)
        top1_prob = top2[0][1]
        top2_prob = top2[1][1] if len(top2) > 1 else 0.0
        margin = top1_prob - top2_prob
        entropy = self.posterior.get_entropy()

        if top1_prob >= self.target_posterior:
            return True, f"Target posterior reached ({top1_prob:.1%} >= {self.target_posterior:.1%})"

        if margin >= self.target_margin:
            return True, f"Target margin reached ({margin:.1%} >= {self.target_margin:.1%})"

        if entropy <= self.target_entropy:
            return True, f"Target entropy reached ({entropy:.2f} <= {self.target_entropy:.2f} bits)"

        if turns_count >= self.max_questions:
            return True, f"Maximum questions limit reached ({turns_count}/{self.max_questions})"

        # Check if remaining questions available
        remaining = [q for q in self.question_library if q.id not in self.state.asked_question_ids]
        if not remaining:
            return True, "No remaining questions available"

        return False, ""
