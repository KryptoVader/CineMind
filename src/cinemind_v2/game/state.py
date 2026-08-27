"""
CineMind V2 — Game State
Data structures tracking game turn history and current belief state.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from cinemind_v2.questions.question import Question
from cinemind_v2.inference.posterior import BayesianPosterior


@dataclass
class GameTurn:
    turn_number: int
    question: Question
    answer: str
    ig_score: float
    entropy_after: float
    top_candidate_id: str
    top_candidate_prob: float


class GameState:
    """
    Tracks state of an active CineMind game session.
    """

    def __init__(self, posterior: BayesianPosterior):
        self.posterior = posterior
        self.turns: list[GameTurn] = []
        self.asked_question_ids: set[str] = set()
        self.is_over: bool = False
        self.final_guess: Optional[dict[str, Any]] = None

    def reset(self) -> None:
        self.posterior.reset()
        self.turns.clear()
        self.asked_question_ids.clear()
        self.is_over = False
        self.final_guess = None

    def record_turn(
        self,
        question: Question,
        answer: str,
        ig_score: float,
        entropy_after: float,
        top_candidate_id: str,
        top_candidate_prob: float,
    ) -> None:
        turn = GameTurn(
            turn_number=len(self.turns) + 1,
            question=question,
            answer=answer,
            ig_score=ig_score,
            entropy_after=entropy_after,
            top_candidate_id=top_candidate_id,
            top_candidate_prob=top_candidate_prob,
        )
        self.turns.append(turn)
        self.asked_question_ids.add(question.id)
        if question.feature_id:
            self.asked_question_ids.add(question.feature_id)
