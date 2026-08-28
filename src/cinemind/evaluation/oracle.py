"""
Simulation Oracle for CineMind.

Holds the hidden target entity and provides deterministic player answers (YES / NO / UNKNOWN).
Maintains strict target isolation — the target entity object is never exposed to the guessing procedure.
"""

from cinemind.data.entity import Entity
from cinemind.questions.schema import PlayerAnswer, Question


class SimulationOracle:
    """Evaluation Oracle wrapping the hidden target entity."""

    def __init__(self, target_entity: Entity) -> None:
        self._target_entity = target_entity

    @property
    def target_id(self) -> str:
        """Expose target entity ID for oracle/simulator bookkeeping."""
        return self._target_entity.cinemind_id

    @property
    def target_title(self) -> str:
        """Expose target entity title for final diagnostic reporting ONLY."""
        return self._target_entity.title

    def answer(self, question: Question) -> PlayerAnswer:
        """
        Evaluate question against hidden target entity.

        Returns:
            PlayerAnswer.YES, PlayerAnswer.NO, or PlayerAnswer.UNKNOWN.
            Preserves exact missingness semantics (missing metadata returns UNKNOWN).
        """
        return question.evaluate(self._target_entity)
