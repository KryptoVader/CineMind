"""
CineMind V2 — Question Validation
Validates quality criteria and filters useless or uninformative questions.
"""

import math
from cinemind_v2.questions.question import Question


def validate_question(
    q: Question,
    min_coverage: float = 0.0005,
    max_yes_rate: float = 0.98,
    min_yes_rate: float = 0.001,
    max_unknown_rate: float = 0.50,
) -> bool:
    """
    Validates whether a question meets information-quality thresholds.
    Filters out:
    - Nearly always yes (yes_rate > max_yes_rate)
    - Nearly always no (yes_rate < min_yes_rate)
    - Insufficient coverage (coverage < min_coverage)
    - Excessive unknown rate (unknown_rate > max_unknown_rate)
    """
    if not q.text or not q.id or not q.feature_id:
        return False

    if q.yes_rate > max_yes_rate or q.yes_rate < min_yes_rate:
        return False

    if q.coverage < min_coverage:
        return False

    if q.unknown_rate > max_unknown_rate:
        return False

    return True
