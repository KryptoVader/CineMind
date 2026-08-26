"""
CineMind Feedback & Recalibration System (Phase 7).

Logs game history to disk and applies count-weighted Bayesian smoothing:
  P(YES | e_i, q_j) = (prior * k + observed_yes) / (k + total_observed)
"""

from pathlib import Path
import json
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR

FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "game_feedback_logs.parquet"

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Persistent Feedback Store and Likelihood Recalibration Engine."""

    _instance: Optional["FeedbackStore"] = None

    def __init__(self):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        # Map: (cinemind_id, q_id) -> {"yes": int, "total": int}
        self.stats: dict[tuple[str, str], dict[str, int]] = {}
        self.load_logs()

    @classmethod
    def get_instance(cls) -> "FeedbackStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_logs(self) -> None:
        """Load persistent game logs from disk."""
        if not FEEDBACK_FILE.exists():
            return

        try:
            df = pd.read_parquet(FEEDBACK_FILE)
            for _, row in df.iterrows():
                cid = row["true_cinemind_id"]
                q_id = row["q_id"]
                ans = str(row["answer"]).lower().strip()

                if not cid or not q_id:
                    continue

                key = (cid, q_id)
                if key not in self.stats:
                    self.stats[key] = {"yes": 0, "total": 0}

                if ans in ["yes", "y"]:
                    self.stats[key]["yes"] += 1
                    self.stats[key]["total"] += 1
                elif ans in ["no", "n"]:
                    self.stats[key]["total"] += 1
                elif ans in ["probably", "py"]:
                    self.stats[key]["yes"] += 1
                    self.stats[key]["total"] += 1
                elif ans in ["probably_not", "pn"]:
                    self.stats[key]["total"] += 1

            logger.info(f"Loaded {len(df)} feedback entries across {len(self.stats)} (entity, question) pairs.")
        except Exception as e:
            logger.warning(f"Failed to load feedback logs: {e}")

    def log_game(self, history: list[dict[str, Any]], guessed_id: str, is_correct: bool, true_id: str) -> None:
        """Log game question-answer history to disk."""
        if not history or not true_id:
            return

        records = []
        for turn, entry in enumerate(history, 1):
            q_id = entry.get("q_id")
            ans = entry.get("answer")
            if q_id and ans:
                records.append({
                    "turn": turn,
                    "q_id": q_id,
                    "answer": ans,
                    "guessed_cinemind_id": guessed_id,
                    "true_cinemind_id": true_id,
                    "is_correct": is_correct
                })
                # In-memory update
                key = (true_id, q_id)
                if key not in self.stats:
                    self.stats[key] = {"yes": 0, "total": 0}
                
                ans_str = str(ans).lower().strip()
                if ans_str in ["yes", "y", "probably", "py"]:
                    self.stats[key]["yes"] += 1
                    self.stats[key]["total"] += 1
                elif ans_str in ["no", "n", "probably_not", "pn"]:
                    self.stats[key]["total"] += 1

        new_df = pd.DataFrame(records)
        if FEEDBACK_FILE.exists():
            try:
                old_df = pd.read_parquet(FEEDBACK_FILE, engine="fastparquet")
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.to_parquet(FEEDBACK_FILE, engine="fastparquet", index=False)
            except Exception:
                try:
                    new_df.to_parquet(FEEDBACK_FILE, engine="fastparquet", index=False)
                except Exception:
                    pass
        else:
            try:
                new_df.to_parquet(FEEDBACK_FILE, engine="fastparquet", index=False)
            except Exception:
                pass

    def recalibrate_likelihood_vector(
        self,
        entity_ids: pd.Series,
        q_id: str,
        synthetic_prior_vec: np.ndarray,
        k: float = 5.0
    ) -> np.ndarray:
        """
        Count-weighted Bayesian smoothing:
        p_yes = (synthetic_prior * k + observed_yes) / (k + total_observed)
        Vectorized O(k) lookup over stored (entity, q_id) pairs.
        """
        if not self.stats:
            return synthetic_prior_vec

        # Build cid_to_idx map once
        if not hasattr(self, "_cid_to_idx") or len(self._cid_to_idx) != len(entity_ids):
            self._cid_to_idx = {cid: idx for idx, cid in enumerate(entity_ids)}

        # Filter stats relevant ONLY to this q_id (typically 0-5 entries instead of 100,000 loop turns)
        matching_entries = [(cid, data) for (cid, q), data in self.stats.items() if q == q_id]
        if not matching_entries:
            return synthetic_prior_vec

        calibrated = synthetic_prior_vec.copy()
        for cid, data in matching_entries:
            if cid in self._cid_to_idx:
                idx = self._cid_to_idx[cid]
                tot_cnt = data["total"]
                if tot_cnt > 0:
                    yes_cnt = data["yes"]
                    prior = synthetic_prior_vec[idx]
                    calibrated[idx] = (prior * k + yes_cnt) / (k + tot_cnt)

        return calibrated


def log_game_feedback(history: list[dict[str, Any]], guessed_id: str, is_correct: bool, true_id: str) -> None:
    """Helper function to log game feedback."""
    store = FeedbackStore.get_instance()
    store.log_game(history, guessed_id, is_correct, true_id)


def get_calibrated_likelihood_vector(
    entity_ids: pd.Series,
    q_id: str,
    synthetic_prior_vec: np.ndarray,
    k: float = 5.0
) -> np.ndarray:
    """Helper function to apply count-weighted Bayesian recalibration."""
    store = FeedbackStore.get_instance()
    return store.recalibrate_likelihood_vector(entity_ids, q_id, synthetic_prior_vec, k=k)
