"""
Checkpoint / resume system for long-running discovery.

Features:
  - JSON-based state persistence
  - Atomic writes (temp file → rename) for crash safety
  - Graceful Ctrl+C handling with cooperative shutdown flag
  - Tracks completed tasks and partial progress
"""

import json
import logging
import signal
import tempfile
from pathlib import Path
from typing import Any

from pipeline.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


# ============================================================
# Graceful shutdown
# ============================================================

class GracefulShutdown:
    """Cooperative shutdown flag set by SIGINT handler."""

    _requested = False
    _handler_installed = False
    _press_count = 0

    @classmethod
    def install(cls) -> None:
        """Install SIGINT handler. Safe to call multiple times."""
        if cls._handler_installed:
            return
        signal.signal(signal.SIGINT, cls._handle_signal)
        cls._handler_installed = True

    @classmethod
    def _handle_signal(cls, signum: int, frame: Any) -> None:
        cls._press_count += 1
        if cls._press_count >= 2:
            logger.warning("Forced shutdown.")
            raise KeyboardInterrupt("Forced shutdown")
        cls._requested = True
        logger.warning(
            "Shutdown requested. Finishing current task... "
            "(Ctrl+C again to force)"
        )

    @classmethod
    def is_requested(cls) -> bool:
        return cls._requested

    @classmethod
    def reset(cls) -> None:
        cls._requested = False
        cls._press_count = 0


# ============================================================
# Checkpoint manager
# ============================================================

class CheckpointManager:
    """Manages checkpoint state for resumable discovery."""

    def __init__(self, name: str) -> None:
        self.name = name
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / f"{name}_state.json"
        self.state: dict[str, Any] = self._load()

        # Use a set internally for O(1) lookup
        self._completed: set[str] = set(
            self.state.get("completed", [])
        )

    def _load(self) -> dict[str, Any]:
        """Load checkpoint from disk."""
        if self.path.exists():
            try:
                data = json.loads(
                    self.path.read_text(encoding="utf-8")
                )
                completed_count = len(data.get("completed", []))
                logger.info(
                    "Loaded checkpoint '%s': %d completed tasks",
                    self.name, completed_count,
                )
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load checkpoint '%s': %s",
                    self.name, exc,
                )
        return {"completed": [], "progress": {}, "stats": {}}

    def save(self) -> None:
        """Atomically save checkpoint to disk."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Sync set back to list for serialization
        self.state["completed"] = sorted(self._completed)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(CHECKPOINT_DIR),
                suffix=".tmp",
            )
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)

            Path(tmp_path).replace(self.path)

        except OSError as exc:
            logger.error("Failed to save checkpoint: %s", exc)

    def is_complete(self, task_key: str) -> bool:
        """Check if a task has already been completed."""
        return task_key in self._completed

    def get_progress(self, task_key: str) -> int:
        """Get the last completed page/offset for a partial task."""
        return self.state.get("progress", {}).get(task_key, 0)

    def update_progress(self, task_key: str, value: int) -> None:
        """Update partial progress for a task (e.g., page number)."""
        self.state.setdefault("progress", {})[task_key] = value

    def complete_task(self, task_key: str) -> None:
        """Mark a task as fully completed and persist."""
        self._completed.add(task_key)
        # Remove from progress since it's finished
        self.state.get("progress", {}).pop(task_key, None)
        self.save()

    def update_stat(self, key: str, value: Any) -> None:
        """Store a named statistic in the checkpoint."""
        self.state.setdefault("stats", {})[key] = value

    def get_stat(self, key: str, default: Any = None) -> Any:
        """Retrieve a named statistic from the checkpoint."""
        return self.state.get("stats", {}).get(key, default)

    @property
    def completed_count(self) -> int:
        return len(self._completed)
