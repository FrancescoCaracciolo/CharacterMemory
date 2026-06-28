"""Emotion status: a user-independent baseline plus configurable per-user dims."""

import json
from typing import Any, Optional

from .base import ExtractionSpec, Memory, MemoryItem
from ..chunking import Chunk
from .store import SQLiteStore


class EmotionStatus(Memory):
    """Tracks how the character feels.

    A fixed `baseline` (user-independent: `joy`, `sadness`…) describes the
    character's resting state. On top of that, a set of per-user dimensions
    (default `affection`, `valence`, `trust` - configurable) tracks how
    the character feels *toward each user*.
    """

    name = "emotion"

    def __init__(
        self,
        store: SQLiteStore,
        *,
        enabled: bool = True,
        baseline: Optional[dict[str, float]] = None,
        user_dims: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.store = store
        self.baseline = dict(baseline or {"neutral": 0.5, "joy": 0.2, "sadness": 0.1, "anger": 0, "anxiety": 0})
        self.user_dims = dict(user_dims or {"affection": 0.0, "valence": 0.0, "trust": 0.0})
        self.table = "emotion"
        self.store.create_table(
            self.table,
            {"user_id": "TEXT PRIMARY KEY", "state": "TEXT NOT NULL"},
        )

    # State
    def _default_user_state(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.user_dims.items()}

    def get_user_state(self, user_id: str) -> dict[str, float]:
        rows = self.store.select(self.table, {"user_id": user_id})
        if rows:
            try:
                state = json.loads(rows[0]["state"])
                # Ensure all configured dims exist.
                return {**self._default_user_state(), **state}
            except (TypeError, ValueError):
                pass
        return self._default_user_state()

    def set_user_state(self, user_id: str, state: dict[str, float]) -> None:
        merged = {**self._default_user_state(), **{k: float(v) for k, v in state.items()}}
        self.store.upsert(self.table, {"user_id": user_id, "state": json.dumps(merged)}, pk="user_id")

    def update(self, user_id: str, deltas: dict[str, float]) -> dict[str, float]:
        """Apply signed `deltas` to the per-user dims, clamped to `[-1, 1]`."""
        current = self.get_user_state(user_id)
        for k, v in deltas.items():
            if k in current:
                current[k] = max(-1.0, min(1.0, current[k] + float(v)))
        self.set_user_state(user_id, current)
        return current

    # Recall
    def recall(self, query: str, user_id: str, limit: int) -> list[MemoryItem]:
        state = self.get_user_state(user_id)
        parts = [f"{k}={v:.2f}" for k, v in self.baseline.items()]
        parts += [f"{k}(toward {user_id})={v:.2f}" for k, v in state.items()]
        return [MemoryItem(text=p, score=1.0, kind=self.name) for p in parts]

    # Implement not needed methods
    def build(self, info_chunks: list[Chunk]) -> None:
        return None

    def load(self, path: str) -> None:
        return None

    def persist(self, path: str) -> None:
        return None

    # Extraction ----------------------------------------------------------
    def extraction_spec(self) -> ExtractionSpec:
        dims = ", ".join(self.user_dims) or "affection, valence, trust"
        return ExtractionSpec(
            field="emotion_deltas",
            schema={
                "type": "object",
                "description": "Signed adjustments to per-user emotion dims.",
                "additionalProperties": {"type": "number"},
            },
            instruction=(
                f"- emotion_deltas: small signed adjustments to the character's "
                f"feelings toward this user. Allowed dims: {dims}."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str) -> None:
        if value:
            self.update(user_id, value)
