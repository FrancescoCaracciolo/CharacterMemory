"""Episodic memory: things that happened, decayed + weighted by emotional shift."""

from typing import Any
from .decay import decay_score, age_seconds
from .base import MemoryItem
from .structured import StructuredMemory

class EpisodicMemory(StructuredMemory):
    """Events the character experienced with a user.

    Each row: `summary` (what happened), `emotional_shift` (signed float;
    positive events amplify importance), plus importance/decay fields.
    """

    name = "episodic"
    table = "episodic"
    extra_columns = {
        "summary": "TEXT NOT NULL",
        "emotional_shift": "REAL NOT NULL DEFAULT 0.0",
    }

    def add_episode(
        self,
        user_id: str,
        summary: str,
        *,
        importance: float = 0.5,
        emotional_shift: float = 0.0,
    ) -> int:
        return self.add(
            user_id,
            importance,
            summary=summary,
            emotional_shift=float(emotional_shift),
        )

    def _effective(self, row: dict[str, Any]) -> float:
        # Emotional magnitude contributes to how strongly an episode is retained.        
        return decay_score(
            base_importance=float(row["importance"]),
            recall_count=int(row.get("recall_count", 0)),
            age_seconds=age_seconds(
                float(row["created_at"]), row.get("last_recalled"), self._now()
            ),
            half_life=self.half_life,
            emotion_impact=float(row.get("emotional_shift", 0.0)),
        )

    def row_text(self, row: dict[str, Any]) -> str:
        return f"{row.get('summary', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        return MemoryItem(text=row.get("summary", ""), score=score, kind=self.name, metadata=dict(row))
