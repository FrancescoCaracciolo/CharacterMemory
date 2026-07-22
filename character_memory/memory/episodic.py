"""Episodic memory: things that happened, decayed + weighted by emotional shift."""

from typing import TYPE_CHECKING, Any, Optional
from ..config import ContradictionPolicy
from .decay import decay_score, age_seconds
from .base import ExtractionSpec, MemoryItem
from .structured import StructuredMemory

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext

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
        # The chat an episode was learned in (NULL ⇒ legacy / single-user).
        # Used by the knowledge graph to link facts and episodes of the same chat.
        "chat_id": "TEXT",
    }
    text_column = "summary"

    def add_episode(
        self,
        user_id: str,
        summary: str,
        *,
        importance: float = 0.5,
        emotional_shift: float = 0.0,
        chat_id: Optional[str] = None,
    ) -> int:
        return self.add(
            user_id,
            importance,
            summary=summary,
            emotional_shift=float(emotional_shift),
            chat_id=chat_id,
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

    def contradiction_policy(self) -> ContradictionPolicy:
        # Episodic summaries are looser than bare facts; lower bar so the gate
        # fires on genuinely incompatible event summaries. Timestamps are
        # critical here: "user was sad Monday" vs "user was happy Tuesday" is
        # a change over time, NOT a contradiction, and the judge needs the
        # timestamps to tell them apart.
        return ContradictionPolicy(enabled=True, similarity_threshold=0.65)

    def row_text(self, row: dict[str, Any]) -> str:
        return f"{row.get('summary', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        return MemoryItem(text=row.get("summary", ""), score=score, kind=self.name, metadata=dict(row))

    # Extraction ----------------------------------------------------------
    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        char = context.character_name if context else "the character"
        user = context.user_name if context else "the user"
        return ExtractionSpec(
            field="episodes",
            per_user=True,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "importance": {"type": "number"},
                        "emotional_shift": {"type": "number"},
                    },
                    "required": ["summary", "importance", "emotional_shift"],
                },
            },
            instruction=(
                f"- episodes: notable things that happened between {char} and "
                f"{user}, written as full sentences from {char}'s point of view. "
                f"emotional_shift -1..1 (negative to positive) captures how the "
                f"event shifted {char}'s feelings toward {user}."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        added: list[MemoryItem] = []
        for e in value or []:
            summary = (e.get("summary") or "").strip()
            if not summary:
                continue
            emotional_shift = float(e.get("emotional_shift", 0.0))
            # Multi-user: attribute to the participant the LLM named, else the
            # caller's default user (the chat owner / current speaker).
            uid = str(e.get("user_id") or user_id)
            row_id = self.add_episode(
                uid,
                summary,
                importance=self._clip(e.get("importance", 0.5)),
                emotional_shift=emotional_shift,
                chat_id=chat_id,
            )
            row = self.get_row(row_id)
            if row is not None:
                added.append(self.row_item(row, self._effective(row)))
        return added
