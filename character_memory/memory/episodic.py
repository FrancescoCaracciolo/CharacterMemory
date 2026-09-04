"""Episodic memory: events weighted by emotional impact and mood congruence."""

import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional
from ..config import ContradictionPolicy
from ..emotion_vectors import (
    decode_emotion_vector,
    emotion_similarity,
    emotion_vector,
    emotional_impact,
    encode_emotion_vector,
)
from .decay import decay_score, age_seconds, intrinsic_score
from .base import ExtractionSpec, MemoryItem
from .structured import StructuredMemory

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext

class EpisodicMemory(StructuredMemory):
    """Events the character experienced with a user.

    Each row stores a summary and a sparse ``emotional_shift`` vector whose
    axes identify the emotions that caused the impact.
    """

    name = "episodic"
    table = "episodic"
    extra_columns = {
        "summary": "TEXT NOT NULL",
        "emotional_shift": "TEXT NOT NULL DEFAULT '{}'",
        # The chat an episode was learned in (NULL ⇒ legacy / single-user).
        # Used by the knowledge graph to link facts and episodes of the same chat.
        "chat_id": "TEXT",
        "source_message_ids": "TEXT NOT NULL DEFAULT '[]'",
        # Semantic event time, distinct from when extraction stored the row.
        "occurred_at": "REAL",
    }
    text_column = "summary"

    def __init__(
        self,
        *args: Any,
        emotion_baseline: Optional[Mapping[str, float]] = None,
        current_mood: Optional[Callable[[], Mapping[str, float]]] = None,
        **kwargs: Any,
    ) -> None:
        self.emotion_baseline = emotion_vector(
            emotion_baseline
            or {"neutral": 0.5, "joy": 0.2, "sadness": 0.1, "anxiety": 0.0, "anger": 0.0}
        )
        self._current_mood_provider = current_mood
        super().__init__(*args, **kwargs)
        if "occurred_at" not in self.store.columns(self.table):
            try:
                self.store.execute(
                    f"ALTER TABLE {self.table} ADD COLUMN occurred_at REAL"
                )
            except sqlite3.OperationalError:
                if "occurred_at" not in self.store.columns(self.table):
                    raise
        self._backfill_occurrence_times()

    def temporal_interval(
        self, row: dict[str, Any]
    ) -> tuple[float, Optional[float]] | None:
        occurred_at = row.get("occurred_at")
        return (float(occurred_at), None) if occurred_at is not None else None

    def _source_timestamp(self, message_ids: list[int]) -> Optional[float]:
        if not message_ids:
            return None
        placeholders = ",".join("?" for _ in message_ids)
        try:
            rows = self.store.execute(
                f"SELECT occurred_at, created_at FROM messages WHERE id IN ({placeholders}) ORDER BY id ASC",
                [int(value) for value in message_ids],
            )
        except sqlite3.OperationalError:
            return None
        for row in rows:
            value = row.get("occurred_at")
            if value is None:
                value = row.get("created_at")
            if value is not None:
                return float(value)
        return None

    def _backfill_occurrence_times(self) -> None:
        for row in self.store.select(self.table):
            if row.get("occurred_at") is not None:
                continue
            try:
                ids = [int(value) for value in json.loads(row.get("source_message_ids") or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError):
                ids = []
            value = self._source_timestamp(ids)
            if value is None:
                continue
            row["occurred_at"] = value
            self.update_row(row)

    def _current_mood(self) -> dict[str, float]:
        if self._current_mood_provider is None:
            return dict(self.emotion_baseline)
        return emotion_vector(
            self._current_mood_provider(), allowed_axes=self.emotion_baseline
        )

    def _vector(self, row: Mapping[str, Any]) -> dict[str, float]:
        return decode_emotion_vector(
            row.get("emotional_shift", "{}"), allowed_axes=self.emotion_baseline
        )

    def add_episode(
        self,
        user_id: str,
        summary: str,
        *,
        importance: float = 0.5,
        emotional_shift: Mapping[str, float] | None = None,
        chat_id: Optional[str] = None,
        source_message_ids: Optional[list[int]] = None,
        occurred_at: Optional[float] = None,
    ) -> int:
        return self.add(
            user_id,
            importance,
            summary=summary,
            emotional_shift=encode_emotion_vector(
                {} if emotional_shift is None else emotional_shift,
                allowed_axes=self.emotion_baseline,
            ),
            chat_id=chat_id,
            source_message_ids=json.dumps(source_message_ids or []),
            occurred_at=(
                float(occurred_at)
                if occurred_at is not None
                else self._source_timestamp(source_message_ids or [])
            ),
        )

    def _effective(self, row: dict[str, Any]) -> float:
        vector = self._vector(row)
        impact = emotional_impact(vector)
        similarity = emotion_similarity(vector, self._current_mood())
        base = decay_score(
            base_importance=float(row["importance"]),
            recall_count=int(row.get("recall_count", 0)),
            age_seconds=age_seconds(
                self._age_anchor(row), row.get("last_recalled"), self._now()
            ),
            half_life=self.half_life,
            emotion_impact=impact,
        )
        return base * (1.0 + similarity)

    def _intrinsic(self, row: dict[str, Any]) -> float:
        vector = self._vector(row)
        return intrinsic_score(
            float(row["importance"]), emotional_impact(vector)
        ) * (1.0 + emotion_similarity(vector, self._current_mood()))

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
        vector = self._vector(row)
        metadata = dict(row)
        metadata["emotional_shift"] = vector
        impact = emotional_impact(vector)
        similarity = emotion_similarity(vector, self._current_mood())
        metadata["raw_emotional_impact"] = impact
        metadata["emotion_similarity"] = similarity
        metadata["impact"] = impact
        metadata["similarity"] = similarity
        return MemoryItem(text=row.get("summary", ""), score=score, kind=self.name, metadata=metadata)

    # Extraction ----------------------------------------------------------
    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        char = context.character_name if context else "the character"
        user = context.user_name if context else "the user"
        shift_schema = {
            "type": "object",
            "properties": {
                axis: {"type": "number", "minimum": 0, "maximum": 1}
                for axis in self.emotion_baseline
            },
            "additionalProperties": False,
        }
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
                        "emotional_shift": shift_schema,
                        "source_message_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": [
                        "summary", "importance", "emotional_shift",
                        "source_message_ids",
                    ],
                },
            },
            instruction=(
                f"- episodes: notable things that happened between {char} and "
                f"{user}, written as full sentences from {char}'s point of view. "
                f"emotional_shift is a sparse 0..1 emotion vector identifying what "
                f"{char} felt because of the event. Allowed axes: "
                f"{', '.join(self.emotion_baseline)}."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        added: list[MemoryItem] = []
        for e in value or []:
            summary = (e.get("summary") or "").strip()
            source_message_ids = e.get("source_message_ids") or []
            if not summary or not source_message_ids:
                continue
            emotional_shift = e.get("emotional_shift", {})
            # Deliberately strict: scalar shifts are not part of this model.
            # Unknown mapping keys are different: an OpenAI-compatible model
            # may ignore ``additionalProperties: false`` and invent a close
            # synonym such as ``annoyance``. Discard those at this untrusted
            # boundary while keeping direct ``add_episode`` calls strict.
            emotional_shift = emotion_vector(
                emotional_shift,
                allowed_axes=self.emotion_baseline,
                ignore_unknown_axes=True,
            )
            # Multi-user: attribute to the participant the LLM named, else the
            # caller's default user (the chat owner / current speaker).
            uid = str(e.get("user_id") or user_id)
            row_id = self.add_episode(
                uid,
                summary,
                importance=self._clip(e.get("importance", 0.5)),
                emotional_shift=emotional_shift,
                chat_id=chat_id,
                source_message_ids=source_message_ids,
                occurred_at=self._source_timestamp(source_message_ids),
            )
            row = self.get_row(row_id)
            if row is not None:
                added.append(self.row_item(row, self._effective(row)))
        return added
