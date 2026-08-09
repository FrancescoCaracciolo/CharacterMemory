"""User facts: structured, multi-user memory with confidence/importance + decay."""

import json
from typing import TYPE_CHECKING, Any, Optional

from ..config import ContradictionPolicy
from .base import ExtractionSpec, MemoryItem
from .structured import StructuredMemory

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext


class UserFactMemory(StructuredMemory):
    """Facts about a user (occupation, preferences, …) and general facts.

    Each row: `type` (e.g. `preference`, `occupation`), `content`,
    `confidence` (0-1), plus the common importance/decay fields. High base
    importance makes a fact "sticky" (always in the prompt).
    """

    name = "user_facts"
    table = "user_facts"
    extra_columns = {
        "type": "TEXT NOT NULL DEFAULT 'general'",
        "content": "TEXT NOT NULL",
        "confidence": "REAL NOT NULL DEFAULT 0.5",
        # The chat a fact was learned in (NULL ⇒ legacy / single-user). Used
        # by the knowledge graph to link facts and episodes of the same chat.
        "chat_id": "TEXT",
        # Stable provenance into the messages table. Empty only for facts added
        # directly through the public API rather than conversation extraction.
        "source_message_ids": "TEXT NOT NULL DEFAULT '[]'",
    }
    text_column = "content"

    def add_fact(
        self,
        user_id: str,
        content: str,
        *,
        type: str = "general",
        importance: float = 0.5,
        confidence: float = 0.5,
        chat_id: Optional[str] = None,
        source_message_ids: Optional[list[int]] = None,
    ) -> int:
        return self.add(
            user_id,
            importance,
            type=type,
            content=content,
            confidence=confidence,
            chat_id=chat_id,
            source_message_ids=json.dumps(source_message_ids or []),
        )

    def row_text(self, row: dict[str, Any]) -> str:
        return f"[{row.get('type', 'general')}] {row.get('content', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        text = f"{row.get('content', '')} (type: {row.get('type', 'general')}, confidence: {row.get('confidence', 0):.2f})"
        return MemoryItem(text=text, score=score, kind=self.name, metadata=dict(row))

    def contradiction_policy(self) -> ContradictionPolicy:
        # Stable facts (occupation, preferences, …) contradict when they assert
        # incompatible current truths — "doctor" vs "engineer". Timestamps let
        # the judge weigh recency, but the default policy opts into the gate.
        return ContradictionPolicy(enabled=True)

    # Extraction
    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        user = context.user_name if context else "the user"
        return ExtractionSpec(
            field="facts",
            per_user=True,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "content": {"type": "string"},
                        "importance": {"type": "number"},
                        "confidence": {"type": "number"},
                        "source_message_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": [
                        "type", "content", "importance", "confidence",
                        "source_message_ids",
                    ],
                },
            },
            instruction=(
                f"- facts: stable facts about {user} (occupation, preferences, "
                f"relationships, goals) or general facts {user} stated. Each "
                f"`content` must be one self-contained full sentence about {user} "
                f"(e.g. \"{user} has an exam on the 17th of July\"). importance 0-1 "
                f"(how much it should shape the character's behaviour), confidence "
                f"0-1."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        added: list[MemoryItem] = []
        for f in value or []:
            content = (f.get("content") or "").strip()
            source_message_ids = f.get("source_message_ids") or []
            if (
                not content
                or not source_message_ids
                or self._has_text(user_id, content, "content")
            ):
                continue
            # Multi-user: attribute to the participant the LLM named, else the
            # caller's default user (the chat owner / current speaker).
            uid = str(f.get("user_id") or user_id)
            if uid != user_id and self._has_text(uid, content, "content"):
                continue
            row_id = self.add_fact(
                uid,
                content,
                type=str(f.get("type", "general")),
                importance=self._clip(f.get("importance", 0.5)),
                confidence=self._clip(f.get("confidence", 0.5)),
                chat_id=chat_id,
                source_message_ids=source_message_ids,
            )
            row = self.get_row(row_id)
            if row is not None:
                added.append(self.row_item(row, self._effective(row)))
        return added
