"""User facts: structured, multi-user memory with confidence/importance + decay."""

from typing import Any

from .base import MemoryItem
from .structured import StructuredMemory


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
    }

    def add_fact(
        self,
        user_id: str,
        content: str,
        *,
        type: str = "general",
        importance: float = 0.5,
        confidence: float = 0.5,
    ) -> int:
        return self.add(
            user_id,
            importance,
            type=type,
            content=content,
            confidence=confidence,
        )

    def row_text(self, row: dict[str, Any]) -> str:
        return f"[{row.get('type', 'general')}] {row.get('content', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        text = f"{row.get('content', '')} (type: {row.get('type', 'general')}, confidence: {row.get('confidence', 0):.2f})"
        return MemoryItem(text=text, score=score, kind=self.name, metadata=dict(row))
