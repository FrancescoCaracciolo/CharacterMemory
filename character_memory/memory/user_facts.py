"""User facts: structured, multi-user memory with confidence/importance + decay."""

from typing import Any

from .base import ExtractionSpec, MemoryItem
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

    # Extraction ----------------------------------------------------------
    def extraction_spec(self) -> ExtractionSpec:
        return ExtractionSpec(
            field="facts",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "content": {"type": "string"},
                        "importance": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["type", "content", "importance", "confidence"],
                },
            },
            instruction=(
                "- facts: stable facts about the user (occupation, preferences, "
                "relationships, goals) or general facts the user stated. importance "
                "0-1 (how much it should shape the character's behaviour), "
                "confidence 0-1."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str) -> None:
        for f in value or []:
            content = (f.get("content") or "").strip()
            if not content or self._has_text(user_id, content, "content"):
                continue
            self.add_fact(
                user_id,
                content,
                type=str(f.get("type", "general")),
                importance=self._clip(f.get("importance", 0.5)),
                confidence=self._clip(f.get("confidence", 0.5)),
            )
