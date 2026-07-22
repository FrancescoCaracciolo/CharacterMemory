"""User directives: per-user standing instructions with importance + keywords."""

import json
from typing import TYPE_CHECKING, Any, Optional

from .base import ExtractionSpec, MemoryItem
from .structured import StructuredMemory

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext


class UserDirectiveMemory(StructuredMemory):
    """Standing instructions from a user.

    Each row: `content` (the instruction), `retrieval_keywords` (a JSON list
    whose presence in the query boosts recall), plus importance/decay fields.
    """

    name = "user_directives"
    table = "user_directives"
    extra_columns = {
        "content": "TEXT NOT NULL",
        "retrieval_keywords": "TEXT NOT NULL DEFAULT '[]'",  # JSON array
    }
    text_column = "content"

    def add_directive(
        self,
        user_id: str,
        content: str,
        *,
        importance: float = 0.5,
        retrieval_keywords: list[str] | None = None,
    ) -> int:
        return self.add(
            user_id,
            importance,
            content=content,
            retrieval_keywords=json.dumps(retrieval_keywords or []),
        )

    def _keywords(self, row: dict[str, Any]) -> list[str]:
        try:
            return list(json.loads(row.get("retrieval_keywords") or "[]"))
        except (TypeError, ValueError):
            return []

    def recall(
        self,
        query: str,
        user_id: str,
        limit: int,
        sticky_limit: int = 10,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        items = super().recall(query, user_id, limit, sticky_limit, state_changing=state_changing)
        # Keyword boost: directives whose keywords appear in the query are
        # surfaced even if hybrid recall missed them.
        ql = query.lower()
        rows_by_id = {r["id"]: r for r in self.store.select(self.table, {"user_id": user_id})}
        have = {int(it.metadata["id"]) for it in items}
        boosted = []
        for rid, row in rows_by_id.items():
            if rid in have:
                continue
            if any(kw and kw.lower() in ql for kw in self._keywords(row)):
                boosted.append((self._effective(row), row))
        boosted.sort(key=lambda kv: kv[0], reverse=True)
        for score, row in boosted[: max(0, limit - len(items))]:
            items.append(self.row_item(row, score))
        return items

    def row_text(self, row: dict[str, Any]) -> str:
        kws = self._keywords(row)
        kw_part = f" (keywords: {', '.join(kws)})" if kws else ""
        return f"{row.get('content', '')}{kw_part}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        return MemoryItem(text=row.get("content", ""), score=score, kind=self.name, metadata=dict(row))

    # Extraction ----------------------------------------------------------
    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        user = context.user_name if context else "the user"
        char = context.character_name if context else "the character"
        return ExtractionSpec(
            field="directives",
            per_user=True,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "importance": {"type": "number"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content", "importance", "keywords"],
                },
            },
            instruction=(
                f"- directives: standing instructions {user} asked {char} to follow "
                f"(e.g. \"{user} wants {char} to always answer formally\"). Each "
                f"`content` must be one self-contained full sentence. importance 0-1. "
                f"keywords: terms that should trigger retrieval."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        added: list[MemoryItem] = []
        for d in value or []:
            content = (d.get("content") or "").strip()
            if not content or self._has_text(user_id, content, "content"):
                continue
            keywords = d.get("keywords") or []
            # Multi-user: attribute to the participant the LLM named, else the
            # caller's default user (the chat owner / current speaker).
            uid = str(d.get("user_id") or user_id)
            if uid != user_id and self._has_text(uid, content, "content"):
                continue
            row_id = self.add_directive(
                uid,
                content,
                importance=self._clip(d.get("importance", 0.5)),
                retrieval_keywords=keywords,
            )
            row = self.get_row(row_id)
            if row is not None:
                added.append(self.row_item(row, self._effective(row)))
        return added
