"""User directives: per-user standing instructions with importance + keywords."""

import json
from typing import Any

from .base import MemoryItem
from .structured import StructuredMemory


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

    def recall(self, query: str, user_id: str, limit: int, sticky_limit: int = 10) -> list[MemoryItem]:
        items = super().recall(query, user_id, limit, sticky_limit)
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
