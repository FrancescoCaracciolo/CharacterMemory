"""Heartbeat journal: discoveries/actions the character makes on its own."""

from typing import Any

from .base import MemoryItem
from .structured import StructuredMemory


class HeartbeatJournal(StructuredMemory):
    """Structured log of the character's autonomous web discoveries/actions.

    Each row: `summary` (what it found/did), `kind` (`discovery` /
    `action`), plus importance/decay fields. Not populated by ordinary chat, but by
    the character's autonomous "browsing" loop writes here.
    """

    name = "heartbeat"
    table = "heartbeat"
    extra_columns = {
        "summary": "TEXT NOT NULL",
        "kind": "TEXT NOT NULL DEFAULT 'discovery'",
    }
    text_column = "summary"

    def add_entry(
        self,
        summary: str,
        *,
        kind: str = "discovery",
        importance: float = 0.5,
        user_id: str = "_self",
    ) -> int:
        return self.add(user_id, importance, summary=summary, kind=kind)

    def recall(
        self,
        query: str,
        user_id: str,
        limit: int,
        sticky_limit: int = 10,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        # The journal is character-scoped, not per-user; ignore user_id.
        # There are no sticky memories in the journal.
        rows = self.store.select(self.table, order_by="id DESC", limit=self.hybrid.candidate_pool)
        if not rows:
            return []
        hits = self.hybrid.search(query, k=limit)
        ids = [int(h.metadata.get("id")) for h in hits if h.metadata.get("id") is not None]
        rows_by_id = {r["id"]: r for r in rows}
        scored = [(self._effective(rows_by_id[i]), rows_by_id[i]) for i in dict.fromkeys(ids) if i in rows_by_id]
        scored.sort(key=lambda kv: kv[0], reverse=True)
        chosen = scored[:limit]
        if state_changing:
            self._bump_recall([row["id"] for _, row in chosen])
        return [self.row_item(row, score) for score, row in chosen]

    def row_text(self, row: dict[str, Any]) -> str:
        return f"[{row.get('kind', 'discovery')}] {row.get('summary', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        return MemoryItem(text=row.get("summary", ""), score=score, kind=self.name, metadata=dict(row))
