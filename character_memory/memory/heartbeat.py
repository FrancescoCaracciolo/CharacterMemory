"""Heartbeat journal: discoveries/actions the character makes on its own."""

from collections.abc import Iterable
from typing import Any

from .base import MemoryItem, MemoryScope
from .structured import StructuredMemory


class HeartbeatJournal(StructuredMemory):
    """Structured log of the character's autonomous web discoveries/actions.

    Each row: `summary` (what it found/did), `kind` (`discovery` /
    `action`), plus importance/decay fields. Not populated by ordinary chat, but by
    the character's autonomous "browsing" loop writes here.
    """

    name = "heartbeat"
    table = "heartbeat"
    # The journal is the character's own log; it already ignores `user_id` in
    # recall. Scope it CHARACTER so a group chat recalls it once instead of per
    # participant.
    scope = MemoryScope.CHARACTER
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

    def _recall_where(self, user_id: str) -> None:
        # The journal is character-scoped, not per-user.
        return None

    def _sticky_rows(
        self, rows: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Journal importance affects ranking but never forces prompt inclusion.
        return []

    def row_text(self, row: dict[str, Any]) -> str:
        return f"[{row.get('kind', 'discovery')}] {row.get('summary', '')}"

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        return MemoryItem(text=row.get("summary", ""), score=score, kind=self.name, metadata=dict(row))
