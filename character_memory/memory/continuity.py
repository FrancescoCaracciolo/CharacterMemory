"""Shared evidence and update plumbing for conversational continuity memories."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Optional

from .base import MemoryItem
from .extract import ExtractionContext
from .structured import StructuredMemory


class ContinuityMemory(StructuredMemory):
    """Structured memory whose extracted mutations require transcript evidence.

    Subclasses implement ``_apply_update``; storage, indexing and recall remain
    in StructuredMemory. Direct callers can use ``apply_extraction`` with a
    persisted chat, or ``apply_extraction_with_context`` with transcript turns.
    """

    evidence_roles = frozenset({"user", "assistant"})
    continuity_columns = {
        "content": "TEXT NOT NULL",
        "chat_id": "TEXT",
        "source_message_ids": "TEXT NOT NULL DEFAULT '[]'",
        "history": "TEXT NOT NULL DEFAULT '[]'",
        "updated_at": "REAL",
    }

    @staticmethod
    def people(context: ExtractionContext) -> list[str]:
        return list(dict.fromkeys(context.participants or [context.user_name]))

    def apply_extraction(
        self, value: Any, user_id: str, *, chat_id: Optional[str] = None
    ) -> list[MemoryItem]:
        turns = []
        participants = [user_id]
        if chat_id is not None and self.store.columns("messages"):
            rows = self.store.select("messages", {"chat_id": chat_id}, order_by="id")
            turns = [{**row, "message_id": row["id"]} for row in rows]
            participants = list(dict.fromkeys(
                str(row.get("user_id") or user_id)
                for row in rows if row.get("role") == "user"
            )) or participants
        return self.apply_extraction_with_context(
            value, ExtractionContext(user_name=user_id, participants=participants, turns=turns),
            chat_id=chat_id,
        )

    def apply_extraction_with_context(
        self, value: Any, context: ExtractionContext, *, chat_id: Optional[str] = None
    ) -> list[MemoryItem]:
        if not self.enabled or not isinstance(value, list):
            return []
        evidence = {
            turn.get("message_id"): turn for turn in context.turns
            if turn.get("role") in self.evidence_roles
            and isinstance(turn.get("message_id"), int)
        }
        changed: dict[int, MemoryItem] = {}
        for update in value:
            if not isinstance(update, dict):
                continue
            supplied = update.get("source_message_ids")
            if not isinstance(supplied, list):
                continue
            ids = list(dict.fromkeys(
                mid for mid in supplied
                if isinstance(mid, int) and not isinstance(mid, bool) and mid in evidence
            ))
            if not ids:
                continue
            try:
                row_id = self._apply_update(update, context, chat_id, ids)
            except (KeyError, TypeError, ValueError):
                # A malformed update must not drop other valid updates.
                continue
            row = self.get_row(row_id) if row_id is not None else None
            if row is not None:
                changed[row_id] = self.row_item(row, self._effective(row))
        return list(changed.values())

    @abstractmethod
    def _apply_update(self, update, context, chat_id, source_ids) -> Optional[int]:
        raise NotImplementedError

    def _save_update(self, row: dict[str, Any], source_ids: list[int], **changes) -> int:
        """Retain prior state/evidence and mirror a mutation into the shared RAG."""
        sources = list(dict.fromkeys([*json.loads(row["source_message_ids"]), *source_ids]))
        changes["source_message_ids"] = json.dumps(sources)
        if all(row.get(key) == value for key, value in changes.items()):
            return int(row["id"])
        history = json.loads(row["history"])
        history.append({key: value for key, value in row.items() if key != "history"})
        self.update_row({
            **row, **changes, "history": json.dumps(history), "updated_at": self._now(),
        })
        self.apply_index_changes(updated_ids=[row["id"]])
        return int(row["id"])


SOURCE_IDS_SCHEMA = {"type": "array", "items": {"type": "integer"}}
