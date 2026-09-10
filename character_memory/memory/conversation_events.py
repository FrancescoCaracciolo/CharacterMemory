"""Immutable source conversation rounds with fact-like retrieval aliases."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any, Optional

from ..chunking.base import Chunk
from .base import Memory, MemoryItem
from .structured import StructuredMemory


class ConversationEventMemory(StructuredMemory):
    """Small raw conversation events retained as retrieval values.

    One event is normally a user message plus the assistant messages that
    follow it, ending before the next user message. The raw exchange is the
    value rendered into context. Extracted facts, directives, and episodes are
    stored as additional index keys which all point back to that same event.
    """

    name = "conversation_events"
    table = "conversation_events"
    text_column = "content"
    rank_by_relevance = True
    extra_columns = {
        "chat_id": "TEXT NOT NULL",
        "content": "TEXT NOT NULL",
        "occurred_at": "REAL",
        "source_message_ids": "TEXT NOT NULL",
        "first_message_id": "INTEGER NOT NULL",
        "last_message_id": "INTEGER NOT NULL",
    }

    _KEY_TABLE = "conversation_event_keys"
    _KEY_COLUMNS = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "event_id": "INTEGER NOT NULL",
        "key_text": "TEXT NOT NULL",
        "key_kind": "TEXT NOT NULL",
        "source_memory": "TEXT NOT NULL",
        "source_row_id": "INTEGER NOT NULL",
        "confidence": "REAL",
        "created_at": "REAL NOT NULL",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store.create_table(self._KEY_TABLE, self._KEY_COLUMNS)
        if getattr(self.hybrid, "remote", False):
            self.store.track_table(self._KEY_TABLE, key="event_id", collection=self.hybrid.collection)
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
                message_ids,
            )
        except self.store.operational_errors:
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
            value = self._source_timestamp(self._decode_ids(row.get("source_message_ids")))
            if value is None:
                continue
            row["occurred_at"] = value
            self.update_row(row)

    @staticmethod
    def _decode_ids(value: Any) -> list[int]:
        if isinstance(value, str):
            value = json.loads(value)
        return [int(item) for item in (value or [])]

    @staticmethod
    def _timestamp(value: float) -> str:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()

    @classmethod
    def _render_message(
        cls, row: dict[str, Any], *, default_user_id: str, character_name: str
    ) -> str:
        if row["role"] == "user":
            speaker = str(row.get("user_id") or default_user_id)
        else:
            speaker = character_name
        timestamp = row.get("occurred_at")
        if timestamp is None:
            timestamp = row.get("created_at")
        prefix = f"[{cls._timestamp(timestamp)}] " if timestamp is not None else ""
        return f"{prefix}{speaker}: {row['content']}"

    def _row_meta(self, row: dict[str, Any]) -> dict:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "_result_id": f"conversation_event:{row['id']}",
        }

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        metadata = dict(row)
        metadata["source_message_ids"] = self._decode_ids(
            metadata["source_message_ids"]
        )
        return MemoryItem(
            text=row["content"],
            score=score,
            kind=self.name,
            metadata=metadata,
        )

    def format(self, items: list[MemoryItem]) -> str:
        return "\n\n".join(
            f"<conversation-event>\n{item.text}\n</conversation-event>"
            for item in items
        )

    def search_events(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        occurred_from: Optional[float] = None,
        occurred_before: Optional[float] = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search raw events with optional occurrence-time bounds.

        Time bounds use ``[occurred_from, occurred_before)``. When bounds are
        supplied the hybrid candidate pass is widened to the complete event
        index before filtering, so a relevant in-range event cannot be hidden
        by out-of-range aliases occupying the normal candidate pool.
        """
        limit = max(1, int(limit))
        rows = self.store.select(
            self.table, {"user_id": user_id} if user_id else None
        )

        def eligible(row: dict[str, Any]) -> bool:
            timestamp = row.get("occurred_at")
            if occurred_from is not None or occurred_before is not None:
                if timestamp is None:
                    return False
                value = float(timestamp)
                if occurred_from is not None and value < occurred_from:
                    return False
                if occurred_before is not None and value >= occurred_before:
                    return False
            return True

        rows_by_id = {int(row["id"]): row for row in rows if eligible(row)}
        if not rows_by_id:
            return []
        if not query.strip():
            ordered = sorted(
                rows_by_id.values(),
                key=lambda row: (
                    row.get("occurred_at") is None,
                    float(row.get("occurred_at") or 0.0),
                    int(row["id"]),
                ),
            )[:limit]
            return [self.row_item(row, 0.0) for row in ordered]

        has_time_filter = occurred_from is not None or occurred_before is not None
        search_k = self.hybrid.count if has_time_filter else max(
            limit, self.hybrid.candidate_pool
        )
        hits = self.hybrid.search(
            query,
            k=max(1, search_k),
            where={"user_id": user_id} if user_id else None,
        )
        items: list[MemoryItem] = []
        seen: set[int] = set()
        for hit in hits:
            event_id = hit.metadata.get("id")
            if event_id is None:
                continue
            event_id = int(event_id)
            row = rows_by_id.get(event_id)
            if row is None or event_id in seen:
                continue
            seen.add(event_id)
            item = self.row_item(row, hit.score)
            item.metadata["matched_key_kind"] = hit.metadata.get("key_kind")
            item.metadata["matched_source_memory"] = hit.metadata.get(
                "source_memory"
            )
            item.metadata["matched_text"] = hit.text
            items.append(item)
            if len(items) >= limit:
                break
        return items

    def events_by_ids(self, event_ids: list[int]) -> list[MemoryItem]:
        """Return immutable event values in the caller's requested order."""
        output: list[MemoryItem] = []
        for event_id in event_ids:
            row = self.get_row(int(event_id))
            if row is not None:
                output.append(self.row_item(row, self._effective(row)))
        return output

    def event_neighbors(
        self, event_id: int, *, before: int = 2, after: int = 2
    ) -> list[MemoryItem]:
        """Return nearby events in the same chat, ordered chronologically."""
        target = self.get_row(int(event_id))
        if target is None:
            return []
        rows = self.store.select(
            self.table, {"chat_id": target["chat_id"]}, order_by="first_message_id ASC"
        )
        index = next(
            (i for i, row in enumerate(rows) if int(row["id"]) == int(event_id)),
            None,
        )
        if index is None:
            return []
        start = max(0, index - max(0, int(before)))
        stop = min(len(rows), index + max(0, int(after)) + 1)
        return [self.row_item(row, self._effective(row)) for row in rows[start:stop]]

    def index_chunks(self, row: dict[str, Any]) -> list[Chunk]:
        base_meta = self._row_meta(row)
        chunks = [
            Chunk(
                text=row["content"],
                source=self.table,
                metadata={**base_meta, "key_kind": "raw_event"},
            )
        ]
        for key in self.store.select(
            self._KEY_TABLE, {"event_id": row["id"]}, order_by="id ASC"
        ):
            chunks.append(
                Chunk(
                    text=key["key_text"],
                    source=self.table,
                    metadata={
                        **base_meta,
                        "key_kind": key["key_kind"],
                        "source_memory": key["source_memory"],
                        "source_row_id": key["source_row_id"],
                        "confidence": key.get("confidence"),
                    },
                )
            )
        return chunks

    def add_event(
        self,
        user_id: str,
        chat_id: str,
        content: str,
        source_message_ids: list[int],
        *,
        occurred_at: Optional[float] = None,
    ) -> int:
        """Store an event once and return its row ID."""
        ids = [int(message_id) for message_id in source_message_ids]
        if not ids:
            raise ValueError("A conversation event needs at least one source message")
        encoded_ids = json.dumps(ids)
        for row in self.store.select(self.table, {"chat_id": chat_id}):
            if row["source_message_ids"] == encoded_ids:
                return int(row["id"])
        return self.add(
            user_id,
            0.5,
            chat_id=chat_id,
            content=content,
            occurred_at=occurred_at,
            source_message_ids=encoded_ids,
            first_message_id=ids[0],
            last_message_id=ids[-1],
        )

    def ingest_messages(
        self,
        rows: list[dict[str, Any]],
        *,
        default_user_id: str,
        chat_id: str,
        character_name: str,
    ) -> dict[int, int]:
        """Group source rows into events and return message-ID → event-ID."""
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            if row["role"] == "user" and current:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)

        mapping: dict[int, int] = {}
        existing = {
            row["source_message_ids"]: int(row["id"])
            for row in self.store.select(self.table, {"chat_id": chat_id})
        }
        pending: list[tuple[list[int], tuple[str, float, dict[str, Any]]]] = []
        for group in groups:
            user_turn = next((row for row in group if row["role"] == "user"), None)
            event_user = str(
                (user_turn or {}).get("user_id") or default_user_id
            )
            ids = [int(row["id"]) for row in group]
            content = "\n".join(
                self._render_message(
                    row,
                    default_user_id=default_user_id,
                    character_name=character_name,
                )
                for row in group
            )
            occurred_at = next(
                (
                    float(
                        row["occurred_at"]
                        if row.get("occurred_at") is not None
                        else row["created_at"]
                    )
                    for row in group
                    if row.get("occurred_at") is not None
                    or row.get("created_at") is not None
                ),
                None,
            )
            encoded_ids = json.dumps(ids)
            event_id = existing.get(encoded_ids)
            if event_id is not None:
                mapping.update({message_id: event_id for message_id in ids})
                continue
            pending.append(
                (
                    ids,
                    (
                        event_user,
                        0.5,
                        {
                            "chat_id": chat_id,
                            "content": content,
                            "occurred_at": occurred_at,
                            "source_message_ids": encoded_ids,
                            "first_message_id": ids[0],
                            "last_message_id": ids[-1],
                        },
                    ),
                )
            )
        new_ids = self.add_many([entry for _, entry in pending])
        for (message_ids, _), event_id in zip(pending, new_ids):
            mapping.update({message_id: event_id for message_id in message_ids})
        return mapping

    def add_alias(
        self,
        event_id: int,
        key_text: str,
        *,
        key_kind: str,
        source_memory: str,
        source_row_id: int,
        confidence: Optional[float] = None,
    ) -> Optional[int]:
        """Add one compact retrieval key pointing to an existing raw event."""
        text = key_text.strip()
        event = self.get_row(event_id)
        if not text or event is None:
            return None
        existing = self.store.select(self._KEY_TABLE, {"event_id": event_id})
        for key in existing:
            if (
                key["source_memory"] == source_memory
                and int(key["source_row_id"]) == int(source_row_id)
                and key["key_kind"] == key_kind
            ):
                return int(key["id"])
        key_id = self.store.upsert(
            self._KEY_TABLE,
            {
                "event_id": int(event_id),
                "key_text": text,
                "key_kind": key_kind,
                "source_memory": source_memory,
                "source_row_id": int(source_row_id),
                "confidence": confidence,
                "created_at": time.time(),
            },
        )
        if getattr(self.hybrid, "remote", False):
            self.hybrid.refresh()
            return key_id
        self.hybrid.add_documents(
            [
                Chunk(
                    text=text,
                    source=self.table,
                    metadata={
                        **self._row_meta(event),
                        "key_kind": key_kind,
                        "source_memory": source_memory,
                        "source_row_id": int(source_row_id),
                        "confidence": confidence,
                    },
                )
            ]
        )
        return key_id

    def link_extracted(
        self,
        added: dict[str, list[MemoryItem]],
        memories: dict[str, Memory],
    ) -> int:
        """Index extracted facts/directives/episodes as event aliases."""
        event_rows = self.all_rows()
        event_sources = {
            int(row["id"]): set(self._decode_ids(row["source_message_ids"]))
            for row in event_rows
        }
        linked = 0
        for memory_name in ("user_facts", "user_directives", "episodic"):
            memory = memories.get(memory_name)
            if not isinstance(memory, StructuredMemory):
                continue
            for item in added.get(memory_name, []):
                row_id = item.metadata.get("id")
                if row_id is None:
                    continue
                row = memory.get_row(int(row_id))
                if row is None:
                    continue
                source_ids = set(self._decode_ids(row["source_message_ids"]))
                if not source_ids:
                    continue
                key_text = memory.row_text(row)
                confidence = row.get("confidence")
                for event_id, event_ids in event_sources.items():
                    if source_ids.isdisjoint(event_ids):
                        continue
                    if self.add_alias(
                        event_id,
                        key_text,
                        key_kind=memory_name,
                        source_memory=memory_name,
                        source_row_id=int(row_id),
                        confidence=(
                            float(confidence) if confidence is not None else None
                        ),
                    ) is not None:
                        linked += 1
        return linked
