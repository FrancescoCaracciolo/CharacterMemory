"""Persistent chat / conversation storage.

A `Chat` is a handle over rows in the shared `SQLiteStore` (the same
`memory.db` that holds the structured memories). The agent owns the store
and hands it to every `Chat`; the chat classes never open a connection of
their own.

Two tables (additive — learned memories are untouched):

* `chats`    - one row per conversation (`id, user_id, title, created_at`)
* `messages` - one row per turn         (`id, chat_id, role, content,
                created_at, extracted`)

The `extracted` flag on messages lets :meth:`CharacterAgent.extract`
process only the messages that have not fed the extractor yet, so extraction
is idempotent and resumable.
"""

import time
import uuid
from typing import Any, Optional

from .memory.store import SQLiteStore

_CHAT_COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "user_id": "TEXT NOT NULL",
    "title": "TEXT NOT NULL DEFAULT ''",
    "created_at": "REAL NOT NULL",
}

_MESSAGE_COLUMNS: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "chat_id": "TEXT NOT NULL",
    "role": "TEXT NOT NULL",
    "content": "TEXT NOT NULL",
    "created_at": "REAL NOT NULL",
    "extracted": "INTEGER NOT NULL DEFAULT 0",
}


class Chat:
    """A single conversation. Reads/writes through the shared `SQLiteStore`."""

    def __init__(
        self,
        chat_id: str,
        user_id: str,
        store: SQLiteStore,
        *,
        title: str = "",
        created_at: Optional[float] = None,
    ) -> None:
        self.id = chat_id
        self.user_id = user_id
        self.store = store
        self.title = title
        self.created_at = created_at if created_at is not None else time.time()

    # ------------------------------------------------------------- messages
    def add_message(
        self, role: str, content: str, *, extracted: bool = False
    ) -> dict[str, Any]:
        """Persist one message and return it as an openai-style dict."""
        row = {
            "chat_id": self.id,
            "role": role,
            "content": content,
            "created_at": time.time(),
            "extracted": 1 if extracted else 0,
        }
        new_id = self.store.upsert("messages", row, pk="id")
        row["id"] = new_id
        return {"role": role, "content": content}

    def messages(self) -> list[dict[str, str]]:
        """All messages, oldest first, as `{role, content}` dicts."""
        rows = self.store.select(
            "messages",
            where={"chat_id": self.id},
            order_by="id ASC",
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # Backwards-friendly alias.
    history = messages

    def last_user_message(self) -> Optional[str]:
        """The most recent `user` message in this chat, or `None`."""
        rows = self.store.select(
            "messages",
            where={"chat_id": self.id, "role": "user"},
            order_by="id DESC",
            limit=1,
        )
        return rows[0]["content"] if rows else None

    def mark_extracted(self, message_ids: list[int]) -> None:
        """Flag the given message ids as already extracted."""
        if not message_ids:
            return
        qs = ", ".join("?" for _ in message_ids)
        self.store.execute(
            f"UPDATE messages SET extracted=1 WHERE id IN ({qs})",
            list(message_ids),
        )

    def unextracted(self) -> list[dict[str, Any]]:
        """Rows not yet extracted, oldest first."""
        return self.store.select(
            "messages",
            where={"chat_id": self.id, "extracted": 0},
            order_by="id ASC",
        )

    def __len__(self) -> int:
        rows = self.store.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id=?",
            [self.id],
        )
        return int(rows[0]["n"]) if rows else 0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Chat(id={self.id!r}, user_id={self.user_id!r}, title={self.title!r})"


class _ChatBackend:
    """CRUD for the `chats` / `messages` tables. Owned by the agent."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self.store.create_table("chats", _CHAT_COLUMNS, pk="id")
        self.store.create_table("messages", _MESSAGE_COLUMNS, pk="id")

    def create_chat(self, user_id: str, *, title: str = "") -> Chat:
        chat_id = uuid.uuid4().hex
        now = time.time()
        self.store.upsert(
            "chats",
            {
                "id": chat_id,
                "user_id": user_id,
                "title": title,
                "created_at": now,
            },
            pk="id",
        )
        return Chat(chat_id, user_id, self.store, title=title, created_at=now)

    def load_chat(self, chat_id: str) -> Optional[Chat]:
        rows = self.store.select("chats", where={"id": chat_id}, limit=1)
        if not rows:
            return None
        r = rows[0]
        return Chat(
            r["id"], r["user_id"], self.store, title=r["title"], created_at=r["created_at"]
        )

    def list_chats(self, user_id: Optional[str] = None) -> list[Chat]:
        if user_id is None:
            rows = self.store.select("chats", order_by="created_at ASC")
        else:
            rows = self.store.select(
                "chats", where={"user_id": user_id}, order_by="created_at ASC"
            )
        return [
            Chat(r["id"], r["user_id"], self.store, title=r["title"], created_at=r["created_at"])
            for r in rows
        ]

    def all_unextracted(self) -> list[dict[str, Any]]:
        """Every un-extracted message row across all chats, oldest first."""
        return self.store.select(
            "messages", where={"extracted": 0}, order_by="chat_id ASC, id ASC"
        )

    def chat_user(self, chat_id: str) -> Optional[str]:
        rows = self.store.select("chats", where={"id": chat_id}, limit=1)
        return rows[0]["user_id"] if rows else None
