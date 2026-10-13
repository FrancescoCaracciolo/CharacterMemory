"""Persistent group identity, saved-message membership, and rolling profiles."""

from __future__ import annotations

import time
import json
from typing import Any, Optional

from ..config import GroupPrivacy
from .base import ExtractionSpec, MemoryItem, MemoryScope, RecallContext
from .store_base import Store
from .structured import COMMON_COLUMNS, StructuredMemory

_UNSET = object()
GROUP_COLUMNS = {
    "group_id": "TEXT NOT NULL UNIQUE",
    "display_name": "TEXT NOT NULL DEFAULT ''",
    # NULL inherits the character's current configured policy.
    "privacy": "TEXT",
    "summary": "TEXT NOT NULL DEFAULT ''",
    "updated_at": "REAL",
}


class GroupDirectory:
    """Administrative metadata; privacy does not depend on profile enablement."""

    def __init__(self, store: Store, *, default_privacy=GroupPrivacy.MEMBERS):
        self.store = store
        self.default_privacy = GroupPrivacy.coerce(default_privacy)
        self.store.create_table("groups", {**COMMON_COLUMNS, **GROUP_COLUMNS})
        for column, declaration in GROUP_COLUMNS.items():
            if column not in self.store.columns("groups"):
                try:
                    self.store.execute(f"ALTER TABLE groups ADD COLUMN {column} {declaration}")
                except self.store.operational_errors:
                    if column not in self.store.columns("groups"):
                        raise
        self.store.execute("CREATE UNIQUE INDEX IF NOT EXISTS cm_groups_identity ON groups(group_id)")

    @staticmethod
    def _validate(group_id: str) -> None:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a nonempty string")

    def ensure(self, group_id: str, *, display_name=None, privacy=None) -> dict[str, Any]:
        """Atomically create once; never reset another chat's existing metadata."""
        self._validate(group_id)
        override = GroupPrivacy.coerce(privacy).value if privacy is not None else None
        now = time.time()
        self.store.execute(
            "INSERT INTO groups (group_id,user_id,display_name,privacy,summary,importance,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(group_id) DO NOTHING",
            [group_id, "_group", display_name if display_name is not None else group_id,
             override, "", 1.0, now, now],
        )
        return self.get(group_id)

    def _metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "effective_privacy": GroupPrivacy.coerce(
                row["privacy"] if row.get("privacy") is not None else self.default_privacy
            ).value,
            "observed_users": self.observed_users(row["group_id"]),
            "linked_chats": self.linked_chats(row["group_id"]),
        }

    def get(self, group_id: str) -> Optional[dict[str, Any]]:
        rows = self.store.select("groups", {"group_id": group_id}, limit=1)
        return self._metadata(rows[0]) if rows else None

    def list(self) -> list[dict[str, Any]]:
        return [self._metadata(row) for row in self.store.select("groups", order_by="created_at,id")]

    def update(self, group_id: str, *, display_name=_UNSET, privacy=_UNSET) -> dict[str, Any]:
        self._validate(group_id)
        changes = {}
        if display_name is not _UNSET:
            if not isinstance(display_name, str):
                raise ValueError("display_name must be a string")
            changes["display_name"] = display_name
        if privacy is not _UNSET:
            changes["privacy"] = None if privacy is None else GroupPrivacy.coerce(privacy).value
        if self.get(group_id) is None:
            raise KeyError(group_id)
        if changes:
            changes["updated_at"] = time.time()
            self.store.execute(
                "UPDATE groups SET " + ",".join(f"{column}=?" for column in changes) + " WHERE group_id=?",
                [*changes.values(), group_id],
            )
        return self.get(group_id)

    def linked_chats(self, group_id: str) -> list[dict[str, Any]]:
        if "group_id" not in self.store.columns("chats"):
            return []
        return self.store.select("chats", {"group_id": group_id}, order_by="created_at,id")

    def observed_users(self, group_id: str) -> list[str]:
        """Only persisted human speakers establish membership, never owners."""
        if "group_id" not in self.store.columns("chats") or not self.store.columns("messages"):
            return []
        rows = self.store.execute(
            "SELECT DISTINCT m.user_id FROM messages m JOIN chats c ON m.chat_id=c.id "
            "WHERE c.group_id=? AND m.role='user' AND m.user_id IS NOT NULL AND m.user_id<>'' "
            "ORDER BY m.user_id",
            [group_id],
        )
        return [row["user_id"] for row in rows]

    def is_eligible(self, group_id: str, recall_context: RecallContext) -> bool:
        rows = self.store.select("groups", {"group_id": group_id}, limit=1)
        if not rows:
            return False
        policy = GroupPrivacy.coerce(rows[0].get("privacy") or self.default_privacy)
        if policy is GroupPrivacy.NONE:
            return True
        if policy is GroupPrivacy.GROUP_ONLY:
            return recall_context.group_id == group_id
        return bool(recall_context.audience) and set(recall_context.audience).issubset(
            self.observed_users(group_id)
        )

    def eligible_ids(self, recall_context: RecallContext) -> set[str]:
        return {row["group_id"] for row in self.store.select("groups")
                if self.is_eligible(row["group_id"], recall_context)}

    def fingerprint(self) -> tuple:
        """Changes on policy edits or saved-message membership changes."""
        return (
            GroupPrivacy.coerce(self.default_privacy).value,
            tuple((row["group_id"], row.get("privacy"), tuple(self.observed_users(row["group_id"])))
                  for row in self.store.select("groups", order_by="group_id")),
        )


class GroupMemory(StructuredMemory):
    """Rolling group summaries using the shared structured recall/index hooks."""

    name = "groups"
    table = "groups"
    scope = MemoryScope.CHARACTER
    extra_columns = GROUP_COLUMNS
    text_column = "summary"
    supports_recall_context = True

    def __init__(self, store, hybrid, *, group_directory=None, **kwargs):
        directory = group_directory if group_directory is not None else GroupDirectory(store)
        super().__init__(store, hybrid, group_directory=directory, **kwargs)
        self._profile_index_signature = None

    def _signature(self):
        return tuple((row["id"], row["display_name"], row["summary"],
                      tuple(self.group_directory.observed_users(row["group_id"])))
                     for row in self.all_rows())

    def prepare_recall(self) -> None:
        signature = self._signature()
        if signature != self._profile_index_signature:
            self.rebuild_index()
            self._profile_index_signature = signature

    def load(self, path: str) -> None:
        super().load(path)
        self._profile_index_signature = None

    def recall(self, *args, **kwargs):
        self.prepare_recall()
        return super().recall(*args, **kwargs)

    def _sticky_rows(self, rows):
        # The active profile owns the reserved slot; unrelated eligible
        # profiles require a semantic match, not sticky importance.
        return []

    def _priority_rows(self, rows, recall_context):
        return [row for row in rows if recall_context is not None
                and row["group_id"] == recall_context.group_id]

    def row_text(self, row):
        users = ", ".join(self.group_directory.observed_users(row["group_id"]))
        return f"{row['display_name'] or row['group_id']} ({row['group_id']}); users: {users}; {row['summary']}"

    def row_item(self, row, score):
        metadata = dict(row)
        metadata["observed_users"] = self.group_directory.observed_users(row["group_id"])
        return MemoryItem(text=self.row_text(row), score=score, kind=self.name, metadata=metadata)

    def add_or_update(self, group_id: str, summary: str) -> int:
        self.group_directory.ensure(group_id)
        self.store.execute("UPDATE groups SET summary=?,updated_at=? WHERE group_id=?",
                           [summary, self._now(), group_id])
        row = self.store.select("groups", {"group_id": group_id})[0]
        self.apply_index_changes(updated_ids=[row["id"]])
        # Directory rows may have been created without indexing while profiles
        # were disabled. A subsequent recall must inspect all profile rows.
        self._profile_index_signature = None
        return int(row["id"])

    def reset(self) -> None:
        """Clear learned profiles but retain identity and privacy metadata."""
        self.store.execute("UPDATE groups SET summary='',updated_at=?", [self._now()])
        self.rebuild_index()
        self._profile_index_signature = self._signature()

    def extraction_spec(self, context=None):
        if context is None or not getattr(context, "group_id", None):
            return None
        return ExtractionSpec(
            field="group_summary", snapshot=True,
            schema={"type": "object", "properties": {"summary": {"type": "string"}},
                    "required": ["summary"]},
            instruction=(
                "- group_summary: a compact rolling profile of this persistent group: "
                "established shared topics, interests, context, dynamics and recurring activities. "
                "Replace the previous summary while retaining supported durable context and applying "
                "explicit corrections. Do not invent members, identities, names or relationships. "
                "Membership and the display label are maintained by the application. Exclude "
                "temporary moods and detailed event recaps. Return an empty summary if unchanged."
                "\nPrevious stored summary (context, not new evidence): "
                + str(getattr(context, "group_summary", "") or "")
                + "\nAuthoritative saved human speakers: "
                + json.dumps(list(getattr(context, "group_users", ()) or ()), ensure_ascii=False)
            ),
        )

    def apply_extraction_with_context(self, value, context, *, chat_id=None):
        group_id = getattr(context, "group_id", None)
        if not group_id or not isinstance(value, dict):
            return []
        summary = str(value.get("summary") or "").strip()
        if not summary:
            return []
        existing = self.group_directory.get(group_id)
        if existing is not None and existing["summary"] == summary:
            return []
        row_id = self.add_or_update(group_id, summary)
        row = self.get_row(row_id)
        return [self.row_item(row, self._effective(row))]
