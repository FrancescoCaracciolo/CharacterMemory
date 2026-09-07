"""User summary: a consolidated per-user profile (name, aliases, quick summary)."""

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Optional

from .base import ExtractionSpec, MemoryItem
from .structured import StructuredMemory

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext


class UserSummaryMemory(StructuredMemory):
    """A single rolling profile per user.

    Each user owns exactly one row holding their `name`, every `aliases`
    (nicknames / other names they go by, stored as a JSON list) and a `summary`
    - a quick, self-contained description of who they are. Extraction refreshes
    the row instead of appending, so the profile stays consolidated. The summary
    is always injected (sticky) whenever the user is present.
    """

    name = "user_summary"
    table = "user_summary"
    extra_columns = {
        "name": "TEXT NOT NULL",
        "aliases": "TEXT NOT NULL DEFAULT '[]'",
        "summary": "TEXT NOT NULL",
    }
    text_column = "summary"

    # The single per-user profile should always be surfaced; default to a sticky
    # base importance so :meth:`StructuredMemory.recall` injects it regardless
    # of the query.
    default_importance: float = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the memory and migrate the profile refresh timestamp.

        ``updated_at`` is intentionally not part of ``extra_columns``: it is
        managed by this memory, not exposed as an editor field. Existing
        databases get the column lazily when the character is loaded.
        """
        super().__init__(*args, **kwargs)
        if "updated_at" not in self.store.columns(self.table):
            try:
                self.store.execute(
                    f"ALTER TABLE {self.table} ADD COLUMN updated_at REAL"
                )
            except sqlite3.OperationalError:
                # Another process may have completed the same additive
                # migration between the column check and ALTER TABLE.
                if "updated_at" not in self.store.columns(self.table):
                    raise
        self.store.execute(
            f"UPDATE {self.table} SET updated_at = created_at "
            "WHERE updated_at IS NULL"
        )

    @staticmethod
    def _parse_aliases(raw: Any) -> list[str]:
        """Normalise stored aliases (JSON list / list / comma string) to a list."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(a).strip() for a in raw if str(a).strip()]
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return [str(a).strip() for a in data if str(a).strip()]
            except (ValueError, TypeError):
                pass
            return [a.strip() for a in raw.split(",") if a.strip()]
        return []

    def add_or_update(
        self,
        user_id: str,
        name: str,
        aliases: Any,
        summary: str,
        *,
        importance: float = default_importance,
    ) -> int:
        """Insert the user's profile, or merge into the existing one.

        On update, new `aliases` are unioned with the stored ones (deduped) and
        `name`/`summary` are overwritten. Only the affected profile's hybrid
        entry is replaced.
        """
        aliases = self._parse_aliases(aliases)
        existing = self.store.select(
            self.table, {"user_id": user_id}, order_by="id DESC", limit=1
        )
        if existing:
            row = existing[0]
            merged = sorted(
                set(self._parse_aliases(row.get("aliases"))) | set(aliases)
            )
            self.update_row(
                {
                    "id": row["id"],
                    "user_id": user_id,
                    "importance": float(importance),
                    "created_at": row.get("created_at"),
                    "last_recalled": row.get("last_recalled"),
                    "recall_count": row.get("recall_count", 0),
                    "name": name or row.get("name") or user_id,
                    "aliases": json.dumps(merged, ensure_ascii=False),
                    "summary": summary,
                }
            )
            row_id = int(row["id"])
            self.apply_index_changes(updated_ids=[row_id])
        else:
            row_id = self.add(
                user_id,
                importance,
                name=name or user_id,
                aliases=json.dumps(aliases, ensure_ascii=False),
                summary=summary,
            )
        return row_id

    def add(self, user_id: str, importance: float, **fields: Any) -> int:
        """Insert a profile row and stamp its initial refresh time."""
        fields["updated_at"] = self._now()
        return super().add(user_id, importance, **fields)

    def update_row(self, row: dict[str, Any]) -> None:
        """Write a profile row and stamp the refresh time."""
        updated = dict(row)
        updated["updated_at"] = self._now()
        super().update_row(updated)

    def get_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        """Return the stored profile for `user_id`, or `None`."""
        rows = self.store.select(
            self.table, {"user_id": user_id}, order_by="id DESC", limit=1
        )
        return rows[0] if rows else None

    def row_text(self, row: dict[str, Any]) -> str:
        aliases = ", ".join(self._parse_aliases(row.get("aliases")))
        parts = [f"name: {row.get('name', '')}"]
        if aliases:
            parts.append(f"aliases: {aliases}")
        parts.append(f"summary: {row.get('summary', '')}")
        return ". ".join(parts)

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        aliases = ", ".join(self._parse_aliases(row.get("aliases")))
        text = f"{row.get('name', '')}"
        if aliases:
            text += f" (aka {aliases})"
        text += f": {row.get('summary', '')}"
        return MemoryItem(text=text, score=score, kind=self.name, metadata=dict(row))

    # Extraction ----------------------------------------------------------
    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        user = context.user_name if context else "the user"
        users = (
            list(context.participants)
            if context is not None and context.participants
            else [user]
        )
        baselines: list[str] = []
        for uid in users:
            row = self.get_summary(uid)
            if row is None:
                continue
            aliases = ", ".join(self._parse_aliases(row.get("aliases"))) or "(none)"
            baselines.append(
                f"Current stored profile for {uid}: name={row.get('name', uid)}; "
                f"aliases={aliases}; summary={row.get('summary', '')}"
            )
        baseline_note = ""
        if baselines:
            baseline_note = (
                "\nCurrent stored profile(s) to selectively rewrite under these rules:\n"
                + "\n".join(f"- {line}" for line in baselines)
            )
        return ExtractionSpec(
            field="user_summaries",
            per_user=True,
            snapshot=True,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                    },
                    "required": ["name", "aliases", "summary"],
                },
            },
            instruction=(
                "- user_summaries: include `name`, established `aliases`, and a "
                "third-person `summary` of each user. Aim for 100–150 words at most; "
                "use fewer when little is known. Retain supported identity, enduring "
                "interests and goals, communication preferences, explicit interpersonal "
                "boundaries, and established relationship dynamics. Core context may "
                "overlap with known facts or directives; leave detailed facts and "
                "individual events to their respective memories. Exclude chronological "
                "recaps, exam logistics, meals, temporary moods, isolated jokes, "
                "lesson details, and narrated character reactions. Do not infer "
                "personality from one incident or treat speculation, teasing, or "
                "role-play as real-world identity, aliases, or evidence that users "
                "are the same person. Rewrite stored profiles as complete compact "
                "replacements: retain supported core information, apply corrections, "
                "and remove irrelevant detail even if uncontradicted. Compact bloated "
                "profiles even without new durable information. Return [] if the "
                "profile already meets these rules and nothing material changed, "
                "or there is insufficient information to create one."
                f"{baseline_note}"
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        added: list[MemoryItem] = []
        for item in value or []:
            uid = str(item.get("user_id") or user_id)
            name = (item.get("name") or "").strip() or uid
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            row_id = self.add_or_update(
                uid, name, item.get("aliases") or [], summary
            )
            row = self.get_row(row_id)
            if row is not None:
                added.append(self.row_item(row, self._effective(row)))
        return added
