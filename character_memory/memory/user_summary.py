"""User summary: a consolidated per-user profile (name, aliases, quick summary)."""

import json
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
        `name`/`summary` are overwritten. The per-user hybrid index is rebuilt so
        it stays in sync with the single row.
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
        else:
            row_id = self.add(
                user_id,
                importance,
                name=name or user_id,
                aliases=json.dumps(aliases, ensure_ascii=False),
                summary=summary,
            )
        self.rebuild_index()
        return row_id

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
        return ExtractionSpec(
            field="user_summaries",
            per_user=True,
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
                f"- user_summaries: for each distinct person, a single consolidated "
                f"profile with their `name`, every `aliases` (nicknames / other names "
                f"they go by), and a `summary`: a quick, self-contained description of "
                f"who they are (interests, role, personality, key facts). Produce one "
                f"item per person, written from {user}'s perspective using the real "
                f"name."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str) -> list[MemoryItem]:
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
