"""Learned character statements and the audience of each disclosure."""

from __future__ import annotations

import json
from typing import Any, Optional

from .base import ExtractionSpec, MemoryItem, MemoryScope
from .continuity import ContinuityMemory, SOURCE_IDS_SCHEMA
from .extract import ExtractionContext


class CharacterSelfMemory(ContinuityMemory):
    """Dynamic self-continuity alongside immutable character-info lore.

    Public statements recall across users. Private statements recall only when
    every current participant is in ``disclosed_to``. That list records who
    actually heard a statement, independently of whether it may be shared.
    """

    name = "character_self"
    table = "character_self"
    scope = MemoryScope.CHARACTER
    evidence_roles = frozenset({"assistant"})
    kinds = ("backstory", "opinion", "preference", "promise", "relationship", "other")
    extra_columns = {
        **ContinuityMemory.continuity_columns,
        "kind": "TEXT NOT NULL DEFAULT 'other'",
        "visibility": "TEXT NOT NULL DEFAULT 'private'",
        "disclosed_to": "TEXT NOT NULL DEFAULT '[]'",
    }

    def add_statement(
        self, content: str, *, kind: str = "other", visibility: str = "private",
        disclosed_to: Optional[list[str]] = None, importance: float = 0.7,
        chat_id: Optional[str] = None, source_message_ids: Optional[list[int]] = None,
    ) -> int:
        if not content.strip() or kind not in self.kinds or visibility not in {"public", "private"}:
            raise ValueError("A statement needs content, a supported kind and visibility")
        return self.add(
            "_self", self._clip(importance), content=content.strip(), kind=kind,
            visibility=visibility, disclosed_to=json.dumps(sorted(set(disclosed_to or []))),
            chat_id=chat_id, source_message_ids=json.dumps(source_message_ids or []),
        )

    def _recall_where(self, user_id: str) -> None:
        return None

    def _filter_recall_rows(self, rows, audience):
        return [row for row in rows if row["visibility"] == "public"
                or set(audience).issubset(json.loads(row["disclosed_to"]))]

    def recall_participants(self, query, participants, limit, state_changing=True, **kwargs):
        if not participants:
            return []
        if kwargs.get("temporal_weight") is None:
            kwargs["temporal_weight"] = getattr(self, "temporal_resolution_weight", 1.0)
        return self.recall(
            query, participants[0], limit, state_changing=state_changing,
            audience=participants, **kwargs,
        )

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        people = ", ".join(json.loads(row["disclosed_to"])) or "nobody recorded"
        return MemoryItem(
            text=f"{row['content']} ({row['kind']}; {row['visibility']}; told to: {people})",
            score=score, kind=self.name, metadata=dict(row),
        )

    def format(self, items: list[MemoryItem]) -> str:
        if not items:
            return ""
        return (
            "Keep your own established statements consistent; static character lore takes precedence. "
            "Do not assume someone already knows a detail unless they are listed as told. "
            "Keep private statements within their recorded audience.\n" + super().format(items)
        )

    def extraction_spec(self, context: Optional[ExtractionContext] = None) -> ExtractionSpec:
        character = context.character_name if context else "the character"
        existing = [{key: row[key] for key in (
            "id", "content", "kind", "visibility", "disclosed_to"
        )} for row in self.all_rows()]
        return ExtractionSpec(
            field="character_statements",
            schema={"type": "array", "items": {
                "type": "object", "properties": {
                    "operation": {"type": "string", "enum": ["create", "revise", "disclose"]},
                    "statement_id": {"type": "integer"},
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": list(self.kinds)},
                    "visibility": {"type": "string", "enum": ["public", "private"]},
                    "importance": {"type": "number"},
                    "source_message_ids": SOURCE_IDS_SCHEMA,
                }, "required": ["operation", "source_message_ids"],
            }},
            instruction=(
                f"- character_statements: durable backstory details, opinions, preferences, relationships "
                f"or promises that {character} actually asserts about itself in ASSISTANT messages. "
                "Do not turn user claims, quotations, jokes, hypotheticals or role-play examples into self facts. "
                "Cite the assistant message that establishes each statement. Create with content, kind, "
                "visibility and importance (0-1). Use private for secrets/confidences or uncertain sharing "
                "permission; public means an ordinary detail that may be shared, not that everyone knows it. "
                "The library records all present participants as the disclosure audience. "
                "Use disclose with an existing statement_id when the character repeats that statement to "
                "this audience. Use revise only for an explicit correction or change of mind, never silently "
                "replace an established detail with a contradiction. Promises also belong in open_threads "
                "when that field is enabled. Phrase promises historically ('promised to ...'), not as "
                "perpetually pending obligations; open_threads tracks their completion. "
                "Stored statements below are context, not new evidence:\n"
                + json.dumps(existing, ensure_ascii=False)
            ),
        )

    def _apply_update(self, update, context, chat_id, source_ids):
        operation = update.get("operation")
        people = self.people(context)
        content = str(update.get("content") or "").strip()
        if operation == "create":
            if not content:
                return None
            # Re-extracting an assertion must still remember its new audience.
            row = next((row for row in self.all_rows()
                        if row["content"].strip().casefold() == content.casefold()), None)
            if row is None:
                return self.add_statement(
                    content, kind=update.get("kind", "other"),
                    visibility=update.get("visibility", "private"), disclosed_to=people,
                    importance=update.get("importance", 0.7), chat_id=chat_id,
                    source_message_ids=source_ids,
                )
        elif operation in {"revise", "disclose"}:
            row = self.get_row(int(update["statement_id"]))
            if row is None:
                return None
        else:
            return None
        changes = {"disclosed_to": json.dumps(sorted(set(json.loads(row["disclosed_to"])) | set(people)))}
        if operation == "revise":
            if not content:
                return None
            changes["content"] = content
            # A revision of a private assertion has a new disclosure audience;
            # hearing the old version does not establish knowing the new one.
            changes["disclosed_to"] = json.dumps(sorted(set(people)))
        # Extraction never automatically makes an existing secret public.
        if update.get("visibility") == "private":
            changes["visibility"] = "private"
        return self._save_update(row, source_ids, **changes)
