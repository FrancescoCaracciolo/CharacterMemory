"""Undated concerns, intentions and promises that remain open until resolved."""

from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Any, Optional

from .base import ExtractionSpec, MemoryItem
from .continuity import ContinuityMemory, SOURCE_IDS_SCHEMA
from .extract import ExtractionContext


class ProspectiveMemory(ContinuityMemory):
    """Per-user open threads with semantic recall and periodic follow-up cues.

    Recall never completes a thread. Resolved/dismissed rows remain in storage
    for inspection and explicit reopening, but leave automatic prompt recall.
    """

    name = "prospective"
    table = "prospective"
    kinds = ("concern", "promise", "intention", "question")
    extra_columns = {
        **ContinuityMemory.continuity_columns,
        "kind": "TEXT NOT NULL DEFAULT 'concern'",
        "status": "TEXT NOT NULL DEFAULT 'open'",
        "follow_up": "TEXT NOT NULL DEFAULT ''",
        "not_before": "REAL",
        "resolution": "TEXT NOT NULL DEFAULT ''",
        "resolved_at": "REAL",
    }

    def __init__(self, *args, follow_up_interval: float = 86_400, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(follow_up_interval) or follow_up_interval < 0:
            raise ValueError("follow_up_interval must be finite and nonnegative")
        self.follow_up_interval = follow_up_interval

    @staticmethod
    def _timestamp(value):
        if value in (None, ""):
            return None
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("not_before needs an explicit timezone")
            value = parsed.timestamp()
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("not_before must be finite")
        return result

    def add_thread(
        self, user_id: str, content: str, *, kind: str = "concern", follow_up: str = "",
        not_before=None, importance: float = 0.7, chat_id: Optional[str] = None,
        source_message_ids: Optional[list[int]] = None,
    ) -> int:
        if not content.strip() or kind not in self.kinds:
            raise ValueError("A thread needs content and a supported kind")
        return self.add(
            user_id, self._clip(importance), content=content.strip(), kind=kind,
            follow_up=follow_up.strip(), not_before=self._timestamp(not_before),
            chat_id=chat_id, source_message_ids=json.dumps(source_message_ids or []),
        )

    def _recall_where(self, user_id):
        return {"user_id": user_id, "status": "open"}

    def _row_meta(self, row):
        return {**super()._row_meta(row), "status": row["status"]}

    def _sticky_rows(self, rows):
        return []

    @staticmethod
    def _hit_relevance(hits):
        # Some indexes return nearest neighbours even with zero relevance.
        # They must not bypass timing/cooldown by becoming query candidates.
        return {rid: score for rid, score in ContinuityMemory._hit_relevance(hits).items()
                if score > 0.0}

    def _effective(self, row):
        # An unanswered promise does not expire merely because it is old.
        return self._intrinsic(row) if row["status"] == "open" else super()._effective(row)

    def _additional_relevance(self, query, rows_by_id):
        now = self._now()
        return {
            rid: 0.7 for rid, row in rows_by_id.items()
            if (row["not_before"] is None or float(row["not_before"]) <= now)
            and (row["last_recalled"] is None
                 or now - float(row["last_recalled"]) >= self.follow_up_interval)
        }

    def row_text(self, row):
        return f"{row['content']} {row['follow_up']}".strip()

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        text = f"{row['content']} ({row['kind']}; {row['status']})"
        if row["follow_up"]:
            text += f" Follow-up cue: {row['follow_up']}"
        if row["not_before"] is not None:
            from datetime import UTC
            text += f" Follow up no earlier than {datetime.fromtimestamp(row['not_before'], UTC).isoformat()}."
        return MemoryItem(text=text, score=score, kind=self.name, metadata=dict(row))

    def format(self, items):
        if not items:
            return ""
        return (
            "These matters are still unresolved. When natural, follow up on one or honor the promise. "
            "Do not assume an outcome or repeat a question already answered in the conversation. "
            "A recall is not evidence of completion.\n" + super().format(items)
        )

    def format_grouped(self, items, participants):
        return (
            "Unresolved matters: follow up naturally without assuming outcomes or repeating answered questions.\n"
            + super().format_grouped(items, participants)
        ) if items else ""

    def extraction_spec(self, context: Optional[ExtractionContext] = None) -> ExtractionSpec:
        people = self.people(context) if context else []
        existing = [{key: row[key] for key in (
            "id", "user_id", "content", "status", "follow_up", "resolution"
        )} for row in self.all_rows() if row["user_id"] in people]
        return ExtractionSpec(
            field="open_threads", per_user=True,
            schema={"type": "array", "items": {
                "type": "object", "properties": {
                    "operation": {"type": "string", "enum": ["create", "update", "resolve", "dismiss", "reopen"]},
                    "thread_id": {"type": "integer"},
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": list(self.kinds)},
                    "follow_up": {"type": "string"},
                    "not_before": {"type": ["string", "null"]},
                    "importance": {"type": "number"},
                    "resolution": {"type": "string"},
                    "source_message_ids": SOURCE_IDS_SCHEMA,
                }, "required": ["operation", "source_message_ids"],
            }},
            instruction=(
                "- open_threads: unresolved concerns, intentions, unanswered questions or promises worth "
                "revisiting with a participant. No date is required. Examples: Francesco is nervous about "
                "his mother's surgery (ask how it went, without assuming it happened); the character "
                "promised to explain X next time (offer that explanation at a suitable opportunity). "
                "Create with content, kind, a natural-language follow_up cue and importance (0-1). "
                "For character promises user_id is the beneficiary. Preserve vague timing in follow_up; "
                "not_before is null unless an explicit, grounded time can be expressed as ISO with timezone. "
                "Do not invent dates, commitments or outcomes. Calendar commitments belong in calendar_updates "
                "when enabled; record a separate thread only if there is something to follow up. "
                "Use the existing thread_id to update, resolve (explicit completion/outcome), dismiss "
                "(explicit cancellation or request not to ask), or reopen (explicit renewed concern). "
                "Resolve/dismiss require a resolution sentence. Mere mention, asking about progress, "
                "elapsed time or recalling a thread does not resolve it. Do not recreate closed threads "
                "or duplicate paraphrases. Stored threads below are context, not new evidence:\n"
                + json.dumps(existing, ensure_ascii=False)
            ),
        )

    def _apply_update(self, update, context, chat_id, source_ids):
        uid = str(update.get("user_id") or context.user_name)
        if uid not in self.people(context):
            return None
        operation = update.get("operation")
        content = str(update.get("content") or "").strip()
        if operation == "create":
            if not content or self._has_text(uid, content):
                return None
            return self.add_thread(
                uid, content, kind=update.get("kind", "concern"),
                follow_up=str(update.get("follow_up") or ""), not_before=update.get("not_before"),
                importance=update.get("importance", 0.7), chat_id=chat_id,
                source_message_ids=source_ids,
            )
        if operation not in {"update", "resolve", "dismiss", "reopen"}:
            return None
        row = self.get_row(int(update["thread_id"]))
        if row is None or row["user_id"] != uid:
            return None
        changes = {}
        if operation in {"resolve", "dismiss"}:
            resolution = str(update.get("resolution") or "").strip()
            if not resolution or row["status"] != "open":
                return None
            changes.update(status="resolved" if operation == "resolve" else "dismissed",
                           resolution=resolution, resolved_at=self._now())
        elif operation == "reopen":
            changes.update(status="open", resolution="", resolved_at=None, last_recalled=None)
        elif row["status"] != "open":
            return None
        if content:
            changes["content"] = content
        if "follow_up" in update:
            changes["follow_up"] = str(update["follow_up"] or "").strip()
        if "not_before" in update:
            changes["not_before"] = self._timestamp(update["not_before"])
        return self._save_update(row, source_ids, **changes)
