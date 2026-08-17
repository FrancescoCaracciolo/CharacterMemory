"""Model-facing calendar tools.

Search is read-only.  Mutations are returned as deferred ``TurnEffect``
objects so the agent commits them together with the final assistant message.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .base import Tool, ToolOutput, TurnEffect

if TYPE_CHECKING:
    from ..agent import CharacterAgent
    from ..memory.calendar import CalendarMemory


# Keep this module importable while the package is wiring the RAG/LLM modules.
# The concrete CalendarMemory import is intentionally lazy (see _calendar),
# just like the agent type import above.
SELF_OWNER = "_self"


def _calendar(agent: "CharacterAgent") -> CalendarMemory:
    from ..memory.calendar import CalendarMemory

    memory = getattr(agent, "calendar_memory", None)
    if memory is None:
        memory = (getattr(agent, "memories", {}) or {}).get("calendar")
    if not isinstance(memory, CalendarMemory) or not memory.enabled:
        raise RuntimeError("calendar memory is not available")
    return memory


def _owners(value: Optional[Iterable[str]]) -> Optional[set[str]]:
    if value is None:
        return None
    return {str(v) for v in value if str(v).strip()}


class _CalendarTool(Tool):
    requires_persisted_chat = False

    def __init__(self, agent: "CharacterAgent", allowed_owners: Optional[set[str]] = None) -> None:
        self.agent = agent
        self.allowed_owners = allowed_owners
        # Advertise the same scope the runtime validator enforces.  Copy the
        # class schema because tool instances are intentionally fresh per
        # agent/chat and class-level schemas must remain reusable.
        self.parameters = deepcopy(type(self).parameters)
        if allowed_owners is not None:
            properties = self.parameters.setdefault("properties", {})
            if "owner_id" in properties:
                properties["owner_id"] = {**properties["owner_id"], "enum": sorted(allowed_owners)}
            if "owner_ids" in properties:
                properties["owner_ids"] = {
                    **properties["owner_ids"],
                    "items": {**properties["owner_ids"].get("items", {}), "enum": sorted(allowed_owners)},
                }

    @property
    def calendar(self) -> CalendarMemory:
        return _calendar(self.agent)

    def _check_owner(self, owner_id: str) -> str:
        owner = str(owner_id or "").strip()
        if not owner:
            raise ValueError("owner_id is required")
        if self.allowed_owners is not None and owner not in self.allowed_owners:
            raise ValueError(f"owner_id {owner!r} is outside this chat's calendar scope")
        return owner

    def _check_row_scope(self, row: dict[str, Any]) -> str:
        """Validate scope and return the row's authoritative owner.

        A shared event has one persisted owner row. An attendee in the current
        chat may request an edit/cancellation, but the deferred transaction
        must still carry the authoritative owner so it cannot be mistaken for
        a different row owner.
        """
        owner = str(row.get("user_id") or SELF_OWNER)
        if self.allowed_owners is None or owner in self.allowed_owners:
            return owner
        attendees = row.get("attendees") or []
        if isinstance(attendees, str):
            try:
                import json
                attendees = json.loads(attendees)
            except (TypeError, ValueError):
                attendees = []
        for attendee in attendees if isinstance(attendees, (list, tuple)) else []:
            candidate = str(attendee)
            if candidate in self.allowed_owners:
                return owner
        raise ValueError(f"calendar event owner {owner!r} is outside this chat's calendar scope")

    def _check_attendees(self, values: dict[str, Any]) -> None:
        """Keep model-created shared events inside the chat's owner scope."""
        if self.allowed_owners is None or "attendees" not in values:
            return
        attendees = values.get("attendees")
        if isinstance(attendees, str):
            attendees = [part.strip() for part in attendees.split(",") if part.strip()]
        if not isinstance(attendees, (list, tuple, set)):
            return  # _clean_event supplies the precise validation message.
        for attendee in attendees:
            self._check_owner(str(attendee))


class SearchCalendarEvents(_CalendarTool):
    name = "search_calendar_events"
    description = (
        "Search the character and users' calendar occurrences. Results include "
        "one-off events, weekly routines, attendees, local times, and whether "
        "an occurrence is a read-only world routine."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "default": ""},
            "start": {"type": "string", "description": "ISO lower bound (inclusive)."},
            "before": {"type": "string", "description": "ISO upper bound (exclusive)."},
            "owner_ids": {"type": "array", "items": {"type": "string"}},
            "include_cancelled": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(
        self,
        query: str = "",
        start: Optional[str] = None,
        before: Optional[str] = None,
        owner_ids: Optional[list[str]] = None,
        include_cancelled: bool = False,
        limit: int = 10,
    ) -> dict[str, Any]:
        from ..memory.calendar import _parse_iso

        zone = self.calendar.default_timezone
        requested_owners = _owners(owner_ids)
        if self.allowed_owners is not None:
            if requested_owners is None:
                requested_owners = set(self.allowed_owners)
            elif not requested_owners.issubset(self.allowed_owners):
                raise ValueError("owner_ids contains a calendar outside this chat's scope")
        items = self.calendar.search_events(
            query or "",
            start=_parse_iso(start, timezone=zone) if start else None,
            before=_parse_iso(before, timezone=zone) if before else None,
            owners=requested_owners,
            include_cancelled=bool(include_cancelled),
            limit=max(1, min(50, int(limit))),
            state_changing=False,
        )
        return {"count": len(items), "events": [item.to_dict() for item in items]}


class CreateCalendarEvent(_CalendarTool):
    name = "create_calendar_event"
    description = (
        "Stage a calendar event or weekly routine for the character or a user. "
        "The change is committed only when the final assistant reply is saved."
    )
    requires_persisted_chat = True
    parameters = {
        "type": "object",
        "properties": {
            "owner_id": {"type": "string"}, "title": {"type": "string"},
            "description": {"type": "string"}, "location": {"type": "string"},
            "timezone": {"type": "string"},
            "kind": {"type": "string", "enum": ["event", "routine"], "default": "event"},
            "start_at": {"type": "string"}, "end_at": {"type": "string"},
            "weekdays": {"type": "array", "items": {"type": "integer"}},
            "start_local": {"type": "string"}, "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["owner_id", "title"],
        "additionalProperties": False,
    }

    def run(self, owner_id: str, title: str, **values: Any) -> ToolOutput:
        owner = self._check_owner(owner_id)
        self._check_attendees(values)
        self.calendar._clean_event({"title": title, **values})
        payload = {"owner_id": owner, "source": "model_tool", "values": {"title": title, **values}}
        return ToolOutput(
            text="Calendar event validated and staged for this reply.",
            data=payload,
            effects=[TurnEffect("calendar_create", payload)],
        )


class UpdateCalendarEvent(_CalendarTool):
    name = "update_calendar_event"
    description = (
        "Stage a partial update to a persisted calendar event or routine. "
        "World routine IDs are read-only and must be changed in World settings."
    )
    requires_persisted_chat = True
    parameters = {
        "type": "object",
        "properties": {
            "event_id": {"type": "integer"}, "title": {"type": "string"},
            "description": {"type": "string"}, "location": {"type": "string"},
            "timezone": {"type": "string"}, "kind": {"type": "string", "enum": ["event", "routine"]},
            "start_at": {"type": "string"}, "end_at": {"type": "string"},
            "weekdays": {"type": "array", "items": {"type": "integer"}},
            "start_local": {"type": "string"}, "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["event_id"], "additionalProperties": False,
    }

    def run(self, event_id: int, **values: Any) -> ToolOutput:
        row = self.calendar.get_row(int(event_id))
        if row is None:
            raise KeyError(event_id)
        owner = self._check_row_scope(row)
        self._check_attendees(values)
        self.calendar.prepare_update(int(event_id), values)
        payload = {"event_id": int(event_id), "owner_id": owner, "values": values, "source": "model_tool"}
        return ToolOutput("Calendar update validated and staged for this reply.", data=payload, effects=[TurnEffect("calendar_update", payload)])


class CancelCalendarEvent(_CalendarTool):
    name = "cancel_calendar_event"
    description = "Stage cancellation of a persisted calendar event or routine."
    requires_persisted_chat = True
    parameters = {
        "type": "object",
        "properties": {"event_id": {"type": "integer"}},
        "required": ["event_id"], "additionalProperties": False,
    }

    def run(self, event_id: int) -> ToolOutput:
        row = self.calendar.get_row(int(event_id))
        if row is None:
            raise KeyError(event_id)
        owner = self._check_row_scope(row)
        payload = {"event_id": int(event_id), "owner_id": owner, "source": "model_tool"}
        return ToolOutput("Calendar cancellation validated and staged for this reply.", data=payload, effects=[TurnEffect("calendar_cancel", payload)])


# "Edit" is a common integration spelling; keep it as an API-compatible
# alias while the wire/tool name remains the more explicit update form.
EditCalendarEvent = UpdateCalendarEvent


def calendar_tools(
    agent: "CharacterAgent",
    *,
    owner_ids: Optional[Iterable[str]] = None,
    include_writes: bool = True,
) -> list[Tool]:
    """Build fresh calendar tools bound to ``agent``.

    ``owner_ids`` should be the participants of the persisted chat.  The
    character's own calendar is always included as ``_self``.
    """
    memory = getattr(agent, "calendar_memory", None)
    if memory is None:
        memory = (getattr(agent, "memories", {}) or {}).get("calendar")
    from ..memory.calendar import CalendarMemory
    if not isinstance(memory, CalendarMemory) or not memory.enabled:
        return []
    allowed = {SELF_OWNER, *(str(v) for v in (owner_ids or []))}
    tools: list[Tool] = [SearchCalendarEvents(agent, allowed)]
    if include_writes:
        tools.extend([
            CreateCalendarEvent(agent, allowed),
            UpdateCalendarEvent(agent, allowed),
            CancelCalendarEvent(agent, allowed),
        ])
    return tools
