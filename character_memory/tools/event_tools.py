"""Read-only, provider-neutral tools for conversation-event retrieval."""

from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from .base import Tool

if TYPE_CHECKING:
    from ..agent import CharacterAgent
    from ..memory.base import MemoryItem


_DATE_PATTERNS = (
    "%Y/%m/%d (%a) %H:%M",
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def parse_time(value: str | float | int) -> datetime:
    """Parse ISO 8601, Unix timestamps, and benchmark date strings as UTC."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if not text:
        raise ValueError("time value cannot be empty")
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        return datetime.fromtimestamp(numeric, tz=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None:
        for pattern in _DATE_PATTERNS:
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError(f"unsupported time format: {value!r}")
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def _shift_months(value: datetime, months: int) -> datetime:
    total = value.year * 12 + value.month - 1 + months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _range_payload(expression: str, reference: datetime, start: datetime, end: datetime) -> dict:
    return {
        "expression": expression,
        "reference_time": reference.isoformat(),
        "occurred_from": start.isoformat(),
        "occurred_before": end.isoformat(),
        "occurred_from_timestamp": start.timestamp(),
        "occurred_before_timestamp": end.timestamp(),
    }


class ResolveTimeRange(Tool):
    name = "resolve_time_range"
    description = (
        "Resolve a relative time expression against a reference timestamp into "
        "a half-open occurrence range. Use it for phrases such as 'last month', "
        "'three weeks ago', or 'yesterday', then pass the returned ISO bounds "
        "to search_conversation_events."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "reference_time": {
                "type": "string",
                "description": "ISO 8601 or the timestamp shown with the question.",
            },
        },
        "required": ["expression", "reference_time"],
    }

    def run(self, expression: str, reference_time: str) -> dict:
        reference = parse_time(reference_time)
        phrase = expression.strip().lower()
        day = reference.replace(hour=0, minute=0, second=0, microsecond=0)
        if phrase == "today":
            return _range_payload(expression, reference, day, day + timedelta(days=1))
        if phrase == "yesterday":
            return _range_payload(expression, reference, day - timedelta(days=1), day)
        if phrase in {"last week", "previous week"}:
            this_week = day - timedelta(days=day.weekday())
            return _range_payload(
                expression, reference, this_week - timedelta(days=7), this_week
            )
        if phrase in {"last month", "previous month"}:
            this_month = day.replace(day=1)
            return _range_payload(
                expression, reference, _shift_months(this_month, -1), this_month
            )
        if phrase in {"last year", "previous year"}:
            this_year = day.replace(month=1, day=1)
            return _range_payload(
                expression, reference, this_year.replace(year=this_year.year - 1), this_year
            )
        match = re.fullmatch(
            r"(?:about\s+)?(\d+)\s+(day|week|month|year)s?\s+ago", phrase
        )
        if not match:
            raise ValueError(
                "unsupported relative expression; use today, yesterday, last "
                "week/month/year, or '<number> <unit> ago'"
            )
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            start = day - timedelta(days=amount)
            end = start + timedelta(days=1)
        elif unit == "week":
            this_week = day - timedelta(days=day.weekday())
            start = this_week - timedelta(weeks=amount)
            end = start + timedelta(weeks=1)
        elif unit == "month":
            this_month = day.replace(day=1)
            start = _shift_months(this_month, -amount)
            end = _shift_months(start, 1)
        else:
            this_year = day.replace(month=1, day=1)
            start = this_year.replace(year=this_year.year - amount)
            end = start.replace(year=start.year + 1)
        return _range_payload(expression, reference, start, end)


class CalculateTimeDifference(Tool):
    name = "calculate_time_difference"
    description = (
        "Calculate the elapsed whole days, weeks, months, or years between two "
        "event/question timestamps. Use this instead of estimating relative "
        "time mentally."
    )
    parameters = {
        "type": "object",
        "properties": {
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
            "unit": {
                "type": "string",
                "enum": ["days", "weeks", "months", "years"],
                "default": "months",
            },
        },
        "required": ["start_time", "end_time"],
    }

    def run(
        self, start_time: str, end_time: str, unit: str = "months"
    ) -> dict:
        if unit not in {"days", "weeks", "months", "years"}:
            raise ValueError("unit must be days, weeks, months, or years")
        start = parse_time(start_time)
        end = parse_time(end_time)
        if end < start:
            raise ValueError("end_time must not precede start_time")
        elapsed_days = (end - start).total_seconds() / 86_400
        if unit == "days":
            value = int(elapsed_days)
        elif unit == "weeks":
            value = int(elapsed_days // 7)
        else:
            months = (end.year - start.year) * 12 + end.month - start.month
            if _shift_months(start, months) > end:
                months -= 1
            value = months if unit == "months" else months // 12
        return {
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "unit": unit,
            "whole_units": value,
            "elapsed_days": elapsed_days,
        }


def _event_memory(agent: "CharacterAgent"):
    from ..memory.conversation_events import ConversationEventMemory

    memory = (getattr(agent, "memories", {}) or {}).get("conversation_events")
    if not isinstance(memory, ConversationEventMemory) or not memory.enabled:
        raise RuntimeError("conversation_events memory is not available")
    return memory


def _event_user(memory, user_id: Optional[str]) -> Optional[str]:
    if user_id:
        return user_id
    users = sorted({str(row["user_id"]) for row in memory.all_rows()})
    return users[0] if len(users) == 1 else None


def _event_record(item: "MemoryItem", *, max_chars: int) -> dict[str, Any]:
    timestamp = item.metadata.get("occurred_at")
    content = item.text
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars].rstrip() + "…"
    return {
        "event_id": int(item.metadata["id"]),
        "user_id": item.metadata.get("user_id"),
        "chat_id": item.metadata.get("chat_id"),
        "occurred_at": (
            datetime.fromtimestamp(float(timestamp), tz=UTC).isoformat()
            if timestamp is not None
            else None
        ),
        "source_message_ids": item.metadata.get("source_message_ids", []),
        "matched_key_kind": item.metadata.get("matched_key_kind"),
        "matched_source_memory": item.metadata.get("matched_source_memory"),
        "score": item.score,
        "content": content,
        "truncated": truncated,
    }


class SearchConversationEvents(Tool):
    name = "search_conversation_events"
    description = (
        "Search immutable raw conversation events and their extracted fact, "
        "directive, and episode aliases. Use focused keyword/sub-question "
        "queries. Optional occurrence bounds are ISO timestamps and use "
        "[occurred_from, occurred_before). Results contain event IDs, timestamps, "
        "source IDs, matching key type, and compact raw snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Focused search query."},
            "user_id": {"type": "string"},
            "occurred_from": {"type": "string"},
            "occurred_before": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
        },
        "required": ["query"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        occurred_from: Optional[str] = None,
        occurred_before: Optional[str] = None,
        limit: int = 8,
    ) -> dict:
        memory = _event_memory(self.agent)
        items = memory.search_events(
            query,
            user_id=_event_user(memory, user_id),
            occurred_from=(parse_time(occurred_from).timestamp() if occurred_from else None),
            occurred_before=(parse_time(occurred_before).timestamp() if occurred_before else None),
            limit=max(1, min(20, int(limit))),
        )
        return {
            "query": query,
            "count": len(items),
            "events": [_event_record(item, max_chars=700) for item in items],
        }


class GetConversationEvents(Tool):
    name = "get_conversation_events"
    description = (
        "Fetch complete immutable raw conversation events by event ID after "
        "search_conversation_events identifies relevant candidates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "event_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 10,
            },
            "max_chars_each": {
                "type": "integer",
                "minimum": 500,
                "maximum": 8000,
                "default": 4000,
            },
        },
        "required": ["event_ids"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, event_ids: list[int], max_chars_each: int = 4000) -> dict:
        memory = _event_memory(self.agent)
        ids = [int(event_id) for event_id in event_ids[:10]]
        items = memory.events_by_ids(ids)
        max_chars = max(500, min(8000, int(max_chars_each)))
        return {
            "requested_event_ids": ids,
            "count": len(items),
            "events": [_event_record(item, max_chars=max_chars) for item in items],
        }


class GetEventNeighbors(Tool):
    name = "get_event_neighbors"
    description = (
        "Fetch events immediately before and after an event in the same chat. "
        "Use this for 'what happened next/before?', vague references, and local "
        "sequence reconstruction."
    )
    parameters = {
        "type": "object",
        "properties": {
            "event_id": {"type": "integer"},
            "before": {"type": "integer", "minimum": 0, "maximum": 10, "default": 2},
            "after": {"type": "integer", "minimum": 0, "maximum": 10, "default": 2},
        },
        "required": ["event_id"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, event_id: int, before: int = 2, after: int = 2) -> dict:
        memory = _event_memory(self.agent)
        items = memory.event_neighbors(
            int(event_id),
            before=max(0, min(10, int(before))),
            after=max(0, min(10, int(after))),
        )
        return {
            "anchor_event_id": int(event_id),
            "count": len(items),
            "events": [_event_record(item, max_chars=2500) for item in items],
        }


def conversation_event_tools(agent: "CharacterAgent") -> list[Tool]:
    """Return the standard read-only event/temporal tools bound to an agent."""
    return [
        ResolveTimeRange(),
        CalculateTimeDifference(),
        SearchConversationEvents(agent),
        GetConversationEvents(agent),
        GetEventNeighbors(agent),
    ]
