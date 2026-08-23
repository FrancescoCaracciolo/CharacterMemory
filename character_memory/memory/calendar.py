"""Structured calendar memory.

Calendar rows are deliberately kept separate from the private world.  A
calendar can attach a :class:`WorldMemory` as a live source, in which case
world routines are projected as read-only occurrences and are never copied
into the calendar table.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dt_time, timedelta
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..chunking import Chunk
from ..rag.base import Query
from ..rag.hybrid import HybridSearch
from .base import ExtractionSpec, MemoryItem, MemoryScope, item_bullet
from .structured import StructuredMemory

if TYPE_CHECKING:
    from .extract import ExtractionContext
    from .store import SQLiteStore
    from .world import WorldMemory


SELF_OWNER = "_self"
EVENT_KIND = "event"
ROUTINE_KIND = "routine"
ACTIVE_STATUS = "scheduled"
CANCELLED_STATUS = "cancelled"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return default


def _owner_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        value = _loads(value, [])
    return [str(v) for v in (value or []) if str(v).strip()]


def _list_value(value: Any, default: Optional[list[Any]] = None) -> list[Any]:
    """Decode a list supplied either as JSON (SQLite) or as a Python list."""
    if isinstance(value, str):
        decoded = _loads(value, None)
        if isinstance(decoded, (list, tuple)):
            return list(decoded)
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return list(default or [])


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(str(name or "UTC"))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone {name!r}") from exc


def _parse_iso(value: Any, *, timezone: str = "UTC") -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        raise ValueError("calendar time cannot be empty")
    # HTML text inputs and JSON round-trips can turn an epoch number into a
    # string; accept that representation alongside ISO-8601.
    try:
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO calendar time {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_zone(timezone))
    return parsed.astimezone(UTC).timestamp()


def _format_local(ts: float, timezone: str) -> str:
    return datetime.fromtimestamp(float(ts), tz=UTC).astimezone(_zone(timezone)).isoformat()


def _parse_local_time(value: Any) -> dt_time:
    text = str(value or "").strip()
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", text)
    if not match:
        raise ValueError(f"Expected local time HH:MM, got {value!r}")
    return dt_time(int(match.group(1)), int(match.group(2)))


@dataclass(frozen=True)
class CalendarEvent:
    """A persisted event or weekly routine definition."""

    id: int
    owner_id: str
    title: str
    description: str = ""
    location: str = ""
    timezone: str = "UTC"
    kind: str = EVENT_KIND
    start_at: Optional[float] = None
    end_at: Optional[float] = None
    weekdays: tuple[int, ...] = ()
    start_local: Optional[str] = None
    duration_minutes: Optional[int] = None
    attendees: tuple[str, ...] = ()
    status: str = ACTIVE_STATUS
    source: str = "manual"
    source_message_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "timezone": self.timezone,
            "kind": self.kind,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "weekdays": list(self.weekdays),
            "start_local": self.start_local,
            "duration_minutes": self.duration_minutes,
            "attendees": list(self.attendees),
            "status": self.status,
            "source": self.source,
            "source_message_ids": list(self.source_message_ids),
            "metadata": dict(self.metadata),
        }
        if self.start_at is not None:
            payload["start"] = _format_local(self.start_at, self.timezone)
        if self.end_at is not None:
            payload["end"] = _format_local(self.end_at, self.timezone)
        return payload


@dataclass(frozen=True)
class CalendarOccurrence:
    """One concrete occurrence, including virtual world routine instances."""

    id: str
    series_id: str
    owner_id: str
    title: str
    start_at: float
    end_at: float
    timezone: str
    description: str = ""
    location: str = ""
    attendees: tuple[str, ...] = ()
    kind: str = EVENT_KIND
    source: str = "manual"
    status: str = ACTIVE_STATUS
    virtual: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cancelled(self) -> bool:
        return self.status == CANCELLED_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "series_id": self.series_id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "timezone": self.timezone,
            "kind": self.kind,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "start": _format_local(self.start_at, self.timezone),
            "end": _format_local(self.end_at, self.timezone),
            "attendees": list(self.attendees),
            "source": self.source,
            "status": self.status,
            "virtual": self.virtual,
            "metadata": dict(self.metadata),
        }


class CalendarSource(ABC):
    """Read-only source of concrete calendar occurrences."""

    name: str = "source"

    @abstractmethod
    def occurrences(
        self, start: float, before: float, *, owners: Optional[set[str]] = None
    ) -> list[CalendarOccurrence]:
        raise NotImplementedError


class WorldRoutineCalendarSource(CalendarSource):
    """Live projection of enabled routines from a character's world."""

    name = "world"

    def __init__(self, world: "WorldMemory") -> None:
        self.world = world

    def occurrences(
        self, start: float, before: float, *, owners: Optional[set[str]] = None
    ) -> list[CalendarOccurrence]:
        if not self.world.enabled:
            return []
        observer = self.world.observer_id
        if owners is not None and observer not in owners and SELF_OWNER not in owners:
            return []
        features = self.world.state_store.effective_features(observer)
        if not features.get("routines", True):
            return []
        snapshot = self.world.snapshot(commit=False)
        timezone = snapshot.timezone
        routines = self.world.state_store.routines(observer)
        out: list[CalendarOccurrence] = []
        zone = _zone(timezone)
        local_start = datetime.fromtimestamp(float(start), tz=UTC).astimezone(zone).date() - timedelta(days=1)
        local_end = datetime.fromtimestamp(float(before), tz=UTC).astimezone(zone).date() + timedelta(days=1)
        cursor = local_start
        while cursor <= local_end:
            weekday = cursor.weekday()
            for routine in routines:
                if not routine.get("enabled") or weekday not in set(routine.get("days") or []):
                    continue
                at = _parse_local_time(routine.get("start_local"))
                start_dt = datetime.combine(cursor, at, tzinfo=zone)
                start_ts = start_dt.astimezone(UTC).timestamp()
                end_ts = start_ts + max(1, int(routine.get("duration_minutes") or 1)) * 60
                if end_ts <= start or start_ts >= before:
                    continue
                rid = str(routine.get("id") or "routine")
                occurrence_date = cursor.isoformat()
                out.append(CalendarOccurrence(
                    id=f"world:{rid}:{occurrence_date}",
                    series_id=f"world:{rid}",
                    owner_id=SELF_OWNER,
                    title=str(routine.get("activity") or routine.get("activity_kind") or rid),
                    start_at=start_ts,
                    end_at=end_ts,
                    timezone=timezone,
                    location=str(routine.get("location_id") or ""),
                    kind=ROUTINE_KIND,
                    source=self.name,
                    virtual=True,
                    metadata={
                        "world_routine_id": rid,
                        "actor_id": observer,
                        "activity_kind": routine.get("activity_kind"),
                    },
                ))
            cursor += timedelta(days=1)
        return out


class CalendarMemory(StructuredMemory):
    """Dated events and weekly routines for the character and its users."""

    name = "calendar"
    table = "calendar_events"
    extra_columns = {
        "title": "TEXT NOT NULL",
        "description": "TEXT NOT NULL DEFAULT ''",
        "location": "TEXT NOT NULL DEFAULT ''",
        "timezone": "TEXT NOT NULL DEFAULT 'UTC'",
        "kind": "TEXT NOT NULL DEFAULT 'event'",
        "start_at": "REAL",
        "end_at": "REAL",
        "weekdays": "TEXT NOT NULL DEFAULT '[]'",
        "start_local": "TEXT",
        "duration_minutes": "INTEGER",
        "attendees": "TEXT NOT NULL DEFAULT '[]'",
        "status": "TEXT NOT NULL DEFAULT 'scheduled'",
        "source": "TEXT NOT NULL DEFAULT 'manual'",
        "source_message_ids": "TEXT NOT NULL DEFAULT '[]'",
        "updated_at": "REAL NOT NULL",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    text_column = "title"

    def __init__(
        self,
        store: "SQLiteStore",
        hybrid: HybridSearch,
        *,
        timezone: str = "UTC",
        near_past_hours: int = 24,
        near_future_days: int = 14,
        extract_updates: bool = True,
        enabled: bool = True,
        half_life: float = 60 * 60 * 24 * 365,
        sticky_threshold: float = 0.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.default_timezone = str(timezone or "UTC")
        _zone(self.default_timezone)
        self.near_past_hours = max(0, int(near_past_hours))
        self.near_future_days = max(1, int(near_future_days))
        self.extract_updates = bool(extract_updates)
        self._sources: list[CalendarSource] = []
        super().__init__(
            store,
            hybrid,
            enabled=enabled,
            half_life=half_life,
            sticky_threshold=sticky_threshold,
            clock=clock,
        )

    @property
    def timezone(self) -> str:
        """Default timezone used for naive event timestamps and searches."""
        return self.default_timezone

    # ------------------------------------------------------------------ rows
    @staticmethod
    def _row_event(row: dict[str, Any]) -> CalendarEvent:
        metadata = _loads(row.get("metadata_json"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        return CalendarEvent(
            id=int(row["id"]),
            owner_id=str(row.get("user_id") or SELF_OWNER),
            title=str(row.get("title") or ""),
            description=str(row.get("description") or ""),
            location=str(row.get("location") or ""),
            timezone=str(row.get("timezone") or "UTC"),
            kind=str(row.get("kind") or EVENT_KIND),
            start_at=float(row["start_at"]) if row.get("start_at") is not None else None,
            end_at=float(row["end_at"]) if row.get("end_at") is not None else None,
            weekdays=tuple(int(v) for v in _list_value(row.get("weekdays"), [])),
            start_local=row.get("start_local"),
            duration_minutes=(int(row["duration_minutes"]) if row.get("duration_minutes") is not None else None),
            attendees=tuple(_owner_ids(row.get("attendees"))),
            status=str(row.get("status") or ACTIVE_STATUS),
            source=str(row.get("source") or "manual"),
            source_message_ids=tuple(int(v) for v in _list_value(row.get("source_message_ids"), [])),
            metadata=metadata,
        )

    def row_text(self, row: dict[str, Any]) -> str:
        bits = [str(row.get("title") or "")]
        for key in ("description", "location"):
            if row.get(key):
                bits.append(str(row[key]))
        attendees = _owner_ids(row.get("attendees"))
        if attendees:
            bits.append("with " + ", ".join(attendees))
        if str(row.get("kind") or EVENT_KIND) == ROUTINE_KIND:
            days = ",".join(str(value) for value in _list_value(row.get("weekdays"), []))
            if days:
                bits.append(f"weekly weekdays {days}")
            if row.get("start_local"):
                bits.append(str(row["start_local"]))
        elif row.get("start_at") is not None:
            timezone = str(row.get("timezone") or self.default_timezone)
            try:
                bits.append(_format_local(float(row["start_at"]), timezone))
            except (TypeError, ValueError):
                pass
        return " — ".join(bit for bit in bits if bit)

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        event = self._row_event(row)
        meta = dict(row)
        meta.update({
            "owner_id": event.owner_id,
            "attendees": list(event.attendees),
            "weekdays": list(event.weekdays),
            "source_message_ids": list(event.source_message_ids),
            "metadata": event.metadata,
        })
        meta.pop("metadata_json", None)
        return MemoryItem(self._display_event(event), score, self.name, meta)

    def _display_event(self, event: CalendarEvent | CalendarOccurrence) -> str:
        if isinstance(event, CalendarEvent):
            if event.kind == ROUTINE_KIND:
                days = ",".join(str(v) for v in event.weekdays)
                when = f"weekly ({days}) {event.start_local or ''}"
            elif event.start_at is not None:
                when = _format_local(event.start_at, event.timezone)
            else:
                when = "unscheduled"
            title = event.title
            owner = "character" if event.owner_id == SELF_OWNER else event.owner_id
            return f"{title} ({when}; {owner})"
        when = f"{_format_local(event.start_at, event.timezone)}–{_format_local(event.end_at, event.timezone)}"
        owner = "character" if event.owner_id == SELF_OWNER else event.owner_id
        suffix = f" at {event.location}" if event.location else ""
        return f"{event.title} ({when}; {owner}{suffix})"

    def index_chunks(self, row: dict[str, Any]) -> list[Chunk]:
        text = self.row_text(row)
        base = {"id": int(row["id"]), "user_id": str(row.get("user_id") or SELF_OWNER)}
        attendees = _owner_ids(row.get("attendees"))
        chunks = [Chunk(text=text, source=self.table, metadata=base)]
        for owner in attendees:
            if owner != base["user_id"]:
                chunks.append(Chunk(text=text, source=self.table, metadata={**base, "user_id": owner}))
        return chunks

    def _effective(self, row: dict[str, Any]) -> float:
        # Calendar entries remain useful because of their date, not because
        # they were recently recalled.  Keep a stable salience score.
        return float(row.get("importance", 1.0))

    # ----------------------------------------------------------- source wiring
    def import_source(self, source: CalendarSource) -> "CalendarMemory":
        if source not in self._sources:
            self._sources.append(source)
        return self

    def import_world(self, world: "WorldMemory") -> "CalendarMemory":
        return self.import_source(WorldRoutineCalendarSource(world))

    @property
    def sources(self) -> tuple[CalendarSource, ...]:
        return tuple(self._sources)

    # -------------------------------------------------------------- validation
    def _clean_event(self, values: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
        raw = dict(values)
        out: dict[str, Any] = {}
        if not partial or "title" in raw:
            title = str(raw.get("title") or "").strip()
            if not title:
                raise ValueError("calendar title is required")
            out["title"] = title
        for name in ("description", "location"):
            if not partial or name in raw:
                out[name] = str(raw.get(name) or "").strip()
        timezone = str(raw.get("timezone") or self.default_timezone)
        if not partial or "timezone" in raw:
            _zone(timezone)
            out["timezone"] = timezone
        if not partial or "kind" in raw:
            kind = str(raw.get("kind") or EVENT_KIND)
            if kind not in {EVENT_KIND, ROUTINE_KIND}:
                raise ValueError("calendar kind must be event or routine")
            out["kind"] = kind
        if not partial or "attendees" in raw:
            attendees = raw.get("attendees", [])
            if isinstance(attendees, str):
                decoded = _loads(attendees, None)
                attendees = decoded if isinstance(decoded, list) else [part.strip() for part in attendees.split(",") if part.strip()]
            if not isinstance(attendees, (list, tuple)):
                raise ValueError("attendees must be a list")
            out["attendees"] = _json(list(dict.fromkeys(str(v).strip() for v in attendees if str(v).strip())))
        kind = out.get("kind")
        if kind is None and partial:
            old = self.get_row(int(raw.get("id"))) if raw.get("id") is not None else None
            kind = str((old or {}).get("kind") or EVENT_KIND)
        if kind == ROUTINE_KIND:
            if not partial or "weekdays" in raw:
                days = sorted({int(v) for v in _list_value(raw.get("weekdays"), [])})
                if not days or any(v < 0 or v > 6 for v in days):
                    raise ValueError("routine weekdays must contain values 0..6")
                out["weekdays"] = _json(days)
            if not partial or "start_local" in raw:
                at = raw.get("start_local")
                _parse_local_time(at)
                out["start_local"] = str(at)
            if not partial or "duration_minutes" in raw:
                duration = int(raw.get("duration_minutes") or 0)
                if duration <= 0:
                    raise ValueError("routine duration_minutes must be positive")
                out["duration_minutes"] = duration
            if not partial:
                out.update(start_at=None, end_at=None)
        else:
            if not partial or "start_at" in raw or "start" in raw:
                start = _parse_iso(raw.get("start_at", raw.get("start")), timezone=timezone)
                out["start_at"] = start
            if not partial or "end_at" in raw or "end" in raw:
                end = _parse_iso(raw.get("end_at", raw.get("end")), timezone=timezone)
                out["end_at"] = end
            if not partial and out["end_at"] <= out["start_at"]:
                raise ValueError("calendar event end must be after start")
            if partial and "start_at" in out and "end_at" not in out:
                old = self.get_row(int(raw["id"]))
                if old and old.get("end_at") is not None and out["start_at"] >= float(old["end_at"]):
                    raise ValueError("calendar event start must be before end")
            if partial and "end_at" in out and "start_at" not in out:
                old = self.get_row(int(raw["id"]))
                if old and old.get("start_at") is not None and out["end_at"] <= float(old["start_at"]):
                    raise ValueError("calendar event end must be after start")
            if not partial:
                out.update(weekdays="[]", start_local=None, duration_minutes=None)
        return out

    @staticmethod
    def _event_values(row: dict[str, Any]) -> dict[str, Any]:
        """Return a decoded, complete event payload suitable for validation."""
        return {
            "title": row.get("title"),
            "description": row.get("description") or "",
            "location": row.get("location") or "",
            "timezone": row.get("timezone") or "UTC",
            "kind": row.get("kind") or EVENT_KIND,
            "start_at": row.get("start_at"),
            "end_at": row.get("end_at"),
            "weekdays": _list_value(row.get("weekdays"), []),
            "start_local": row.get("start_local"),
            "duration_minutes": row.get("duration_minutes"),
            "attendees": _list_value(row.get("attendees"), []),
        }

    def _prepare_update_row(self, row: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
        """Validate a complete result of a partial event update for ``row``."""
        merged = self._event_values(row)
        for key, value in values.items():
            if key == "start":
                key = "start_at"
            elif key == "end":
                key = "end_at"
            if key in merged:
                merged[key] = value
        return self._clean_event(merged)

    def prepare_update(self, event_id: int, values: dict[str, Any]) -> dict[str, Any]:
        """Validate a complete result of a partial event update without writing it."""
        row = self.get_row(int(event_id))
        if row is None:
            raise KeyError(event_id)
        return self._prepare_update_row(row, values)

    def create_event(self, owner_id: str = SELF_OWNER, title: Optional[str] = None, *, importance: float = 1.0, source: str = "manual", source_message_ids: Optional[Iterable[int]] = None, **values: Any) -> int:
        if owner_id == SELF_OWNER:
            # Structured-memory callers often use the shared ``user_id``
            # spelling. Accept it as an ergonomic alias while keeping the
            # calendar-facing ``owner_id`` API explicit.
            alias = values.pop("user_id", None)
            if alias is not None:
                owner_id = str(alias)
        owner = str(owner_id or "").strip()
        if not owner:
            raise ValueError("calendar owner_id is required")
        if title is not None:
            values.setdefault("title", title)
        clean = self._clean_event(values)
        stamp = self._now()
        source_ids = list(dict.fromkeys(int(v) for v in (source_message_ids or [])))
        if source_ids:
            for existing in self.all_rows(owner):
                if str(existing.get("title") or "").strip().lower() != clean["title"].lower():
                    continue
                prior = set(int(v) for v in _list_value(existing.get("source_message_ids"), []))
                if prior and prior == set(source_ids):
                    return int(existing["id"])
        return self.add(
            owner,
            float(importance),
            **clean,
            status=ACTIVE_STATUS,
            source=str(source or "manual"),
            source_message_ids=_json(source_ids),
            updated_at=stamp,
            metadata_json=_json(values.get("metadata") or {}),
        )

    # Friendly aliases used by integrations that call calendar records
    # "events" rather than generic structured-memory rows.
    add_event = create_event
    create = create_event

    def update_event(self, event_id: int, *, source_message_ids: Optional[Iterable[int]] = None, **values: Any) -> int:
        row = self.get_row(int(event_id))
        if row is None:
            raise KeyError(event_id)
        clean = self.prepare_update(int(event_id), values)
        if source_message_ids is not None:
            prior = _list_value(row.get("source_message_ids"), [])
            clean["source_message_ids"] = _json(list(dict.fromkeys([*map(int, prior), *map(int, source_message_ids)])))
        clean["updated_at"] = self._now()
        row.update(clean)
        self.update_row(row)
        self.apply_index_changes(updated_ids=[int(event_id)])
        return int(event_id)

    edit_event = update_event
    update = update_event

    def cancel_event(self, event_id: int, *, source_message_ids: Optional[Iterable[int]] = None) -> int:
        row = self.get_row(int(event_id))
        if row is None:
            raise KeyError(event_id)
        row["status"] = CANCELLED_STATUS
        row["updated_at"] = self._now()
        if source_message_ids is not None:
            prior = _list_value(row.get("source_message_ids"), [])
            row["source_message_ids"] = _json(list(dict.fromkeys([*map(int, prior), *map(int, source_message_ids)])))
        self.update_row(row)
        self.apply_index_changes(updated_ids=[int(event_id)])
        return int(event_id)

    cancel = cancel_event

    def apply_deferred_effect(
        self, conn: Any, kind: str, payload: dict[str, Any], *, source_message_id: int, now: float
    ) -> int:
        """Apply one model-tool effect inside an existing SQLite transaction."""
        if kind == "calendar_create":
            owner = str(payload.get("owner_id") or "").strip()
            if not owner:
                raise ValueError("calendar owner_id is required")
            values = dict(payload.get("values") or {})
            clean = self._clean_event(values)
            row = {
                "user_id": owner,
                "importance": 1.0,
                "created_at": now,
                "last_recalled": None,
                "recall_count": 0,
                **clean,
                "status": ACTIVE_STATUS,
                "source": "model_tool",
                "source_message_ids": _json([int(source_message_id)]),
                "updated_at": now,
                "metadata_json": _json(values.get("metadata") or {}),
            }
            cols = list(row)
            cur = conn.execute(
                f"INSERT INTO {self.table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [row[col] for col in cols],
            )
            return int(cur.lastrowid)
        event_id = int(payload.get("event_id"))
        current_row = conn.execute(f"SELECT * FROM {self.table} WHERE id=?", [event_id]).fetchone()
        if current_row is None:
            raise KeyError(event_id)
        current = dict(current_row)
        expected_owner = str(payload.get("owner_id") or "").strip()
        if expected_owner and expected_owner != str(current.get("user_id") or SELF_OWNER):
            raise ValueError("calendar event is outside this chat's calendar scope")
        if kind == "calendar_cancel":
            current["status"] = CANCELLED_STATUS
        elif kind == "calendar_update":
            values = dict(payload.get("values") or {})
            clean = self._prepare_update_row(current, values)
            current.update(clean)
        else:
            raise ValueError(f"Unsupported calendar effect {kind!r}")
        prior_ids = [int(v) for v in _list_value(current.get("source_message_ids"), [])]
        current["source_message_ids"] = _json(list(dict.fromkeys([*prior_ids, int(source_message_id)])))
        current["updated_at"] = now
        cols = [
            "user_id", "importance", "created_at", "last_recalled", "recall_count", "title",
            "description", "location", "timezone", "kind", "start_at", "end_at", "weekdays",
            "start_local", "duration_minutes", "attendees", "status", "source", "source_message_ids",
            "updated_at", "metadata_json",
        ]
        conn.execute(
            f"UPDATE {self.table} SET {', '.join(f'{col}=?' for col in cols)} WHERE id=?",
            [current.get(col) for col in cols] + [event_id],
        )
        return event_id

    # ------------------------------------------------------------- projection
    def _row_occurrences(
        self,
        event: CalendarEvent,
        start: float,
        before: float,
        *,
        include_cancelled: bool = False,
    ) -> list[CalendarOccurrence]:
        if event.status == CANCELLED_STATUS and not include_cancelled:
            return []
        if event.kind == EVENT_KIND:
            if event.start_at is None or event.end_at is None or event.end_at <= start or event.start_at >= before:
                return []
            return [CalendarOccurrence(
                id=str(event.id), series_id=str(event.id), owner_id=event.owner_id,
                title=event.title, description=event.description, location=event.location,
                start_at=event.start_at, end_at=event.end_at, timezone=event.timezone,
                attendees=event.attendees, kind=event.kind, source=event.source,
                status=event.status, metadata={"event_id": event.id, **event.metadata},
            )]
        at = _parse_local_time(event.start_local)
        zone = _zone(event.timezone)
        local_start = datetime.fromtimestamp(float(start), tz=UTC).astimezone(zone).date() - timedelta(days=1)
        local_end = datetime.fromtimestamp(float(before), tz=UTC).astimezone(zone).date() + timedelta(days=1)
        out: list[CalendarOccurrence] = []
        cursor = local_start
        while cursor <= local_end:
            if cursor.weekday() in set(event.weekdays):
                start_dt = datetime.combine(cursor, at, tzinfo=zone)
                start_ts = start_dt.astimezone(UTC).timestamp()
                end_ts = start_ts + max(1, int(event.duration_minutes or 1)) * 60
                if end_ts > start and start_ts < before:
                    out.append(CalendarOccurrence(
                        id=f"{event.id}:{cursor.isoformat()}", series_id=str(event.id),
                        owner_id=event.owner_id, title=event.title,
                        description=event.description, location=event.location,
                        start_at=start_ts, end_at=end_ts, timezone=event.timezone,
                        attendees=event.attendees, kind=event.kind, source=event.source,
                        status=event.status, metadata={"event_id": event.id, **event.metadata},
                    ))
            cursor += timedelta(days=1)
        return out

    @staticmethod
    def _visible(event: CalendarEvent, owners: Optional[set[str]]) -> bool:
        if owners is None:
            return True
        return event.owner_id in owners or bool(set(event.attendees) & owners)

    def occurrences(
        self, start: float, before: float, *, owners: Optional[Iterable[str]] = None,
        include_cancelled: bool = False,
    ) -> list[CalendarOccurrence]:
        owner_set = {str(v) for v in owners} if owners is not None else None
        out: list[CalendarOccurrence] = []
        for row in self.all_rows():
            event = self._row_event(row)
            if not self._visible(event, owner_set):
                continue
            if not include_cancelled and event.status == CANCELLED_STATUS:
                continue
            out.extend(self._row_occurrences(
                event,
                float(start),
                float(before),
                include_cancelled=include_cancelled,
            ))
        for source in self._sources:
            out.extend(source.occurrences(float(start), float(before), owners=owner_set))
        # A shared persisted event should only occur once even when requested
        # through several owners.
        out.sort(key=lambda item: (item.start_at, item.end_at, item.id))
        return list({item.id: item for item in out}.values())

    def _semantic_scores(self, query: Query) -> dict[int, float]:
        if not query:
            return {}
        try:
            hits = self.hybrid.search(query, k=max(20, self.hybrid.candidate_pool))
        except Exception:
            return {}
        scores: dict[int, float] = {}
        for hit in hits:
            rid = hit.metadata.get("id")
            if rid is not None:
                scores[int(rid)] = max(scores.get(int(rid), 0.0), float(hit.score))
        return scores

    def search_events(
        self,
        query: Query = "",
        *,
        start: Optional[float] = None,
        before: Optional[float] = None,
        owners: Optional[Iterable[str]] = None,
        include_cancelled: bool = False,
        limit: int = 8,
        state_changing: bool = False,
    ) -> list[CalendarOccurrence]:
        now = self._now()
        lo = now - self.near_past_hours * 3600 if start is None else float(start)
        hi = now + self.near_future_days * 86_400 if before is None else float(before)
        if hi <= lo:
            raise ValueError("calendar range must end after it starts")
        limit = int(limit)
        if limit <= 0:
            return []
        occurrences = self.occurrences(lo, hi, owners=owners, include_cancelled=include_cancelled)
        if not occurrences:
            return []
        if isinstance(query, list):
            query_terms = []
            for entry in query:
                if isinstance(entry, str):
                    term = entry.strip().lower()
                    weight = 1.0
                elif isinstance(entry, (tuple, list)) and entry:
                    term = str(entry[0] or "").strip().lower()
                    try:
                        weight = float(entry[1]) if len(entry) > 1 else 1.0
                    except (TypeError, ValueError):
                        weight = 1.0
                    if term:
                        query_terms.append((term, max(0.0, weight)))
        else:
            text = str(query or "").strip().lower()
            query_terms = [(text, 1.0)] if text else []
        has_query = bool(query_terms)
        semantic = self._semantic_scores(query)
        matched: list[tuple[float, CalendarOccurrence]] = []
        for occurrence in occurrences:
            hay = " ".join((occurrence.title, occurrence.description, occurrence.location)).lower()
            lexical = sum(hay.count(term) * weight for term, weight in query_terms)
            row_id = int(occurrence.metadata.get("event_id", -1)) if not occurrence.virtual else -1
            score = lexical + semantic.get(row_id, 0.0)
            if has_query and not score:
                continue
            # Ongoing and imminent occurrences are naturally more useful than
            # distant ones, while relevance remains the dominant signal.
            proximity = 2.0 if occurrence.start_at <= now <= occurrence.end_at else 1.0 / (1.0 + max(0.0, occurrence.start_at - now) / 86_400)
            matched.append((score * 10.0 + proximity, occurrence))
        if has_query:
            matched.sort(key=lambda pair: (-pair[0], pair[1].start_at))
        else:
            matched = [(0.0, item) for item in occurrences]
        return [item for _, item in matched[:limit]]

    def search(
        self,
        query: Query = "",
        *,
        start: Optional[float] = None,
        before: Optional[float] = None,
        owners: Optional[Iterable[str]] = None,
        include_cancelled: bool = False,
        limit: int = 8,
        state_changing: bool = False,
    ) -> list[CalendarOccurrence]:
        """Convenience alias for :meth:`search_events`."""
        return self.search_events(
            query,
            start=start,
            before=before,
            owners=owners,
            include_cancelled=include_cancelled,
            limit=limit,
            state_changing=state_changing,
        )

    # Memory interface ---------------------------------------------------
    def recall(self, query: Query, user_id: str, limit: int, state_changing: bool = True) -> list[MemoryItem]:
        owners = {SELF_OWNER, str(user_id)}
        events = self.search_events(query, owners=owners, limit=limit, state_changing=state_changing)
        return [MemoryItem(self._display_event(item), 1.0, self.name, {
            "id": item.id, "event_id": item.metadata.get("event_id", item.id),
            "owner_id": item.owner_id, "user_id": item.owner_id,
            "attendees": list(item.attendees), "occurred_at": item.start_at,
            "start_at": item.start_at, "end_at": item.end_at, "timezone": item.timezone,
            "source": item.source, "virtual": item.virtual, **item.metadata,
        }) for item in events]

    def recall_participants(self, query: Query, participants: list[str], limit: int, state_changing: bool = True) -> list[MemoryItem]:
        owners = {SELF_OWNER, *(str(v) for v in participants)}
        events = self.search_events(query, owners=owners, limit=limit, state_changing=state_changing)
        return [MemoryItem(self._display_event(item), 1.0, self.name, {
            "id": item.id, "event_id": item.metadata.get("event_id", item.id),
            "owner_id": item.owner_id, "user_id": item.owner_id,
            "attendees": list(item.attendees), "occurred_at": item.start_at,
            "start_at": item.start_at, "end_at": item.end_at, "timezone": item.timezone,
            "source": item.source, "virtual": item.virtual, **item.metadata,
        }) for item in events]

    def format(self, items: list[MemoryItem]) -> str:
        return "\n".join(item_bullet(item, self.timestamp_style) for item in items)

    # Extraction ---------------------------------------------------------
    def extraction_spec(self, context: Optional["ExtractionContext"] = None) -> Optional[ExtractionSpec]:
        if not self.enabled or not self.extract_updates:
            return None
        people = [SELF_OWNER]
        if context:
            if str(context.user_name) and str(context.user_name) not in people:
                people.append(str(context.user_name))
            people.extend(str(v) for v in context.participants if str(v) not in people)
        visible = self.occurrences(
            self._now() - self.near_past_hours * 3600,
            self._now() + self.near_future_days * 86_400,
            owners=people,
        )[:50]
        existing = "; ".join(
            f"id={item.metadata.get('event_id', item.id)} {item.title} at {_format_local(item.start_at, item.timezone)}"
            for item in visible if not item.virtual
        )
        owner_enum = {"type": "string", "enum": people}
        return ExtractionSpec(
            field="calendar_updates",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["create", "update", "cancel"]},
                        "event_id": {"type": "integer"},
                        "owner_id": owner_enum,
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "location": {"type": "string"},
                        "timezone": {"type": "string"},
                        "start_at": {"type": "string"},
                        "end_at": {"type": "string"},
                        "kind": {"type": "string", "enum": [EVENT_KIND, ROUTINE_KIND]},
                        "weekdays": {"type": "array", "items": {"type": "integer"}},
                        "start_local": {"type": "string"},
                        "duration_minutes": {"type": "integer"},
                        "attendees": {"type": "array", "items": {"type": "string"}},
                        "source_message_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["operation", "source_message_ids"],
                },
            },
            instruction=(
                "- calendar_updates: create, update, or cancel explicit calendar commitments. "
                "Use ISO timestamps, never infer a vague possibility, and use the listed event_id "
                "for updates/cancellations. Owner IDs are _self for the character or the named user. "
                f"Current active events: {existing or 'none'}."
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        allowed_owners = {SELF_OWNER, str(user_id)}
        if chat_id:
            # Extraction is normally called with a persisted Chat. Derive the
            # complete speaker set here so a malformed model payload cannot
            # write to an unrelated user's calendar merely by naming them as
            # ``owner_id``. Direct/unit callers without a messages table keep
            # the single-user fallback above.
            try:
                rows = self.store.execute(
                    "SELECT DISTINCT user_id FROM messages "
                    "WHERE chat_id=? AND role='user' AND user_id IS NOT NULL",
                    [chat_id],
                )
                allowed_owners.update(str(row["user_id"]) for row in rows if row.get("user_id"))
            except Exception:
                pass
        added: list[MemoryItem] = []
        for update in value or []:
            if not isinstance(update, dict):
                continue
            operation = str(update.get("operation") or "")
            try:
                ids = [int(v) for v in update.get("source_message_ids") or []]
            except (TypeError, ValueError):
                continue
            if not ids:
                continue
            try:
                if operation == "create":
                    owner = str(update.get("owner_id") or update.get("user_id") or user_id)
                    if owner not in allowed_owners:
                        continue
                    row_id = self.create_event(owner, source="extraction", source_message_ids=ids, **{
                        key: update[key] for key in (
                            "title", "description", "location", "timezone", "kind", "start_at", "end_at",
                            "weekdays", "start_local", "duration_minutes", "attendees",
                        ) if key in update
                    })
                elif operation in {"update", "cancel"}:
                    event_id = int(update["event_id"])
                    existing = self.get_row(event_id)
                    if existing is None:
                        continue
                    actor = str(update.get("owner_id") or update.get("user_id") or user_id)
                    if actor not in allowed_owners:
                        continue
                    event_owner = str(existing.get("user_id") or SELF_OWNER)
                    attendees = set(_owner_ids(existing.get("attendees")))
                    if actor != event_owner and actor not in attendees:
                        continue
                    if operation == "update":
                        row_id = self.update_event(event_id, source_message_ids=ids, **{
                            key: update[key] for key in (
                                "title", "description", "location", "timezone", "kind", "start_at", "end_at",
                                "weekdays", "start_local", "duration_minutes", "attendees",
                            ) if key in update
                        })
                    else:
                        row_id = self.cancel_event(event_id, source_message_ids=ids)
                else:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            row = self.get_row(int(row_id))
            if row:
                added.append(self.row_item(row, self._effective(row)))
        return added
