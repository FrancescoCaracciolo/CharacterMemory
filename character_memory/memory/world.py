"""Private, configurable world simulation for one role-play character.

``WorldMemory`` is a CHARACTER-scoped prompt adapter. Exact mutable state is
kept in ordinary SQLite tables; searchable facts and events are delegated to
an internal ``WorldRecordMemory`` so the StructuredMemory DRY invariant stays
intact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from ..chunking import Chunk
from ..config import WorldConfig
from ..rag.base import Query
from ..rag.hybrid import HybridSearch
from .base import ExtractionSpec, Memory, MemoryItem, MemoryScope, item_bullet
from .store import SQLiteStore
from .structured import StructuredMemory

if TYPE_CHECKING:
    from .extract import ExtractionContext
    from ..tools.base import TurnEffect
    from ..temporal import TemporalResolution, TemporalResolutionEngine


FEATURE_DEFAULTS: dict[str, bool] = {
    "locations": True,
    "activities": True,
    "routines": True,
    "hunger": True,
    "energy": True,
    "sleep": True,
    "autonomous_needs": True,
}
ACTIVITY_KINDS = {
    "idle", "work", "school", "travel", "eat", "sleep",
    "leisure", "social", "other",
}
COMMAND_KINDS = {"move", "start_activity", "eat", "sleep", "wake", "schedule"}
WORLD_RECORD_USER_ID = "_world"


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return clean or "character"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return copy.deepcopy(default)
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return copy.deepcopy(default)
    return parsed


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lo
    return max(lo, min(hi, number))


def _timestamp(value: Any, *, timezone: str = "UTC") -> Optional[float]:
    """Normalize an optional epoch/ISO timestamp used by authored facts."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid world timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return parsed.astimezone(UTC).timestamp()


@dataclass
class WorldCommand:
    kind: str
    actor_id: str
    at: Optional[float] = None
    location_id: Optional[str] = None
    activity_kind: Optional[str] = None
    activity: Optional[str] = None
    until: Optional[float] = None
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    source_message_id: Optional[int] = None
    dedupe_key: Optional[str] = None


@dataclass
class WorldEvent:
    id: int
    occurred_at: float
    actor_id: str
    kind: str
    summary: str
    location_id: Optional[str] = None
    source_message_id: Optional[int] = None
    created: bool = True


@dataclass(frozen=True)
class RoutineOccurrence:
    """One routine resolved to an absolute interval in a specific timezone."""

    routine: dict[str, Any]
    start_at: float
    end_at: float
    timezone: str

    @property
    def id(self) -> str:
        return str(self.routine.get("id") or "routine")


@dataclass
class WorldSnapshot:
    now: float
    timezone: str
    observer_id: str
    observer: dict[str, Any]
    location: Optional[dict[str, Any]] = None
    visible_actors: list[dict[str, Any]] = field(default_factory=list)
    local_facts: list[dict[str, Any]] = field(default_factory=list)
    next_action: Optional[dict[str, Any]] = None
    features: dict[str, bool] = field(default_factory=dict)
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorldRecordMemory(StructuredMemory):
    """Searchable world facts and immutable historical events."""

    name = "world_records"
    table = "world_records"
    scope = MemoryScope.CHARACTER
    extra_columns = {
        "record_type": "TEXT NOT NULL DEFAULT 'fact'",
        "subject_id": "TEXT",
        "location_id": "TEXT",
        "content": "TEXT NOT NULL",
        "occurred_at": "REAL",
        "valid_until": "REAL",
        "visibility": "TEXT NOT NULL DEFAULT 'known'",
        "witnesses": "TEXT NOT NULL DEFAULT '[]'",
        "source": "TEXT NOT NULL DEFAULT 'runtime'",
        "source_message_id": "INTEGER",
        "source_message_ids": "TEXT NOT NULL DEFAULT '[]'",
        "dedupe_key": "TEXT",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    text_column = "content"

    def temporal_interval(
        self, row: dict[str, Any]
    ) -> tuple[float, Optional[float]] | None:
        # Only immutable events have a semantic occurrence time. Durable and
        # temporary facts must not match merely because they were stored then.
        occurred_at = row.get("occurred_at")
        if row.get("record_type") != "event" or occurred_at is None:
            return None
        return (float(occurred_at), None)

    def row_text(self, row: dict[str, Any]) -> str:
        return str(row.get("content") or "")

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        meta = dict(row)
        meta["witnesses"] = _loads(meta.get("witnesses"), [])
        meta["source_message_ids"] = _loads(meta.get("source_message_ids"), [])
        meta["metadata"] = _loads(meta.pop("metadata_json", "{}"), {})
        return MemoryItem(self.row_text(row), score, "world", meta)


class WorldStateStore:
    """Exact state and authored definitions for one private world."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self._create_tables()

    def _create_tables(self) -> None:
        self.store.create_table("world_meta", {
            "key": "TEXT PRIMARY KEY", "value": "TEXT NOT NULL",
        }, pk="key")
        self.store.create_table("world_locations", {
            "id": "TEXT PRIMARY KEY", "name": "TEXT NOT NULL",
            "description": "TEXT NOT NULL DEFAULT ''", "parent_id": "TEXT",
            "timezone": "TEXT", "tags": "TEXT NOT NULL DEFAULT '[]'",
        }, pk="id")
        self.store.create_table("world_actors", {
            "id": "TEXT PRIMARY KEY", "name": "TEXT NOT NULL",
            "home_location": "TEXT", "timezone": "TEXT",
            "public_activity": "INTEGER NOT NULL DEFAULT 0",
            "features": "TEXT NOT NULL DEFAULT '{}'",
            "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        }, pk="id")
        self.store.create_table("world_actor_state", {
            "actor_id": "TEXT PRIMARY KEY", "location_id": "TEXT",
            "activity_kind": "TEXT NOT NULL DEFAULT 'idle'",
            "activity": "TEXT NOT NULL DEFAULT 'idle'",
            "activity_started_at": "REAL", "activity_until": "REAL",
            "activity_source": "TEXT NOT NULL DEFAULT 'seed'",
            "active_routine_id": "TEXT",
            "interruptible": "INTEGER NOT NULL DEFAULT 1",
            "sleeping": "INTEGER NOT NULL DEFAULT 0",
            "hunger": "REAL NOT NULL DEFAULT 0.2",
            "energy": "REAL NOT NULL DEFAULT 0.8", "last_meal_at": "REAL",
            "updated_at": "REAL NOT NULL", "version": "INTEGER NOT NULL DEFAULT 0",
            "override_until": "REAL", "resume_json": "TEXT NOT NULL DEFAULT '{}'",
        }, pk="actor_id")
        self.store.create_table("world_routines", {
            "id": "TEXT PRIMARY KEY", "actor_id": "TEXT NOT NULL",
            "days": "TEXT NOT NULL DEFAULT '[]'", "start_local": "TEXT NOT NULL",
            "duration_minutes": "INTEGER NOT NULL", "location_id": "TEXT",
            "activity_kind": "TEXT NOT NULL DEFAULT 'other'",
            "activity": "TEXT NOT NULL DEFAULT ''", "priority": "INTEGER NOT NULL DEFAULT 0",
            "interruptible": "INTEGER NOT NULL DEFAULT 1",
            "enabled": "INTEGER NOT NULL DEFAULT 1",
        }, pk="id")
        self.store.create_table("world_scheduled_actions", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", "actor_id": "TEXT NOT NULL",
            "due_at": "REAL NOT NULL", "command_json": "TEXT NOT NULL",
            "status": "TEXT NOT NULL DEFAULT 'pending'", "dedupe_key": "TEXT UNIQUE",
            "created_at": "REAL NOT NULL", "source": "TEXT NOT NULL DEFAULT 'manual'",
        })
        # Existing world databases predate explicit activity provenance.  A
        # legacy marker lets the first projection reconcile those ambiguous
        # rows against the authored routines instead of trusting stale state.
        actor_state_columns = set(self.store.columns("world_actor_state"))
        if "activity_source" not in actor_state_columns:
            try:
                self.store.execute(
                    "ALTER TABLE world_actor_state ADD COLUMN activity_source "
                    "TEXT NOT NULL DEFAULT 'legacy'"
                )
            except sqlite3.OperationalError:
                if "activity_source" not in self.store.columns("world_actor_state"):
                    raise
        if "active_routine_id" not in actor_state_columns:
            try:
                self.store.execute(
                    "ALTER TABLE world_actor_state ADD COLUMN active_routine_id TEXT"
                )
            except sqlite3.OperationalError:
                if "active_routine_id" not in self.store.columns("world_actor_state"):
                    raise

    def meta(self, key: str, default: Any = None) -> Any:
        rows = self.store.select("world_meta", {"key": key})
        return _loads(rows[0]["value"], default) if rows else copy.deepcopy(default)

    def set_meta(self, key: str, value: Any) -> None:
        self.store.upsert("world_meta", {"key": key, "value": _json(value)}, pk="key")

    def world_features(self) -> dict[str, bool]:
        stored = self.meta("features", {})
        return {k: bool(stored.get(k, v)) for k, v in FEATURE_DEFAULTS.items()}

    def simulation(self) -> dict[str, Any]:
        defaults = {
            "hunger_per_hour": 0.04, "energy_awake_per_hour": 0.04,
            "energy_sleep_per_hour": 0.12, "hunger_trigger": 0.75,
            "fatigue_trigger": 0.15, "meal_minutes": 30,
            "max_catchup_events": 500,
        }
        return {**defaults, **dict(self.meta("simulation", {}) or {})}

    @property
    def observer_id(self) -> str:
        return str(self.meta("observer_id", "character"))

    @property
    def timezone(self) -> str:
        return str(self.meta("timezone", "UTC"))

    def locations(self) -> list[dict[str, Any]]:
        rows = self.store.select("world_locations", order_by="id")
        for row in rows:
            row["tags"] = _loads(row.get("tags"), [])
        return rows

    def actors(self) -> list[dict[str, Any]]:
        rows = self.store.select("world_actors", order_by="id")
        for row in rows:
            row["features"] = _loads(row.get("features"), {})
            row["metadata"] = _loads(row.pop("metadata_json", "{}"), {})
            row["public_activity"] = bool(row.get("public_activity"))
        return rows

    def actor(self, actor_id: str) -> Optional[dict[str, Any]]:
        rows = self.store.select("world_actors", {"id": actor_id}, limit=1)
        if not rows:
            return None
        row = rows[0]
        row["features"] = _loads(row.get("features"), {})
        row["metadata"] = _loads(row.pop("metadata_json", "{}"), {})
        row["public_activity"] = bool(row.get("public_activity"))
        return row

    def location(self, location_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not location_id:
            return None
        rows = self.store.select("world_locations", {"id": location_id}, limit=1)
        if not rows:
            return None
        row = rows[0]
        row["tags"] = _loads(row.get("tags"), [])
        return row

    def state(self, actor_id: str) -> Optional[dict[str, Any]]:
        rows = self.store.select("world_actor_state", {"actor_id": actor_id}, limit=1)
        if not rows:
            return None
        row = rows[0]
        row["sleeping"] = bool(row.get("sleeping"))
        row["interruptible"] = bool(row.get("interruptible"))
        row["resume"] = _loads(row.pop("resume_json", "{}"), {})
        return row

    def effective_features(self, actor_id: str) -> dict[str, bool]:
        base = self.world_features()
        actor = self.actor(actor_id)
        overrides = (actor or {}).get("features", {})
        return {k: bool(overrides[k]) if k in overrides else v for k, v in base.items()}

    def effective_timezone(
        self, actor_id: str, *, location_id: Optional[str] = None,
    ) -> str:
        """Resolve local time from location ancestry, then actor/world defaults."""
        actor = self.actor(actor_id) or {}
        if location_id is None:
            state = self.state(actor_id) or {}
            location_id = state.get("location_id")
        seen: set[str] = set()
        current = location_id
        while current and current not in seen:
            seen.add(current)
            location = self.location(current)
            if location is None:
                break
            if location.get("timezone"):
                return str(location["timezone"])
            current = location.get("parent_id")
        return str(actor.get("timezone") or self.timezone or "UTC")

    def routines(self, actor_id: Optional[str] = None) -> list[dict[str, Any]]:
        rows = self.store.select(
            "world_routines", {"actor_id": actor_id} if actor_id else None,
            order_by="priority DESC, id ASC",
        )
        for row in rows:
            row["days"] = [int(x) for x in _loads(row.get("days"), [])]
            row["enabled"] = bool(row.get("enabled"))
            row["interruptible"] = bool(row.get("interruptible"))
        return rows

    def set_features(
        self, values: dict[str, Optional[bool]], *, actor_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> dict[str, bool]:
        unknown = set(values) - set(FEATURE_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown world features: {sorted(unknown)}")
        stamp = time.time() if now is None else float(now)
        if actor_id is None:
            current = dict(self.meta("features", {}) or {})
            for key, value in values.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = bool(value)
            self.set_meta("features", current)
            self.store.execute("UPDATE world_actor_state SET updated_at=?", [stamp])
            return self.world_features()
        actor = self.actor(actor_id)
        if actor is None:
            raise ValueError(f"Unknown world actor {actor_id!r}")
        overrides = dict(actor.get("features") or {})
        for key, value in values.items():
            if value is None:
                overrides.pop(key, None)
            else:
                overrides[key] = bool(value)
        self.store.execute(
            "UPDATE world_actors SET features=? WHERE id=?", [_json(overrides), actor_id]
        )
        self.store.execute(
            "UPDATE world_actor_state SET updated_at=? WHERE actor_id=?", [stamp, actor_id]
        )
        return self.effective_features(actor_id)

    def reconcile_seed(self, seed: dict[str, Any], *, now: float) -> None:
        observer = str(seed.get("observer_id") or self.observer_id or "character")
        timezone = str(seed.get("timezone") or "UTC")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone {timezone!r}") from exc
        features = dict(FEATURE_DEFAULTS)
        features.update({
            k: bool(v) for k, v in (seed.get("features") or {}).items()
            if k in features and v is not None
        })
        self.set_meta("observer_id", observer)
        self.set_meta("timezone", timezone)
        self.set_meta("features", features)
        self.set_meta("simulation", dict(seed.get("simulation") or {}))

        location_ids: set[str] = set()
        for entry in seed.get("locations") or []:
            lid = str(entry.get("id") or "").strip()
            if not lid:
                raise ValueError("Every world location needs an id")
            location_timezone = entry.get("timezone")
            if location_timezone:
                try:
                    ZoneInfo(str(location_timezone))
                except ZoneInfoNotFoundError as exc:
                    raise ValueError(
                        f"Unknown IANA timezone {location_timezone!r} for location {lid!r}"
                    ) from exc
            location_ids.add(lid)
            self.store.upsert("world_locations", {
                "id": lid, "name": str(entry.get("name") or lid),
                "description": str(entry.get("description") or ""),
                "parent_id": entry.get("parent_id"), "timezone": location_timezone,
                "tags": _json(entry.get("tags") or []),
            }, pk="id")
        for row in self.store.select("world_locations"):
            if row["id"] not in location_ids:
                self.store.delete("world_locations", {"id": row["id"]})

        actor_ids: set[str] = set()
        for entry in seed.get("actors") or []:
            aid = str(entry.get("id") or "").strip()
            if not aid:
                raise ValueError("Every world actor needs an id")
            actor_timezone = str(entry.get("timezone") or timezone)
            try:
                ZoneInfo(actor_timezone)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(
                    f"Unknown IANA timezone {actor_timezone!r} for actor {aid!r}"
                ) from exc
            overrides = {
                key: bool(value) for key, value in dict(entry.get("features") or {}).items()
                if value is not None
            }
            unknown = set(overrides) - set(FEATURE_DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown features for {aid}: {sorted(unknown)}")
            actor_ids.add(aid)
            self.store.upsert("world_actors", {
                "id": aid, "name": str(entry.get("name") or aid),
                "home_location": entry.get("home_location") or entry.get("location_id"),
                "timezone": actor_timezone,
                "public_activity": 1 if entry.get("public_activity") else 0,
                "features": _json(overrides),
                "metadata_json": _json(entry.get("metadata") or {}),
            }, pk="id")
            if self.state(aid) is None:
                self.store.upsert("world_actor_state", {
                    "actor_id": aid,
                    "location_id": entry.get("location_id") or entry.get("home_location"),
                    "activity_kind": str(entry.get("activity_kind") or "idle"),
                    "activity": str(entry.get("activity") or "idle"),
                    "activity_started_at": now, "activity_until": None,
                    "activity_source": "seed", "active_routine_id": None,
                    "interruptible": 1, "sleeping": 1 if entry.get("sleeping") else 0,
                    "hunger": _clip(entry.get("hunger", 0.2)),
                    "energy": _clip(entry.get("energy", 0.8)),
                    "last_meal_at": entry.get("last_meal_at"), "updated_at": now,
                    "version": 0, "override_until": None, "resume_json": "{}",
                }, pk="actor_id")
        if observer not in actor_ids:
            raise ValueError(f"observer_id {observer!r} is not present in actors")
        for row in self.store.select("world_actors"):
            if row["id"] not in actor_ids:
                self.store.delete("world_actors", {"id": row["id"]})
                self.store.delete("world_actor_state", {"actor_id": row["id"]})

        routine_ids: set[str] = set()
        for i, entry in enumerate(seed.get("routines") or []):
            rid = str(entry.get("id") or f"routine_{i}")
            aid = str(entry.get("actor_id") or observer)
            if aid not in actor_ids:
                raise ValueError(f"Routine {rid!r} names unknown actor {aid!r}")
            kind = str(entry.get("activity_kind") or "other")
            if kind not in ACTIVITY_KINDS:
                raise ValueError(f"Unknown activity kind {kind!r}")
            routine_ids.add(rid)
            self.store.upsert("world_routines", {
                "id": rid, "actor_id": aid, "days": _json(entry.get("days") or []),
                "start_local": str(entry.get("start_local") or "00:00"),
                "duration_minutes": max(1, int(entry.get("duration_minutes") or 1)),
                "location_id": entry.get("location_id"), "activity_kind": kind,
                "activity": str(entry.get("activity") or kind),
                "priority": int(entry.get("priority") or 0),
                "interruptible": 1 if entry.get("interruptible", True) else 0,
                "enabled": 1 if entry.get("enabled", True) else 0,
            }, pk="id")
        for row in self.store.select("world_routines"):
            if row["id"] not in routine_ids:
                self.store.delete("world_routines", {"id": row["id"]})
        advanced = self.meta("advanced_at", None)
        self.set_meta("advanced_at", float(now if advanced is None else advanced))

    def seed_view(self) -> dict[str, Any]:
        actors = self.actors()
        return {
            "version": 1, "observer_id": self.observer_id,
            "timezone": self.timezone, "features": self.world_features(),
            "simulation": self.simulation(), "locations": self.locations(),
            "actors": actors, "routines": self.routines(),
        }


class WorldSimulator(ABC):
    """Extension seam for projecting and advancing exact world state."""

    @abstractmethod
    def project(self, now: float) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def advance(self, now: float) -> list[WorldEvent]:
        raise NotImplementedError


class RuleBasedWorldSimulator(WorldSimulator):
    """Small real-time routine/needs simulator with analytical catch-up."""

    def __init__(
        self, state: WorldStateStore, records: WorldRecordMemory,
        *, clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.state_store = state
        self.records = records
        self.clock = clock or time.time

    @staticmethod
    def _routine_at(
        routines: list[dict[str, Any]], now: float, timezone: str,
    ) -> Optional[RoutineOccurrence]:
        zone = ZoneInfo(timezone)
        current = datetime.fromtimestamp(now, tz=UTC).astimezone(zone)
        matches: list[tuple[int, datetime, RoutineOccurrence]] = []
        for routine in routines:
            if not routine.get("enabled"):
                continue
            try:
                hour, minute = (int(x) for x in routine["start_local"].split(":", 1))
            except (TypeError, ValueError):
                continue
            for day_offset in (0, -1):
                local_day = current.date() + timedelta(days=day_offset)
                if local_day.weekday() not in routine.get("days", []):
                    continue
                start = datetime.combine(
                    local_day, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=zone
                )
                end = start + timedelta(minutes=int(routine["duration_minutes"]))
                if start <= current < end:
                    matches.append((
                        int(routine.get("priority", 0)),
                        start,
                        RoutineOccurrence(
                            routine=routine,
                            start_at=start.timestamp(),
                            end_at=end.timestamp(),
                            timezone=timezone,
                        ),
                    ))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return matches[0][2]

    def _routine_boundaries(
        self, routines: list[dict[str, Any]], start: float, end: float, timezone: str
    ) -> list[float]:
        zone = ZoneInfo(timezone)
        start_day = datetime.fromtimestamp(start, tz=UTC).astimezone(zone).date() - timedelta(days=1)
        end_day = datetime.fromtimestamp(end, tz=UTC).astimezone(zone).date() + timedelta(days=1)
        days = (end_day - start_day).days
        values: set[float] = set()
        for offset in range(days + 1):
            day = start_day + timedelta(days=offset)
            for routine in routines:
                if not routine.get("enabled") or day.weekday() not in routine.get("days", []):
                    continue
                try:
                    hour, minute = (int(x) for x in routine["start_local"].split(":", 1))
                except (TypeError, ValueError):
                    continue
                local = datetime.combine(
                    day, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=zone
                )
                begun = local.timestamp()
                finished = (local + timedelta(minutes=int(routine["duration_minutes"]))).timestamp()
                if start < begun < end:
                    values.add(begun)
                if start < finished < end:
                    values.add(finished)
        return sorted(values)

    @staticmethod
    def _apply_routine(
        projected: dict[str, Any], actor: dict[str, Any], features: dict[str, bool],
        occurrence: Optional[RoutineOccurrence], at: float,
    ) -> None:
        if occurrence is None:
            if projected.get("activity_source") == "routine":
                if features["activities"]:
                    projected.update(activity_kind="idle", activity="idle", activity_started_at=at,
                                     activity_until=None, interruptible=True)
                if features["sleep"]:
                    projected["sleeping"] = False
                projected["activity_source"] = "idle"
                projected["active_routine_id"] = None
            return
        routine = occurrence.routine
        entering = not (
            projected.get("activity_source") == "routine"
            and projected.get("active_routine_id") == occurrence.id
            and projected.get("activity_started_at") == occurrence.start_at
            and projected.get("activity_until") == occurrence.end_at
        )
        kind = routine["activity_kind"]
        if features["locations"] and routine.get("location_id"):
            projected["location_id"] = routine["location_id"]
        if features["activities"]:
            projected.update(
                activity_kind=kind, activity=routine.get("activity") or kind,
                activity_started_at=occurrence.start_at,
                activity_until=occurrence.end_at,
                interruptible=bool(routine.get("interruptible")),
            )
        if features["sleep"]:
            projected["sleeping"] = kind == "sleep"
        if entering and kind == "eat" and features["hunger"]:
            projected["hunger"] = 0.1
            projected["last_meal_at"] = occurrence.start_at
        projected["activity_source"] = "routine"
        projected["active_routine_id"] = occurrence.id

    def _evolve_segment(
        self, projected: dict[str, Any], actor: dict[str, Any], features: dict[str, bool],
        start: float, end: float, *, allow_autonomous: bool,
        resume_routine: Optional[RoutineOccurrence] = None,
    ) -> None:
        sim = self.state_store.simulation()
        cursor = start
        interventions = 0
        cap = max(1, int(sim["max_catchup_events"]))
        while cursor < end:
            sleeping = bool(projected.get("sleeping")) and features["sleep"]
            next_at = end
            intervention: Optional[str] = None
            if (
                projected.get("activity_source") == "autonomous"
                and projected.get("activity_kind") == "eat"
                and projected.get("activity_until")
                and cursor < float(projected["activity_until"]) < next_at
            ):
                next_at = float(projected["activity_until"])
                intervention = "meal_end"
            if allow_autonomous and features["autonomous_needs"] and bool(projected.get("interruptible", True)):
                if features["energy"] and features["sleep"] and not sleeping:
                    rate = float(sim["energy_awake_per_hour"])
                    if rate > 0:
                        hours = (float(projected.get("energy", 0.8)) - float(sim["fatigue_trigger"])) / rate
                        candidate = cursor + max(0.0, hours) * 3600
                        if candidate < next_at:
                            next_at, intervention = candidate, "sleep"
                if features["hunger"]:
                    rate = float(sim["hunger_per_hour"])
                    if rate > 0:
                        hours = (float(sim["hunger_trigger"]) - float(projected.get("hunger", 0.2))) / rate
                        candidate = cursor + max(0.0, hours) * 3600
                        if candidate < next_at:
                            next_at, intervention = candidate, "eat"
            hours = max(0.0, next_at - cursor) / 3600.0
            if features["hunger"]:
                projected["hunger"] = _clip(float(projected.get("hunger", 0.2)) + hours * float(sim["hunger_per_hour"]))
            if features["energy"]:
                delta = float(sim["energy_sleep_per_hour"] if sleeping else -sim["energy_awake_per_hour"])
                projected["energy"] = _clip(float(projected.get("energy", 0.8)) + hours * delta)
            cursor = next_at
            if intervention is None or cursor >= end or interventions >= cap:
                break
            interventions += 1
            if intervention == "meal_end":
                if resume_routine is not None:
                    self._apply_routine(
                        projected, actor, features, resume_routine, cursor
                    )
                elif features["activities"]:
                    projected.update(activity_kind="idle", activity="idle", activity_started_at=cursor,
                                     activity_until=None, interruptible=True)
                    projected["activity_source"] = "idle"
                    projected["active_routine_id"] = None
            elif intervention == "sleep":
                projected["sleeping"] = True
                if features["activities"]:
                    projected.update(activity_kind="sleep", activity="sleeping", activity_started_at=cursor)
                if features["locations"] and actor.get("home_location"):
                    projected["location_id"] = actor["home_location"]
                projected["activity_source"] = "autonomous"
                projected["active_routine_id"] = None
            else:
                projected["hunger"] = 0.1
                projected["last_meal_at"] = cursor
                meal_end = cursor + int(sim["meal_minutes"]) * 60
                if features["activities"]:
                    projected.update(activity_kind="eat", activity="eating", activity_started_at=cursor,
                                     activity_until=meal_end)
                projected["activity_source"] = "autonomous"
                projected["active_routine_id"] = None
                # Prevent a zero-duration retrigger and analytically continue.
                cursor = min(end, cursor + 1e-6)

    def _project_state(self, row: dict[str, Any], actor: dict[str, Any], now: float) -> dict[str, Any]:
        projected = copy.deepcopy(row)
        features = self.state_store.effective_features(actor["id"])
        updated = row.get("updated_at")
        start = float(now if updated is None else updated)
        if now < start:
            return projected
        routines = self.state_store.routines(actor["id"])
        routines_by_id = {str(routine.get("id")): routine for routine in routines}
        boundary_cap = max(1, int(self.state_store.simulation()["max_catchup_events"]))

        # Rows created before activity provenance existed cannot safely be
        # treated as current. Preserve exact needs/location, but let routines
        # and autonomous rules reconstruct the activity from this point.
        if projected.get("activity_source") in {None, "legacy"}:
            if not projected.get("override_until") or float(projected["override_until"]) <= start:
                projected.update(
                    activity_kind="idle", activity="idle", activity_started_at=start,
                    activity_until=None, activity_source="idle", active_routine_id=None,
                    interruptible=True, sleeping=False, override_until=None,
                )

        def expire_override(at: float) -> None:
            until = projected.get("override_until")
            if (
                projected.get("activity_source") == "override"
                and until is not None
                and float(until) <= at
            ):
                projected.update(
                    activity_kind="idle", activity="idle", activity_started_at=at,
                    activity_until=None, activity_source="idle", active_routine_id=None,
                    interruptible=True, sleeping=False, override_until=None,
                )

        def stored_occurrence(at: float, timezone: str) -> Optional[RoutineOccurrence]:
            rid = projected.get("active_routine_id")
            begun = projected.get("activity_started_at")
            until = projected.get("activity_until")
            routine = routines_by_id.get(str(rid)) if rid is not None else None
            if (
                projected.get("activity_source") == "routine"
                and routine is not None
                and begun is not None
                and until is not None
                and float(begun) <= at < float(until)
            ):
                return RoutineOccurrence(routine, float(begun), float(until), timezone)
            return None

        boundary_start = start
        if routines and (now - start) > 8 * 86_400:
            scan_days = max(8, boundary_cap // max(1, len(routines) * 2) + 2)
            boundary_start = max(start, now - scan_days * 86_400)
            if boundary_start > start:
                projected["_catchup_skipped"] = max(
                    1, int((boundary_start - start) / 86_400) * len(routines) * 2
                )

        # For very long gaps, analytically evolve needs through the skipped
        # prefix, then resolve detailed routine boundaries in the recent tail.
        if boundary_start > start:
            self._evolve_segment(
                projected, actor, features, start, boundary_start,
                allow_autonomous=features["autonomous_needs"],
            )
        cursor = start
        if boundary_start > start:
            cursor = boundary_start
        transitions = 0
        while cursor < now and transitions < boundary_cap:
            expire_override(cursor)
            timezone = self.state_store.effective_timezone(
                actor["id"], location_id=projected.get("location_id")
            )
            overridden = bool(
                projected.get("override_until")
                and float(projected["override_until"]) > cursor
            )
            occurrence = stored_occurrence(cursor + 1e-4, timezone)
            if occurrence is None and features["routines"] and not overridden:
                occurrence = self._routine_at(routines, cursor + 1e-4, timezone)
            self._apply_routine(projected, actor, features, occurrence, cursor)

            # Applying a routine may move the actor into another timezone.
            # Re-resolve future boundaries there, while preserving the active
            # occurrence's already-fixed absolute end.
            timezone = self.state_store.effective_timezone(
                actor["id"], location_id=projected.get("location_id")
            )
            boundaries = (
                self._routine_boundaries(routines, cursor, now, timezone)
                if features["routines"] else []
            )
            for candidate in (projected.get("override_until"), projected.get("activity_until")):
                if candidate is not None and cursor < float(candidate) < now:
                    boundaries.append(float(candidate))
            boundary = min(boundaries) if boundaries else now
            self._evolve_segment(
                projected, actor, features, cursor, boundary,
                allow_autonomous=(
                    not overridden
                    and (
                        occurrence is None
                        or bool(occurrence.routine.get("interruptible", True))
                    )
                ),
                resume_routine=occurrence,
            )
            cursor = boundary
            transitions += 1
        if cursor < now:
            projected["_catchup_skipped"] = int(projected.get("_catchup_skipped", 0)) + 1
            self._evolve_segment(
                projected, actor, features, cursor, now,
                allow_autonomous=features["autonomous_needs"],
            )

        # Reconcile exact state at the requested instant. Occurrence timestamps
        # remain anchored to their authored start/end, so this is idempotent.
        expire_override(now)
        timezone = self.state_store.effective_timezone(
            actor["id"], location_id=projected.get("location_id")
        )
        overridden = bool(
            projected.get("override_until")
            and float(projected["override_until"]) > now
        )
        occurrence = stored_occurrence(now + 1e-4, timezone)
        if occurrence is None and features["routines"] and not overridden:
            occurrence = self._routine_at(routines, now + 1e-4, timezone)
        self._apply_routine(projected, actor, features, occurrence, now)
        projected["updated_at"] = now
        return projected

    def project(self, now: float) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for actor in self.state_store.actors():
            row = self.state_store.state(actor["id"])
            if row is not None:
                output[actor["id"]] = self._project_state(row, actor, now)
        return output

    def advance(self, now: float) -> list[WorldEvent]:
        stored_previous = self.state_store.meta("advanced_at", None)
        previous = float(now if stored_previous is None else stored_previous)
        if now <= previous:
            return []
        prior_revision = int(self.state_store.meta("revision", 0) or 0)
        before = {actor["id"]: self.state_store.state(actor["id"]) for actor in self.state_store.actors()}
        projected = self.project(now)
        changed: list[tuple[str, str, dict[str, Any]]] = []
        observer_id = self.state_store.observer_id
        record_ids: list[int] = []
        stored_events: list[WorldEvent] = []
        with self.state_store.store.transaction(immediate=True) as conn:
            for actor_id, row in projected.items():
                old = before.get(actor_id) or {}
                skipped = int(row.pop("_catchup_skipped", 0) or 0)
                meaningful = any(old.get(key) != row.get(key) for key in (
                    "location_id", "activity_kind", "activity", "sleeping",
                    "activity_source", "active_routine_id",
                ))
                conn.execute(
                    "UPDATE world_actor_state SET location_id=?, activity_kind=?, activity=?, "
                    "activity_started_at=?, activity_until=?, activity_source=?, active_routine_id=?, "
                    "interruptible=?, sleeping=?, hunger=?, energy=?, last_meal_at=?, updated_at=?, "
                    "version=version+1, override_until=? "
                    "WHERE actor_id=?",
                    [row.get("location_id"), row.get("activity_kind"), row.get("activity"),
                     now if row.get("activity_started_at") is None else row.get("activity_started_at"),
                     row.get("activity_until"),
                     row.get("activity_source") or "idle", row.get("active_routine_id"),
                     1 if row.get("interruptible", True) else 0, 1 if row.get("sleeping") else 0,
                     row.get("hunger", 0.2), row.get("energy", 0.8), row.get("last_meal_at"),
                     now, row.get("override_until"), actor_id],
                )
                if meaningful:
                    bits = []
                    if old.get("location_id") != row.get("location_id"):
                        bits.append(f"moved to {row.get('location_id')}")
                    if old.get("activity") != row.get("activity"):
                        bits.append(f"started {row.get('activity')}")
                    summary = f"{actor_id} " + " and ".join(bits or ["changed state"])
                    changed.append((actor_id, summary, row))
                    rec = conn.execute(
                        "INSERT OR IGNORE INTO world_records(user_id,importance,created_at,last_recalled,recall_count,"
                        "record_type,subject_id,location_id,content,occurred_at,valid_until,visibility,"
                        "witnesses,source,source_message_id,source_message_ids,dedupe_key,metadata_json) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [WORLD_RECORD_USER_ID, 0.45, now, None, 0, "event", actor_id,
                         row.get("location_id"), summary, now, None, "known", _json([observer_id]),
                         "simulation", None, "[]",
                         f"simulation:{actor_id}:{int(now)}:{row.get('activity_kind')}", "{}"],
                    )
                    if rec.rowcount:
                        record_id = int(rec.lastrowid)
                        record_ids.append(record_id)
                        stored_events.append(WorldEvent(
                            record_id, now, actor_id, "state", summary, row.get("location_id")
                        ))
                if skipped:
                    summary = f"{actor_id}'s long downtime skipped {skipped} detailed routine boundaries."
                    rec = conn.execute(
                        "INSERT OR IGNORE INTO world_records(user_id,importance,created_at,last_recalled,recall_count,"
                        "record_type,subject_id,location_id,content,occurred_at,valid_until,visibility,"
                        "witnesses,source,source_message_id,source_message_ids,dedupe_key,metadata_json) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [WORLD_RECORD_USER_ID, 0.35, now, None, 0, "event", actor_id,
                         row.get("location_id"), summary, now, None, "known", _json([observer_id]),
                         "simulation", None, "[]", f"catchup:{actor_id}:{int(previous)}:{int(now)}", "{}"],
                    )
                    if rec.rowcount:
                        record_id = int(rec.lastrowid)
                        record_ids.append(record_id)
                        stored_events.append(WorldEvent(
                            record_id, now, actor_id, "catchup", summary, row.get("location_id")
                        ))
            conn.execute(
                "INSERT INTO world_meta(key,value) VALUES('advanced_at',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [_json(now)]
            )
            revision = prior_revision + len(changed)
            conn.execute(
                "INSERT INTO world_meta(key,value) VALUES('revision',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [_json(revision)]
            )
            if record_ids:
                meta = conn.execute(
                    "SELECT value FROM world_meta WHERE key='record_revision'"
                ).fetchone()
                record_revision = int(_loads(meta[0], 0) if meta else 0) + len(record_ids)
                conn.execute(
                    "INSERT INTO world_meta(key,value) VALUES('record_revision',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [_json(record_revision)]
                )
        if record_ids:
            self.records.apply_index_changes(updated_ids=record_ids)
        return stored_events


class WorldMemory(Memory):
    """Exact private-world state plus semantically recalled facts/events."""

    name = "world"
    scope = MemoryScope.CHARACTER
    supports_temporal_resolution = True

    def __init__(
        self, store: SQLiteStore, hybrid: HybridSearch, *, character_name: str,
        character_dir: str, config: Optional[WorldConfig] = None,
        enabled: bool = True, half_life: float = 60 * 60 * 24 * 7,
        sticky_threshold: float = 0.95,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.store = store
        self.character_name = character_name
        self.character_dir = character_dir
        self.config = config or WorldConfig()
        self._now = clock or time.time
        self.state_store = WorldStateStore(store)
        self.records = WorldRecordMemory(
            store, hybrid, enabled=enabled, half_life=half_life,
            sticky_threshold=sticky_threshold, clock=self._now,
        )
        self.store.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS world_records_dedupe_idx "
            "ON world_records(dedupe_key) WHERE dedupe_key IS NOT NULL"
        )
        self.simulator: WorldSimulator = RuleBasedWorldSimulator(
            self.state_store, self.records, clock=self._now
        )
        self.seed_path = os.path.join(character_dir, self.config.seed_file)
        if enabled:
            self.load_seed(scaffold=True)

    @staticmethod
    def default_seed(character_name: str) -> dict[str, Any]:
        actor_id = _slug(character_name)
        days = list(range(7))
        routines = [
            {"id": "sleep", "actor_id": actor_id, "days": days,
             "start_local": "23:00", "duration_minutes": 480,
             "location_id": "home", "activity_kind": "sleep",
             "activity": "sleeping", "priority": 10, "interruptible": False},
        ]
        for label, at in (("breakfast", "08:00"), ("lunch", "13:00"), ("dinner", "19:00")):
            routines.append({
                "id": label, "actor_id": actor_id, "days": days,
                "start_local": at, "duration_minutes": 30,
                "location_id": "home", "activity_kind": "eat",
                "activity": f"eating {label}", "priority": 5, "interruptible": True,
            })
        return {
            "version": 1, "observer_id": actor_id, "timezone": "UTC",
            "features": dict(FEATURE_DEFAULTS),
            "simulation": {
                "hunger_per_hour": 0.04, "energy_awake_per_hour": 0.04,
                "energy_sleep_per_hour": 0.12, "hunger_trigger": 0.75,
                "fatigue_trigger": 0.15, "meal_minutes": 30,
                "max_catchup_events": 500,
            },
            "locations": [{
                "id": "home", "name": "Home", "description": "The character's home.",
                "timezone": None,
            }],
            "actors": [{
                "id": actor_id, "name": character_name, "home_location": "home",
                "location_id": "home", "activity_kind": "idle", "activity": "relaxing",
                "hunger": 0.2, "energy": 0.8, "features": {},
            }],
            "routines": routines, "facts": [],
        }

    def load_seed(self, *, scaffold: bool = False) -> dict[str, Any]:
        if not os.path.isfile(self.seed_path):
            seed = self.default_seed(self.character_name)
            if scaffold:
                os.makedirs(os.path.dirname(self.seed_path), exist_ok=True)
                with open(self.seed_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(seed, handle, sort_keys=False, allow_unicode=True)
        else:
            with open(self.seed_path, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError("world.yaml must contain a YAML object")
            seed = loaded
        encoded = yaml.safe_dump(seed, sort_keys=True, allow_unicode=True)
        stamp = self._now()
        prior_hash = self.state_store.meta("seed_hash", None)
        if self.state_store.actors():
            self.advance(stamp)
        self.state_store.reconcile_seed(seed, now=stamp)
        new_hash = hashlib.sha256(encoded.encode()).hexdigest()
        if prior_hash is not None and str(prior_hash) != new_hash:
            self.store.execute(
                "UPDATE world_actor_state SET activity_source='legacy', "
                "active_routine_id=NULL WHERE override_until IS NULL OR override_until<=?",
                [stamp],
            )
        self.store.execute("UPDATE world_actor_state SET updated_at=?", [stamp])
        self.state_store.set_meta("seed_hash", new_hash)
        self._reconcile_seed_facts(seed.get("facts") or [])
        return seed

    def save_seed(self, seed: dict[str, Any]) -> dict[str, Any]:
        os.makedirs(os.path.dirname(self.seed_path), exist_ok=True)
        with open(self.seed_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(seed, handle, sort_keys=False, allow_unicode=True)
        return self.load_seed()

    def seed_view(self) -> dict[str, Any]:
        """Return the complete authored document, including reconciled seed facts."""
        seed = self.state_store.seed_view()
        seed["facts"] = [
            {
                "id": row.get("dedupe_key"),
                "content": row.get("content"),
                "subject_id": row.get("subject_id"),
                "location_id": row.get("location_id"),
                "visibility": row.get("visibility"),
                "importance": row.get("importance"),
                "valid_until": row.get("valid_until"),
                "temporal_kind": _loads(row.get("metadata_json"), {}).get(
                    "temporal_kind", "temporary" if row.get("valid_until") is not None else "durable"
                ),
            }
            for row in self.records.all_rows()
            if row.get("source") == "seed"
        ]
        return seed

    def _reconcile_seed_facts(self, facts: list[dict[str, Any]]) -> None:
        existing = [r for r in self.records.all_rows() if r.get("source") == "seed"]
        by_key = {str(r.get("dedupe_key") or ""): r for r in existing}
        wanted: set[str] = set()
        mutations = 0
        for i, fact in enumerate(facts):
            content = str(fact.get("content") or fact.get("text") or "").strip()
            if not content:
                continue
            key = str(fact.get("id") or f"seed_fact_{i}")
            valid_until = _timestamp(
                fact.get("valid_until"), timezone=self.state_store.timezone
            )
            temporal_kind = str(
                fact.get("temporal_kind")
                or ("temporary" if valid_until is not None else "durable")
            )
            if temporal_kind == "temporary" and valid_until is None:
                raise ValueError(f"Temporary seed fact {key!r} requires valid_until")
            if temporal_kind == "durable" and valid_until is not None:
                raise ValueError(f"Durable seed fact {key!r} cannot have valid_until")
            wanted.add(key)
            if key in by_key:
                row = by_key[key]
                changes = {
                    "content": content, "subject_id": fact.get("subject_id"),
                    "location_id": fact.get("location_id"),
                    "visibility": str(fact.get("visibility") or "known"),
                    "valid_until": valid_until,
                    "metadata_json": _json({"temporal_kind": temporal_kind}),
                }
                if any(row.get(name) != value for name, value in changes.items()):
                    row.update(changes)
                    self.records.update_row(row)
                    self.records.apply_index_changes(updated_ids=[int(row["id"])])
                    mutations += 1
            else:
                self.add_fact(
                    content, subject_id=fact.get("subject_id"), location_id=fact.get("location_id"),
                    visibility=str(fact.get("visibility") or "known"),
                    importance=float(fact.get("importance", 0.8)), source="seed", dedupe_key=key,
                    valid_until=valid_until, temporal_kind=temporal_kind,
                )
        removed = [int(row["id"]) for key, row in by_key.items() if key not in wanted]
        for row_id in removed:
            self.records.delete_row(row_id)
        if removed:
            self.records.apply_index_changes(removed_ids=removed)
            mutations += len(removed)
        if mutations:
            revision = int(self.state_store.meta("record_revision", 0) or 0) + mutations
            self.state_store.set_meta("record_revision", revision)

    def build(self, info_chunks: list[Chunk]) -> None:
        self.records.rebuild_index()

    def rebuild_index(self) -> None:
        self.records.rebuild_index()

    def persist(self, path: str) -> None:
        self.records.persist(path)
        self.state_store.set_meta(
            "indexed_record_revision", int(self.state_store.meta("record_revision", 0) or 0)
        )

    def load(self, path: str) -> None:
        self.records.load(path)
        indexed = int(self.state_store.meta("indexed_record_revision", 0) or 0)
        current = int(self.state_store.meta("record_revision", 0) or 0)
        if indexed != current:
            self.records.rebuild_index()

    @property
    def observer_id(self) -> str:
        return self.state_store.observer_id

    def advance(self, now: Optional[float] = None) -> list[WorldEvent]:
        stamp = self._now() if now is None else float(now)
        stored_previous = self.state_store.meta("advanced_at", None)
        previous = float(stamp if stored_previous is None else stored_previous)
        if stamp <= previous:
            return []
        events: list[WorldEvent] = []
        due = self.store.execute(
            "SELECT * FROM world_scheduled_actions WHERE status='pending' AND due_at>? "
            "AND due_at<=? ORDER BY due_at ASC, id ASC", [previous, stamp]
        )
        for row in due:
            due_at = float(row["due_at"])
            events.extend(self.simulator.advance(due_at))
            payload = _loads(row.get("command_json"), {})
            try:
                payload.setdefault("actor_id", row["actor_id"])
                payload["at"] = due_at
                payload["source"] = "scheduled"
                payload.setdefault("dedupe_key", row.get("dedupe_key") or f"scheduled:{row['id']}")
                command = self.validate_command_dict(payload)
                if command.kind == "schedule":
                    raise ValueError("A scheduled command cannot schedule another command")
                event = self.apply_command(command)
                if event.created:
                    events.append(event)
                self.store.execute(
                    "UPDATE world_scheduled_actions SET status='completed' WHERE id=?", [row["id"]]
                )
            except (TypeError, ValueError):
                self.store.execute(
                    "UPDATE world_scheduled_actions SET status='failed' WHERE id=?", [row["id"]]
                )
        events.extend(self.simulator.advance(stamp))
        return events

    def _snapshot_from_states(
        self, states: dict[str, dict[str, Any]], now: float,
    ) -> WorldSnapshot:
        observer_id = self.observer_id
        observer_actor = self.state_store.actor(observer_id)
        observer_state = states.get(observer_id)
        if observer_actor is None or observer_state is None:
            raise RuntimeError(f"World observer {observer_id!r} has no actor/state")
        features = self.state_store.effective_features(observer_id)
        observer = {**observer_actor, **observer_state}
        if not features["locations"]:
            observer.pop("location_id", None)
        if not features["activities"]:
            for key in ("activity_kind", "activity", "activity_started_at", "activity_until"):
                observer.pop(key, None)
        if not features["hunger"]:
            observer.pop("hunger", None)
        if not features["energy"]:
            observer.pop("energy", None)
        observer["sleeping"] = bool(observer.get("sleeping")) if features["sleep"] else False

        location = None
        raw_location = observer_state.get("location_id")
        if features["locations"] and raw_location:
            location = next((x for x in self.state_store.locations() if x["id"] == raw_location), None)
        visible: list[dict[str, Any]] = []
        for actor in self.state_store.actors():
            if actor["id"] == observer_id:
                continue
            row = states.get(actor["id"])
            if row is None:
                continue
            colocated = features["locations"] and raw_location and row.get("location_id") == raw_location
            if not colocated and not actor.get("public_activity"):
                continue
            item = {"id": actor["id"], "name": actor["name"]}
            actor_features = self.state_store.effective_features(actor["id"])
            if features["locations"] and actor_features["locations"]:
                item["location_id"] = row.get("location_id")
            if actor_features["activities"]:
                item["activity"] = row.get("activity")
                item["activity_kind"] = row.get("activity_kind")
            if actor_features["sleep"]:
                item["sleeping"] = bool(row.get("sleeping"))
            visible.append(item)
        local_facts = []
        for row in self.records.all_rows():
            if row.get("record_type") != "fact":
                continue
            valid_until = row.get("valid_until")
            if valid_until is not None and float(valid_until) <= now:
                continue
            metadata = _loads(row.get("metadata_json"), {})
            # Pre-validity learned rows are ambiguous: preserve them in the
            # searchable ledger, but do not assert them as exact current state.
            if (
                row.get("source") == "extraction"
                and metadata.get("temporal_kind") not in {"durable", "temporary"}
            ):
                continue
            visibility = row.get("visibility")
            if visibility in {"public", "known"} or (
                visibility == "local" and raw_location and row.get("location_id") == raw_location
            ):
                local_facts.append(self.records.row_item(row, 1.0).metadata | {"content": row["content"]})
        local_facts.sort(key=lambda fact: (
            0 if fact.get("valid_until") is not None else 1,
            0 if raw_location and fact.get("location_id") == raw_location else 1,
            -float(fact.get("created_at") or 0),
            -int(fact.get("id") or 0),
        ))
        return WorldSnapshot(
            now=now,
            timezone=self.state_store.effective_timezone(
                observer_id, location_id=observer_state.get("location_id")
            ),
            observer_id=observer_id, observer=observer, location=location,
            visible_actors=visible, local_facts=local_facts[:10],
            next_action=self.next_action(
                observer_id, now, location_id=observer_state.get("location_id")
            ), features=features,
            revision=int(self.state_store.meta("revision", 0) or 0),
        )

    def snapshot(self, *, commit: bool = False, now: Optional[float] = None) -> WorldSnapshot:
        stamp = self._now() if now is None else float(now)
        if commit and self.config.auto_advance:
            self.advance(stamp)
        # Always return a projection at the requested instant. This also
        # reconciles legacy rows when no wall-clock time elapsed since load.
        states = self.simulator.project(stamp)
        return self._snapshot_from_states({k: v for k, v in states.items() if v is not None}, stamp)

    def next_action(
        self, actor_id: str, now: float, *, location_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        features = self.state_store.effective_features(actor_id)
        scheduled = self.store.execute(
            "SELECT * FROM world_scheduled_actions WHERE actor_id=? AND status='pending' "
            "AND due_at>=? ORDER BY due_at ASC LIMIT 50", [actor_id, now]
        )
        for row in scheduled:
            command_kind = _loads(row["command_json"], {}).get("kind", "scheduled")
            if not (
                (command_kind == "move" and not features["locations"])
                or (command_kind == "start_activity" and not features["activities"])
                or (command_kind in {"sleep", "wake"} and not features["sleep"])
            ):
                return {"at": row["due_at"], "kind": command_kind}
        if not features["routines"]:
            return None
        zone = ZoneInfo(self.state_store.effective_timezone(
            actor_id, location_id=location_id
        ))
        current = datetime.fromtimestamp(now, tz=UTC).astimezone(zone)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for routine in self.state_store.routines(actor_id):
            if not routine.get("enabled"):
                continue
            kind = routine["activity_kind"]
            has_enabled_target = (
                (bool(routine.get("location_id")) and features["locations"])
                or (kind == "sleep" and features["sleep"])
                or (kind == "eat" and features["hunger"])
                or features["activities"]
            )
            if not has_enabled_target:
                continue
            hour, minute = (int(x) for x in routine["start_local"].split(":", 1))
            for offset in range(8):
                day = current.date() + timedelta(days=offset)
                if day.weekday() not in routine["days"]:
                    continue
                local = datetime.combine(day, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=zone)
                if local > current:
                    candidates.append((local.timestamp(), routine))
                    break
        if not candidates:
            return None
        at, routine = min(candidates, key=lambda x: x[0])
        return {"at": at, "kind": routine["activity_kind"], "activity": routine["activity"]}

    @staticmethod
    def _qualitative(value: float, labels: tuple[str, str, str, str]) -> str:
        if value < 0.25:
            return labels[0]
        if value < 0.55:
            return labels[1]
        if value < 0.8:
            return labels[2]
        return labels[3]

    def _snapshot_text(self, snap: WorldSnapshot) -> str:
        zone = ZoneInfo(snap.timezone)
        local = datetime.fromtimestamp(snap.now, tz=UTC).astimezone(zone)
        actor = snap.observer
        lines = [f"Local time: {local.strftime('%Y-%m-%d %H:%M (%A)')} ({snap.timezone})."]
        if snap.features["locations"] and snap.location:
            line = f"You are at {snap.location['name']}"
            if snap.location.get("description"):
                line += f": {str(snap.location['description']).rstrip('. ')}"
            lines.append(line.rstrip(". ") + ".")
        if snap.features["activities"] and actor.get("activity"):
            timing: list[str] = []
            if actor.get("activity_started_at") is not None:
                started = datetime.fromtimestamp(
                    float(actor["activity_started_at"]), tz=UTC
                ).astimezone(zone)
                timing.append(f"since {started.strftime('%H:%M')}")
            if actor.get("activity_until") is not None:
                until = datetime.fromtimestamp(
                    float(actor["activity_until"]), tz=UTC
                ).astimezone(zone)
                timing.append(f"until {until.strftime('%H:%M')}")
            suffix = f" ({'; '.join(timing)})" if timing else ""
            lines.append(f"You are currently {actor['activity']}{suffix}.")
        state_bits = []
        if snap.features["sleep"] and actor.get("activity_kind") != "sleep":
            state_bits.append("asleep" if actor.get("sleeping") else "awake")
        if snap.features["hunger"]:
            state_bits.append("hunger: " + self._qualitative(
                float(actor.get("hunger", 0)), ("low", "moderate", "high", "very high")
            ))
        if snap.features["energy"]:
            state_bits.append("energy reserve: " + self._qualitative(
                float(actor.get("energy", 1)), ("depleted", "low", "high", "full")
            ))
        if state_bits:
            lines.append("Physical state: " + "; ".join(state_bits) + ".")
        for other in snap.visible_actors:
            detail = other.get("activity") or "present"
            lines.append(f"Also present/visible: {other['name']} ({detail}).")
        for fact in snap.local_facts[:3]:
            if fact.get("valid_until") is not None:
                until = datetime.fromtimestamp(
                    float(fact["valid_until"]), tz=UTC
                ).astimezone(zone)
                prefix = f"Current observation until {until.strftime('%H:%M')}"
            elif snap.location and fact.get("location_id") == snap.location.get("id"):
                prefix = f"Known at {snap.location['name']}"
            else:
                prefix = "Known world fact"
            lines.append(f"{prefix}: {fact['content']}")
        if snap.next_action:
            when = datetime.fromtimestamp(float(snap.next_action["at"]), tz=UTC).astimezone(zone)
            delta_days = (when.date() - local.date()).days
            day = "today" if delta_days == 0 else "tomorrow" if delta_days == 1 else when.strftime("%Y-%m-%d")
            lines.append(
                f"Next plan {day} at {when.strftime('%H:%M')} ({snap.timezone}): "
                f"{snap.next_action.get('activity') or snap.next_action['kind']}."
            )
        return "\n".join(lines)

    def recall(
        self, query: Query, user_id: str, limit: int, state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: float = 1.0,
    ) -> list[MemoryItem]:
        resolution = self.resolve_temporal(
            query, temporal_resolution, temporal_resolution_engine
        )
        snap = self.snapshot(commit=state_changing and self.config.auto_advance)
        items = [MemoryItem(
            self._snapshot_text(snap), 1.0, self.name,
            {"world_current": True, "snapshot": snap.to_dict()},
        )]
        if limit > 0:
            items.extend(self._visible_record_items(
                query,
                limit=limit,
                state_changing=state_changing,
                temporal_resolution=resolution,
                temporal_resolution_engine=None,
                temporal_weight=temporal_weight,
            ))
        return items

    def format(self, items: list[MemoryItem]) -> str:
        if not items:
            return ""
        current = items[0].text
        history = [item_bullet(item, self.timestamp_style) for item in items[1:]]
        return current + ("\nRelevant world facts/events:\n" + "\n".join(history) if history else "")

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        return self.records.get_memories(limit)

    def add_fact(
        self, content: str, *, subject_id: Optional[str] = None,
        location_id: Optional[str] = None, visibility: str = "known",
        importance: float = 0.6, source: str = "runtime",
        source_message_ids: Optional[list[int]] = None,
        dedupe_key: Optional[str] = None, valid_until: Optional[float] = None,
        temporal_kind: Optional[str] = None,
    ) -> int:
        text = content.strip()
        if not text:
            raise ValueError("World fact content cannot be empty")
        expiry = None if valid_until is None else float(valid_until)
        kind = str(temporal_kind or ("temporary" if expiry is not None else "durable"))
        if kind not in {"durable", "temporary"}:
            raise ValueError("World fact temporal_kind must be durable or temporary")
        if kind == "temporary" and expiry is None:
            raise ValueError("Temporary world facts require valid_until")
        if kind == "durable" and expiry is not None:
            raise ValueError("Durable world facts cannot have valid_until")
        if dedupe_key:
            matches = self.store.select("world_records", {"dedupe_key": dedupe_key})
            if matches:
                return int(matches[0]["id"])
        row_id = self.records.add(
            WORLD_RECORD_USER_ID, importance, record_type="fact", subject_id=subject_id,
            location_id=location_id, content=text, occurred_at=None, valid_until=expiry,
            visibility=visibility,
            witnesses=_json([self.observer_id] if visibility == "known" else []),
            source=source,
            source_message_id=None, source_message_ids=_json(source_message_ids or []),
            dedupe_key=dedupe_key, metadata_json=_json({"temporal_kind": kind}),
        )
        self._bump_record_revision()
        return row_id

    def record_event(
        self, content: str, *, actor_id: Optional[str] = None,
        location_id: Optional[str] = None, occurred_at: Optional[float] = None,
        visibility: str = "known", source: str = "runtime",
        source_message_id: Optional[int] = None,
        source_message_ids: Optional[list[int]] = None,
        dedupe_key: Optional[str] = None,
    ) -> int:
        if dedupe_key:
            matches = self.store.select("world_records", {"dedupe_key": dedupe_key})
            if matches:
                return int(matches[0]["id"])
        row_id = self.records.add(
            WORLD_RECORD_USER_ID, 0.5, record_type="event", subject_id=actor_id,
            location_id=location_id, content=content.strip(),
            occurred_at=self._now() if occurred_at is None else float(occurred_at),
            valid_until=None, visibility=visibility,
            witnesses=_json([self.observer_id]), source=source,
            source_message_id=source_message_id,
            source_message_ids=_json(source_message_ids or []), dedupe_key=dedupe_key,
            metadata_json="{}",
        )
        self._bump_record_revision()
        return row_id

    def _bump_record_revision(self) -> int:
        revision = int(self.state_store.meta("record_revision", 0) or 0) + 1
        self.state_store.set_meta("record_revision", revision)
        return revision

    def _visible_record_items(
        self,
        query: Query,
        *,
        limit: int,
        state_changing: bool = False,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: float = 1.0,
    ) -> list[MemoryItem]:
        snapshot = self.snapshot(commit=False)
        location_id = snapshot.observer.get("location_id")
        # Retrieve a wider candidate pool before applying stable visibility.
        items = self.records.recall(
            query,
            WORLD_RECORD_USER_ID,
            max(limit * 4, limit),
            state_changing=False,
            temporal_resolution=temporal_resolution,
            temporal_resolution_engine=temporal_resolution_engine,
            temporal_weight=temporal_weight,
        )
        visible: list[MemoryItem] = []
        now = self._now()
        for item in items:
            meta = item.metadata
            valid_until = meta.get("valid_until")
            if valid_until is not None and float(valid_until) <= now:
                continue
            visibility = meta.get("visibility")
            witnesses = meta.get("witnesses") or []
            allowed = visibility in {"public", "known"}
            allowed = allowed or self.observer_id in witnesses
            allowed = allowed or (
                visibility == "local" and location_id and meta.get("location_id") == location_id
            )
            if allowed:
                visible.append(item)
            if len(visible) >= limit:
                break
        if state_changing:
            self.records._bump_recall([
                int(item.metadata["id"]) for item in visible if item.metadata.get("id") is not None
            ])
        return visible

    def search_visible(self, query: Query, *, limit: int = 4) -> list[dict[str, Any]]:
        """Search only records the observer may perceive."""
        return [
            {"content": item.text, **item.metadata}
            for item in self._visible_record_items(query, limit=limit, state_changing=False)
        ]

    def list_events(self, *, limit: int = 100, before: Optional[float] = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM world_records WHERE record_type='event'"
        if before is not None:
            sql += " AND occurred_at<?"
            params.append(float(before))
        sql += " ORDER BY occurred_at DESC, id DESC LIMIT ?"
        params.append(max(1, min(1000, int(limit))))
        rows = self.store.execute(sql, params)
        out = []
        for row in rows:
            witnesses = _loads(row.get("witnesses"), [])
            if row.get("visibility") in {"public", "known"} or self.observer_id in witnesses:
                row["witnesses"] = witnesses
                row["source_message_ids"] = _loads(row.get("source_message_ids"), [])
                row["metadata"] = _loads(row.pop("metadata_json", "{}"), {})
                out.append(row)
        return out

    def validate_command_dict(self, value: dict[str, Any]) -> WorldCommand:
        allowed = {field.name for field in WorldCommand.__dataclass_fields__.values()}
        command = WorldCommand(**{k: v for k, v in value.items() if k in allowed})
        self.validate_command(command)
        return command

    def validate_command(self, command: WorldCommand) -> None:
        if command.kind not in COMMAND_KINDS:
            raise ValueError(f"Unknown world command {command.kind!r}")
        actor = self.state_store.actor(command.actor_id)
        if actor is None:
            raise ValueError(f"Unknown world actor {command.actor_id!r}")
        features = self.state_store.effective_features(command.actor_id)
        if command.kind == "move" and not features["locations"]:
            raise ValueError("locations are disabled for this actor")
        if command.kind == "start_activity" and not features["activities"]:
            raise ValueError("activities are disabled for this actor")
        if command.kind in {"sleep", "wake"} and not features["sleep"]:
            raise ValueError("sleep is disabled for this actor")
        if command.kind == "eat" and not (features["hunger"] or features["activities"]):
            raise ValueError("both hunger and activities are disabled for this actor")
        if command.location_id and not self.state_store.actor(command.actor_id):
            raise ValueError(f"Unknown actor {command.actor_id!r}")
        if command.location_id and not self.store.select("world_locations", {"id": command.location_id}):
            raise ValueError(f"Unknown world location {command.location_id!r}")
        if command.activity_kind and command.activity_kind not in ACTIVITY_KINDS:
            raise ValueError(f"Unknown activity kind {command.activity_kind!r}")
        if command.kind == "schedule":
            nested = dict(command.payload or {})
            nested_kind = str(nested.get("kind") or "")
            if not nested_kind or nested_kind == "schedule":
                raise ValueError("schedule requires one non-schedule command in payload")
            nested.setdefault("actor_id", command.actor_id)
            self.validate_command_dict(nested)

    def apply_command(self, command: WorldCommand) -> WorldEvent:
        self.validate_command(command)
        now = self._now() if command.at is None else float(command.at)
        if command.kind == "schedule":
            nested = dict(command.payload)
            nested.setdefault("actor_id", command.actor_id)
            if command.dedupe_key and self.store.select(
                "world_scheduled_actions", {"dedupe_key": command.dedupe_key}, limit=1
            ):
                return WorldEvent(
                    0, now, command.actor_id, "schedule", "Scheduled world action", created=False
                )
            self.store.execute(
                "INSERT OR IGNORE INTO world_scheduled_actions(actor_id,due_at,command_json,status,"
                "dedupe_key,created_at,source) VALUES(?,?,?,?,?,?,?)",
                [command.actor_id, now, _json(nested), "pending", command.dedupe_key,
                 self._now(), command.source],
            )
            return WorldEvent(0, now, command.actor_id, "schedule", "Scheduled world action")
        self._fill_command_until(command, now)
        features = self.state_store.effective_features(command.actor_id)
        observer_id = self.observer_id
        with self.store.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM world_actor_state WHERE actor_id=?", [command.actor_id]
            ).fetchone()
            if row is None:
                raise ValueError(f"Actor {command.actor_id!r} has no state")
            values, summary = self._command_transition(
                dict(row), command, now, features=features
            )
            conn.execute(
                "UPDATE world_actor_state SET location_id=?, activity_kind=?, activity=?, "
                "activity_started_at=?, activity_until=?, sleeping=?, hunger=?, energy=?, "
                "last_meal_at=?, updated_at=?, override_until=?, activity_source=?, "
                "active_routine_id=?, version=version+1 WHERE actor_id=?",
                [*values, command.actor_id],
            )
            rec = conn.execute(
                "INSERT OR IGNORE INTO world_records(user_id,importance,created_at,last_recalled,recall_count,"
                "record_type,subject_id,location_id,content,occurred_at,valid_until,visibility,"
                "witnesses,source,source_message_id,source_message_ids,dedupe_key,metadata_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [WORLD_RECORD_USER_ID, 0.5, now, None, 0, "event", command.actor_id,
                 values[0], summary, now, None, "known", _json([observer_id]), command.source,
                 command.source_message_id,
                 _json([command.source_message_id] if command.source_message_id else []),
                 command.dedupe_key, "{}"],
            )
            if rec.rowcount:
                rid = int(rec.lastrowid)
            elif command.dedupe_key:
                found = conn.execute(
                    "SELECT id FROM world_records WHERE dedupe_key=?", [command.dedupe_key]
                ).fetchone()
                rid = int(found[0]) if found else 0
            else:
                rid = 0
            meta_keys = ["revision"] + (["record_revision"] if rec.rowcount else [])
            for key in meta_keys:
                meta = conn.execute("SELECT value FROM world_meta WHERE key=?", [key]).fetchone()
                value = int(_loads(meta[0], 0) if meta else 0) + 1
                conn.execute(
                    "INSERT INTO world_meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [key, _json(value)]
                )
        if rec.rowcount:
            self.records.apply_index_changes(updated_ids=[rid])
        return WorldEvent(
            rid, now, command.actor_id, command.kind, summary, values[0],
            command.source_message_id, bool(rec.rowcount)
        )

    def commit_turn(
        self, chat_id: str, reply: str, effects: list["TurnEffect"],
        *, calendar: Optional[Any] = None,
    ) -> int:
        """Atomically persist final assistant text and staged world/calendar effects."""
        commands: list[WorldCommand] = []
        calendar_effects: list["TurnEffect"] = []
        for effect in effects:
            if effect.kind in {"calendar_create", "calendar_update", "calendar_cancel"}:
                calendar_effects.append(effect)
                continue
            if effect.kind != "world_command":
                raise ValueError(f"Unsupported deferred effect {effect.kind!r}")
            commands.append(self.validate_command_dict(effect.payload))
        if calendar_effects and calendar is None:
            raise ValueError("Calendar effects require an enabled CalendarMemory")
        now = self._now()
        observer_id = self.observer_id
        for command in commands:
            if command.kind != "schedule":
                self._fill_command_until(command, now)
        command_features = {
            command.actor_id: self.state_store.effective_features(command.actor_id)
            for command in commands
        }
        indexed_ids: list[int] = []
        with self.store.transaction(immediate=True) as conn:
            cur = conn.execute(
                "INSERT INTO messages(chat_id,role,content,user_id,occurred_at,created_at,extracted) "
                "VALUES(?,?,?,?,?,?,0)",
                [chat_id, "assistant", reply, None, None, now],
            )
            message_id = int(cur.lastrowid)
            for command in commands:
                command.source = "model_tool"
                command.source_message_id = message_id
                if command.kind == "schedule":
                    payload = dict(command.payload)
                    payload.setdefault("actor_id", command.actor_id)
                    conn.execute(
                        "INSERT INTO world_scheduled_actions(actor_id,due_at,command_json,status,"
                        "dedupe_key,created_at,source) VALUES(?,?,?,?,?,?,?) "
                        "ON CONFLICT(dedupe_key) DO NOTHING",
                        [command.actor_id, float(command.at or now), _json(payload), "pending",
                         command.dedupe_key, now, "model_tool"],
                    )
                    summary = f"{command.actor_id} scheduled a world action"
                    location_id = None
                    occurred_at = now
                else:
                    row = conn.execute(
                        "SELECT * FROM world_actor_state WHERE actor_id=?", [command.actor_id]
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"Actor {command.actor_id!r} has no state")
                    state = dict(row)
                    values, summary = self._command_transition(
                        state, command, now, features=command_features[command.actor_id]
                    )
                    conn.execute(
                        "UPDATE world_actor_state SET location_id=?, activity_kind=?, activity=?, "
                        "activity_started_at=?, activity_until=?, sleeping=?, hunger=?, energy=?, "
                        "last_meal_at=?, updated_at=?, override_until=?, activity_source=?, "
                        "active_routine_id=?, version=version+1 WHERE actor_id=?",
                        [*values, command.actor_id],
                    )
                    location_id = values[0]
                    occurred_at = now
                dedupe = command.dedupe_key or f"tool:{message_id}:{len(indexed_ids)}:{command.kind}"
                rec = conn.execute(
                    "INSERT OR IGNORE INTO world_records(user_id,importance,created_at,last_recalled,recall_count,"
                    "record_type,subject_id,location_id,content,occurred_at,valid_until,visibility,"
                    "witnesses,source,source_message_id,source_message_ids,dedupe_key,metadata_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [WORLD_RECORD_USER_ID, 0.5, now, None, 0, "event", command.actor_id,
                     location_id, summary, occurred_at, None, "known", _json([observer_id]),
                     "model_tool", message_id, _json([message_id]), dedupe, "{}"],
                )
                if rec.rowcount:
                    indexed_ids.append(int(rec.lastrowid))
            calendar_ids: list[int] = []
            if calendar_effects:
                for effect in calendar_effects:
                    calendar_ids.append(calendar.apply_deferred_effect(
                        conn, effect.kind, effect.payload,
                        source_message_id=message_id, now=now,
                    ))
            revision_row = conn.execute(
                "SELECT value FROM world_meta WHERE key='revision'"
            ).fetchone()
            revision = int(_loads(revision_row[0], 0) if revision_row else 0) + len(commands)
            conn.execute(
                "INSERT INTO world_meta(key,value) VALUES('revision',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [_json(revision)]
            )
            record_row = conn.execute(
                "SELECT value FROM world_meta WHERE key='record_revision'"
            ).fetchone()
            record_revision = int(_loads(record_row[0], 0) if record_row else 0) + len(indexed_ids)
            conn.execute(
                "INSERT INTO world_meta(key,value) VALUES('record_revision',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [_json(record_revision)]
            )
        if indexed_ids:
            self.records.apply_index_changes(updated_ids=indexed_ids)
        if calendar_effects:
            calendar.apply_index_changes(updated_ids=calendar_ids)
        return message_id

    def _command_transition(
        self, state: dict[str, Any], command: WorldCommand, now: float,
        *, features: Optional[dict[str, bool]] = None,
    ) -> tuple[list[Any], str]:
        features = features or self.state_store.effective_features(command.actor_id)
        location = state.get("location_id")
        activity_kind = state.get("activity_kind") or "idle"
        activity = state.get("activity") or "idle"
        sleeping = bool(state.get("sleeping"))
        hunger = float(state.get("hunger", 0.2))
        energy = float(state.get("energy", 0.8))
        last_meal = state.get("last_meal_at")
        if command.kind == "move":
            location = command.location_id
            summary = f"{command.actor_id} moved to {location}"
        elif command.kind == "start_activity":
            activity_kind = command.activity_kind or "other"
            activity = command.activity or activity_kind
            summary = f"{command.actor_id} started {activity}"
        elif command.kind == "eat":
            if features["hunger"]:
                hunger, last_meal = 0.1, now
            if features["activities"]:
                activity_kind, activity = "eat", command.activity or "eating"
            summary = f"{command.actor_id} ate"
        elif command.kind == "sleep":
            sleeping = True
            if features["activities"]:
                activity_kind, activity = "sleep", "sleeping"
            summary = f"{command.actor_id} went to sleep"
        elif command.kind == "wake":
            sleeping = False
            if features["activities"] and activity_kind == "sleep":
                activity_kind, activity = "idle", "idle"
            summary = f"{command.actor_id} woke up"
        else:
            raise ValueError("schedule does not directly change actor state")
        return [
            location, activity_kind, activity, now, command.until,
            1 if sleeping else 0, hunger, energy, last_meal, now, command.until,
            "override", None,
        ], summary

    def _fill_command_until(self, command: WorldCommand, now: float) -> None:
        """Give manual overrides a small deterministic lifetime when omitted."""
        if command.until is not None:
            return
        if command.kind == "eat":
            command.until = now + int(self.state_store.simulation()["meal_minutes"]) * 60
            return
        features = self.state_store.effective_features(command.actor_id)
        if features["routines"] and isinstance(self.simulator, RuleBasedWorldSimulator):
            state = self.state_store.state(command.actor_id) or {}
            timezone = self.state_store.effective_timezone(
                command.actor_id, location_id=state.get("location_id")
            )
            boundaries = self.simulator._routine_boundaries(
                self.state_store.routines(command.actor_id), now, now + 8 * 86_400, timezone
            )
            if boundaries:
                command.until = boundaries[0]
                return
        command.until = now + 3_600

    def set_features(
        self, values: dict[str, Optional[bool]], *, actor_id: Optional[str] = None,
    ) -> dict[str, bool]:
        self.advance()
        return self.state_store.set_features(values, actor_id=actor_id, now=self._now())

    def extraction_spec(self, context: Optional["ExtractionContext"] = None) -> Optional[ExtractionSpec]:
        if not self.config.extract_updates:
            return None
        char = context.character_name if context else self.character_name
        return ExtractionSpec(
            field="world_updates",
            schema={
                "type": "object",
                "properties": {
                    "facts": {"type": "array", "description": (
                        "Established shared-setting properties or explicitly time-bounded conditions. "
                        "Exclude personal activities, exam logistics, and chat incidents; return [] when none qualify."
                    ), "items": {"type": "object", "properties": {
                        "content": {"type": "string"}, "subject_id": {"type": "string"},
                        "location_id": {"type": "string"}, "visibility": {"type": "string"},
                        "importance": {"type": "number"},
                        "temporal_kind": {
                            "type": "string", "enum": ["durable", "temporary"]
                        },
                        "valid_for_minutes": {"type": "number", "minimum": 1},
                        "source_message_ids": {"type": "array", "items": {"type": "integer"}},
                    }, "required": ["content", "temporal_kind", "source_message_ids"]}},
                    "events": {"type": "array", "description": (
                        "Completed happenings that affect the shared setting, not personal or conversation recaps. "
                        "Return [] when none qualify."
                    ), "items": {"type": "object", "properties": {
                        "content": {"type": "string"}, "actor_id": {"type": "string"},
                        "location_id": {"type": "string"},
                        "source_message_ids": {"type": "array", "items": {"type": "integer"}},
                    }, "required": ["content", "source_message_ids"]}},
                    "actions": {"type": "array", "description": (
                        "The character's current actions supported by assistant messages, not users' activities or plans."
                    ), "items": {"type": "object", "properties": {
                        "kind": {"type": "string", "enum": ["move", "start_activity", "eat", "sleep", "wake"]},
                        "location_id": {"type": "string"}, "activity_kind": {"type": "string"},
                        "activity": {"type": "string"},
                        "source_message_ids": {"type": "array", "items": {"type": "integer"}},
                    }, "required": ["kind", "source_message_ids"]}},
                }, "additionalProperties": False,
            },
            instruction=(
                f"- world_updates: explicit updates to {char}'s shared setting only. "
                "facts: durable setting properties or temporary conditions with a positive, "
                "conversation-grounded valid_for_minutes. events: completed changes to the setting. "
                "Examples: the library has a printing press (durable fact); the bridge is closed "
                "for 20 minutes (temporary fact); a storm destroyed the pier (event). "
                "Exclude personal activities, milestones, and chat incidents from both: e.g. "
                "Mira baked cookies, Leo earned a certificate, or a voice call disconnected. "
                "Dates and character mentions do not make these world knowledge. Extract any "
                "actual setting change, not the personal recap. Use top-level facts/episodes "
                "only if enabled and appropriate; otherwise omit, never spill into world_updates. "
                "actions: only the character's current actions supported by assistant messages, "
                "not user activities or user claims about NPC locations. Omit vague future plans. "
                "Include source_message_ids; return empty arrays when nothing qualifies."
            ),
        )

    def apply_extraction(
        self, value: Any, user_id: str, *, chat_id: Optional[str] = None,
    ) -> list[MemoryItem]:
        if not isinstance(value, dict):
            return []
        added: list[MemoryItem] = []
        for fact in value.get("facts") or []:
            ids = [int(x) for x in fact.get("source_message_ids") or []]
            content = str(fact.get("content") or "").strip()
            if not ids or not content:
                continue
            temporal_kind = str(fact.get("temporal_kind") or "")
            if temporal_kind not in {"durable", "temporary"}:
                continue
            valid_until = None
            if temporal_kind == "temporary":
                try:
                    valid_for_minutes = float(fact.get("valid_for_minutes"))
                except (TypeError, ValueError):
                    continue
                if valid_for_minutes <= 0:
                    continue
                valid_until = self._now() + valid_for_minutes * 60
            key = f"extract:fact:{ids}:{content.lower()}"
            rid = self.add_fact(
                content, subject_id=fact.get("subject_id"), location_id=fact.get("location_id"),
                visibility=str(fact.get("visibility") or "known"),
                importance=_clip(fact.get("importance", 0.6)), source="extraction",
                source_message_ids=ids, dedupe_key=key,
                valid_until=valid_until, temporal_kind=temporal_kind,
            )
            row = self.records.get_row(rid)
            if row:
                added.append(self.records.row_item(row, self.records._effective(row)))
        for event in value.get("events") or []:
            ids = [int(x) for x in event.get("source_message_ids") or []]
            content = str(event.get("content") or "").strip()
            if not ids or not content:
                continue
            rid = self.record_event(
                content, actor_id=event.get("actor_id"), location_id=event.get("location_id"),
                source="extraction", source_message_ids=ids,
                dedupe_key=f"extract:event:{ids}:{content.lower()}",
            )
            row = self.records.get_row(rid)
            if row:
                added.append(self.records.row_item(row, self.records._effective(row)))
        for action in value.get("actions") or []:
            ids = [int(x) for x in action.get("source_message_ids") or []]
            if not ids:
                continue
            id_marks = ",".join("?" for _ in ids)
            source_rows = self.store.execute(
                f"SELECT id,role FROM messages WHERE id IN ({id_marks})", ids
            )
            assistant_ids = [int(row["id"]) for row in source_rows if row.get("role") == "assistant"]
            if not assistant_ids:
                continue
            # A tool-committed action for the same assistant message wins.
            q = ",".join("?" for _ in assistant_ids)
            committed = self.store.execute(
                f"SELECT id FROM world_records WHERE source='model_tool' AND source_message_id IN ({q}) LIMIT 1",
                assistant_ids,
            )
            if committed:
                continue
            try:
                result = self.apply_command(WorldCommand(
                    kind=str(action.get("kind") or ""), actor_id=self.observer_id,
                    location_id=action.get("location_id"),
                    activity_kind=action.get("activity_kind"), activity=action.get("activity"),
                    source="extraction", source_message_id=assistant_ids[-1],
                    dedupe_key=f"extract:action:{ids}:{action.get('kind')}",
                ))
            except ValueError:
                continue
            row = self.records.get_row(result.id)
            if row:
                added.append(self.records.row_item(row, self.records._effective(row)))
        return added
