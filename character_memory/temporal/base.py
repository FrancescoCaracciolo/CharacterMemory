"""Public temporal-resolution types and extension seam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..rag.base import Query


UTC = timezone.utc


def coerce_datetime(value: datetime | float | int | str | None) -> datetime:
    """Return an aware UTC datetime, using ``now`` for a missing value."""
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (float, int)):
        return datetime.fromtimestamp(float(value), UTC)
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    else:
        raise TypeError(f"Unsupported datetime value: {type(value).__name__}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class TemporalRange:
    """One resolved, half-open UTC interval: ``start <= t < end``."""

    start: datetime
    end: datetime
    expression: str = ""
    grain: str = "unknown"
    confidence: float = 1.0
    query_weight: float = 1.0

    def __post_init__(self) -> None:
        start = coerce_datetime(self.start)
        end = coerce_datetime(self.end)
        if end <= start:
            raise ValueError("TemporalRange.end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self, "confidence", max(0.0, min(1.0, float(self.confidence)))
        )
        object.__setattr__(
            self, "query_weight", max(0.0, min(1.0, float(self.query_weight)))
        )

    @property
    def start_timestamp(self) -> float:
        return self.start.timestamp()

    @property
    def end_timestamp(self) -> float:
        return self.end.timestamp()


@dataclass(frozen=True)
class TemporalMatch:
    """Best match between a memory timestamp/interval and a resolution."""

    score: float = 0.0
    range: Optional[TemporalRange] = None


@dataclass(frozen=True)
class TemporalResolution:
    """Temporal ranges extracted from a message or weighted query history."""

    ranges: tuple[TemporalRange, ...] = field(default_factory=tuple)
    reference_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    timezone: str = "UTC"
    engine: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranges", tuple(self.ranges))
        object.__setattr__(self, "reference_time", coerce_datetime(self.reference_time))

    @property
    def is_empty(self) -> bool:
        return not self.ranges

    def match(
        self,
        start: datetime | float | int | str | None,
        end: datetime | float | int | str | None = None,
    ) -> TemporalMatch:
        """Return the strongest overlap with a point or half-open interval."""
        if start is None or not self.ranges:
            return TemporalMatch()
        try:
            item_start = coerce_datetime(start)
            # A point is represented as one microsecond so half-open overlap
            # semantics remain consistent at exact boundaries.
            item_end = coerce_datetime(end) if end is not None else item_start
        except (TypeError, ValueError, OverflowError):
            return TemporalMatch()

        best = TemporalMatch()
        for resolved in self.ranges:
            if end is None:
                overlaps = resolved.start <= item_start < resolved.end
            else:
                overlaps = item_start < resolved.end and item_end > resolved.start
            if not overlaps:
                continue
            score = resolved.confidence * resolved.query_weight
            if score > best.score:
                best = TemporalMatch(score=score, range=resolved)
        return best


class TemporalResolutionEngine(ABC):
    """Resolve temporal expressions without knowing anything about memories."""

    name: str = "base"

    @abstractmethod
    def resolve(
        self,
        message_or_history: Query,
        *,
        reference_time: datetime | float | int | str | None = None,
        timezone: str = "UTC",
    ) -> TemporalResolution:
        """Resolve a message or weighted history relative to ``reference_time``."""
