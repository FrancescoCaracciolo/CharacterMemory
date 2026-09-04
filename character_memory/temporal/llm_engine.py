"""Single-call LLM temporal-resolution engine."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..llm.base import LLMClient
from ..rag.base import Query, as_queries
from .base import (
    TemporalRange,
    TemporalResolution,
    TemporalResolutionEngine,
    coerce_datetime,
)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _range_datetime(value: Any, timezone_name: str) -> datetime:
    """Parse an LLM range bound; naive strings mean the configured timezone."""
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
            except (ZoneInfoNotFoundError, ValueError):
                parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return coerce_datetime(parsed)
    return coerce_datetime(value)


class LLMTemporalResolutionEngine(TemporalResolutionEngine):
    """Resolve references with exactly one ``LLMClient.chat`` call."""

    name = "llm"

    def __init__(self, llm: LLMClient, *, max_tokens: int = 768) -> None:
        self.llm = llm
        self.max_tokens = max(64, int(max_tokens))

    def resolve(
        self,
        message_or_history: Query,
        *,
        reference_time: datetime | float | int | str | None = None,
        timezone: str = "UTC",
    ) -> TemporalResolution:
        reference = coerce_datetime(reference_time)
        queries = as_queries(message_or_history)
        if not queries:
            return TemporalResolution((), reference, timezone, self.name)
        payload = [
            {"index": index, "text": text, "weight": weight}
            for index, (text, weight) in enumerate(queries)
        ]
        prompt = (
            "Extract every temporal reference relevant to recalling past memories. "
            "Resolve relative expressions against the supplied reference time and timezone. "
            "Return JSON only with this shape: "
            '{"ranges":[{"start":"ISO-8601","end":"ISO-8601",'
            '"expression":"source phrase","grain":"second|minute|hour|day|week|month|year|range",'
            '"confidence":0.0,"query_index":0}]}. '
            "Intervals are half-open; use calendar boundaries in the supplied timezone. "
            "Do not invent a range when a query has no temporal reference.\n\n"
            f"Reference time (UTC): {reference.isoformat()}\n"
            f"Timezone: {timezone}\n"
            f"Queries: {json.dumps(payload, ensure_ascii=False)}"
        )
        # Deliberately use chat(), not chat_structured(): some structured
        # clients retry malformed JSON, which would violate the one-call seam.
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You resolve temporal expressions into exact intervals.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        parsed = _json_object(raw)
        ranges: list[TemporalRange] = []
        for item in (
            parsed.get("ranges", ())
            if isinstance(parsed.get("ranges", ()), list)
            else ()
        ):
            if not isinstance(item, dict):
                continue
            try:
                query_index = int(item.get("query_index", 0))
                query_weight = (
                    queries[query_index][1] if 0 <= query_index < len(queries) else 1.0
                )
                ranges.append(
                    TemporalRange(
                        _range_datetime(item.get("start"), timezone),
                        _range_datetime(item.get("end"), timezone),
                        expression=str(item.get("expression") or ""),
                        grain=str(item.get("grain") or "unknown"),
                        confidence=float(item.get("confidence", 1.0)),
                        query_weight=query_weight,
                    )
                )
            except (TypeError, ValueError, OverflowError, IndexError):
                continue
        return TemporalResolution(tuple(ranges), reference, timezone, self.name)
