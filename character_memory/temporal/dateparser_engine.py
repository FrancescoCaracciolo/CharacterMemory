"""Low-latency multilingual temporal resolution powered by dateparser data."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import regex
from dateparser.languages.loader import default_loader
from dateparser.search import search_dates

from ..rag.base import Query, as_queries
from .base import (
    TemporalRange,
    TemporalResolution,
    TemporalResolutionEngine,
    coerce_datetime,
)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


@dataclass(frozen=True)
class _RelativePhrase:
    canonical: str


DEFAULT_FAST_LANGUAGES = (
    "en",
    "it",
    "es",
    "fr",
    "de",
    "pt",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sk",
    "sl",
    "sv",
    "da",
    "nb",
    "fi",
    "tr",
    "ro",
    "hu",
    "el",
    "ar",
    "he",
    "hi",
    "zh",
    "ja",
    "ko",
    "id",
    "vi",
    "th",
)


@lru_cache(maxsize=8)
def _language_data(
    languages: tuple[str, ...],
) -> tuple[
    dict[str, _RelativePhrase],
    dict[str, tuple[str, ...]],
    tuple[tuple[Any, str], ...],
    dict[str, tuple[str, ...]],
]:
    """Compile locale dictionaries once; query-time work stays CPU-cheap."""
    locale_map = default_loader.get_locale_map(
        languages=list(languages) or None,
        use_given_order=bool(languages),
        allow_conflicting_locales=True,
    )
    phrases: dict[str, _RelativePhrase] = {}
    numeric_patterns: list[tuple[Any, str]] = []
    absolute_tokens: dict[str, list[str]] = {}
    for locale_name, locale in locale_map.items():
        language = locale_name.split("-", 1)[0]
        info = locale.info
        for canonical, surfaces in (info.get("relative-type") or {}).items():
            for surface in surfaces:
                normalized = _normalize(surface)
                if normalized:
                    phrases.setdefault(normalized, _RelativePhrase(canonical))
        for canonical, patterns in (info.get("relative-type-regex") or {}).items():
            for pattern in patterns:
                try:
                    numeric_patterns.append(
                        (
                            regex.compile(
                                rf"(?<!\w)(?:{pattern})(?!\w)", regex.IGNORECASE
                            ),
                            canonical,
                        )
                    )
                except regex.error:
                    continue
        for key in (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ):
            for token in info.get(key, ()):
                normalized = _normalize(token)
                if len(normalized) >= 3:
                    absolute_tokens.setdefault(normalized, []).append(language)

    phrases_by_first: dict[str, list[str]] = {}
    for phrase in sorted(phrases, key=len, reverse=True):
        phrases_by_first.setdefault(phrase[0], []).append(phrase)
    return (
        phrases,
        {key: tuple(values) for key, values in phrases_by_first.items()},
        tuple(numeric_patterns),
        {
            token: tuple(dict.fromkeys(langs))
            for token, langs in absolute_tokens.items()
        },
    )


class DateParserTemporalResolutionEngine(TemporalResolutionEngine):
    """Fast multilingual resolver with no model or embedding round-trip.

    Relative expressions are matched against dateparser's locale dictionaries
    (a broad 30-language set by default, or every locale with ``["*"]``).
    Absolute dates are parsed only when the query contains a date-shaped
    token, keeping the common no-date path cheap and avoiding the false
    positives of blind date search.
    """

    name = "dateparser"

    def __init__(self, languages: list[str] | tuple[str, ...] | None = None) -> None:
        requested = tuple(dict.fromkeys(languages or DEFAULT_FAST_LANGUAGES))
        self.languages = () if requested == ("*",) else requested
        (
            self._phrases,
            self._phrases_by_first,
            self._numeric_patterns,
            self._absolute_tokens,
        ) = _language_data(self.languages)

    @staticmethod
    def _grain(canonical: str) -> str:
        for grain in ("second", "minute", "hour", "day", "week", "month", "year"):
            if grain in canonical:
                return grain
        return "unknown"

    @staticmethod
    def _calendar_range(dt: datetime, grain: str) -> tuple[datetime, datetime]:
        if grain == "year":
            start = dt.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            return start, start.replace(year=start.year + 1)
        if grain == "month":
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            year, month = (
                (start.year + 1, 1)
                if start.month == 12
                else (start.year, start.month + 1)
            )
            return start, start.replace(year=year, month=month)
        if grain == "week":
            start = (dt - timedelta(days=dt.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, start + timedelta(days=7)
        if grain == "day":
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, start + timedelta(days=1)
        if grain == "hour":
            start = dt.replace(minute=0, second=0, microsecond=0)
            return start, start + timedelta(hours=1)
        if grain == "minute":
            start = dt.replace(second=0, microsecond=0)
            return start, start + timedelta(minutes=1)
        start = dt.replace(microsecond=0)
        return start, start + timedelta(seconds=1)

    def _relative_range(
        self,
        expression: str,
        canonical: str,
        reference: datetime,
        timezone_name: str,
    ) -> TemporalRange | None:
        local_reference = reference.astimezone(_zone(timezone_name))
        canonical = canonical.replace("\\1", "1")
        match = regex.fullmatch(
            r"(?:(in)\s+)?(\d+(?:[.,]\d+)?)\s+"
            r"(second|minute|hour|day|week|month|year)s?(?:\s+(ago))?",
            canonical,
            regex.IGNORECASE,
        )
        if match is None:
            return None
        amount = float(match.group(2).replace(",", "."))
        direction = 1.0 if match.group(1) else -1.0 if match.group(4) else 0.0
        unit = match.group(3).lower()
        if direction == 0.0 or amount == 0.0:
            parsed = local_reference
        elif unit in {"month", "year"} and amount.is_integer():
            months = int(amount) * (12 if unit == "year" else 1)
            months = months if direction > 0 else -months
            total = local_reference.year * 12 + local_reference.month - 1 + months
            year, month0 = divmod(total, 12)
            month = month0 + 1
            day = min(local_reference.day, calendar.monthrange(year, month)[1])
            parsed = local_reference.replace(year=year, month=month, day=day)
        else:
            seconds = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
                "day": 86_400,
                "week": 604_800,
                "month": 2_629_746,
                "year": 31_556_952,
            }[unit]
            parsed = local_reference + timedelta(seconds=direction * amount * seconds)
        grain = self._grain(canonical)
        start, end = self._calendar_range(parsed, grain)
        return TemporalRange(start, end, expression=expression, grain=grain)

    def _absolute_ranges(
        self,
        text: str,
        reference: datetime,
        timezone_name: str,
    ) -> list[TemporalRange]:
        normalized = _normalize(text)
        date_shape = bool(
            regex.search(
                r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?)\b",
                normalized,
            )
        )
        candidate_languages: list[str] = []
        words = set(regex.findall(r"[\p{L}]+", normalized))
        for token in words:
            candidate_languages.extend(self._absolute_tokens.get(token, ()))
        if not date_shape and not (
            any(ch.isdigit() for ch in normalized) and candidate_languages
        ):
            return []
        languages = list(dict.fromkeys(candidate_languages))[:8]
        # Pure numeric/ISO dates are language-neutral. A single deterministic
        # locale is both faster and avoids cross-locale false positives (for
        # example ordinary two-letter words interpreted as dates).
        if not languages:
            languages = ["en"]
        settings = {
            "RELATIVE_BASE": reference.astimezone(_zone(timezone_name)),
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": timezone_name,
            "PREFER_DATES_FROM": "past",
        }
        try:
            found = (
                search_dates(text, languages=languages or None, settings=settings) or []
            )
        except (ValueError, TypeError, OverflowError):
            return []
        out: list[TemporalRange] = []
        for expression, parsed, *_ in found:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_zone(timezone_name))
            grain = "day"
            if regex.search(r"\b\d{1,2}:\d{2}\b", expression):
                grain = "minute"
            elif regex.search(r"\b\d{4}\b", expression) and not regex.search(
                r"[-/.]", expression
            ):
                grain = "year"
            elif any(
                token in _normalize(expression) for token in self._absolute_tokens
            ):
                grain = "day"
            start, end = self._calendar_range(parsed, grain)
            out.append(
                TemporalRange(
                    start, end, expression=expression, grain=grain, confidence=0.9
                )
            )
        return out

    def _resolve_one(
        self,
        text: str,
        weight: float,
        reference: datetime,
        timezone_name: str,
    ) -> list[TemporalRange]:
        normalized = _normalize(text)
        ranges: list[TemporalRange] = []
        occupied: list[tuple[int, int]] = []
        for first in set(normalized):
            for phrase in self._phrases_by_first.get(first, ()):
                start = normalized.find(phrase)
                if start < 0:
                    continue
                end = start + len(phrase)
                # Latin tokens need word boundaries; scripts without spaces
                # (Japanese/Chinese/Thai) are safely matched as substrings.
                if phrase[0].isascii() and phrase[0].isalnum():
                    if start > 0 and normalized[start - 1].isalnum():
                        continue
                    if end < len(normalized) and normalized[end].isalnum():
                        continue
                spec = self._phrases[phrase]
                resolved = self._relative_range(
                    phrase, spec.canonical, reference, timezone_name
                )
                if resolved is not None:
                    ranges.append(
                        TemporalRange(
                            resolved.start,
                            resolved.end,
                            expression=phrase,
                            grain=resolved.grain,
                            confidence=resolved.confidence,
                            query_weight=weight,
                        )
                    )
                    occupied.append((start, end))

        # Numeric relatives ("3 days ago", "tra 2 settimane", ...) are less
        # common, so they use precompiled locale patterns after the one-pass
        # fixed-phrase matcher. Stop after the first non-overlapping match.
        if any(ch.isdigit() for ch in normalized):
            for pattern, canonical in self._numeric_patterns:
                match = pattern.search(normalized)
                if match is None or any(
                    match.start() < end and match.end() > start
                    for start, end in occupied
                ):
                    continue
                expression = match.group(0)
                concrete = canonical.replace(
                    "\\1", match.group(1) if match.lastindex else "1"
                )
                resolved = self._relative_range(
                    expression, concrete, reference, timezone_name
                )
                if resolved is not None:
                    ranges.append(
                        TemporalRange(
                            resolved.start,
                            resolved.end,
                            expression=resolved.expression,
                            grain=resolved.grain,
                            confidence=resolved.confidence,
                            query_weight=weight,
                        )
                    )
                    occupied.append(match.span())
                    break

        for resolved in self._absolute_ranges(text, reference, timezone_name):
            ranges.append(
                TemporalRange(
                    resolved.start,
                    resolved.end,
                    expression=resolved.expression,
                    grain=resolved.grain,
                    confidence=resolved.confidence,
                    query_weight=weight,
                )
            )
        return ranges

    def resolve(
        self,
        message_or_history: Query,
        *,
        reference_time: datetime | float | int | str | None = None,
        timezone: str = "UTC",
    ) -> TemporalResolution:
        reference = coerce_datetime(reference_time)
        best_by_interval: dict[tuple[float, float, str], TemporalRange] = {}
        for text, weight in as_queries(message_or_history):
            for resolved in self._resolve_one(text, weight, reference, timezone):
                key = (
                    resolved.start_timestamp,
                    resolved.end_timestamp,
                    resolved.grain,
                )
                current = best_by_interval.get(key)
                current_strength = (
                    current.confidence * current.query_weight if current else -1.0
                )
                strength = resolved.confidence * resolved.query_weight
                if current is None or (strength, len(resolved.expression)) > (
                    current_strength,
                    len(current.expression),
                ):
                    best_by_interval[key] = resolved
        return TemporalResolution(
            tuple(best_by_interval.values()), reference, timezone, self.name
        )
