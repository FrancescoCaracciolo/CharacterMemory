"""Temporal resolution: public types, built-ins, and registry."""

from .base import (
    TemporalMatch,
    TemporalRange,
    TemporalResolution,
    TemporalResolutionEngine,
)
from .dateparser_engine import DateParserTemporalResolutionEngine
from .llm_engine import LLMTemporalResolutionEngine
from .registry import (
    get_temporal_resolution_engine,
    register_temporal_resolution_engine,
)

register_temporal_resolution_engine("dateparser", DateParserTemporalResolutionEngine)
register_temporal_resolution_engine("fast", DateParserTemporalResolutionEngine)
register_temporal_resolution_engine("llm", LLMTemporalResolutionEngine)

__all__ = [
    "TemporalMatch",
    "TemporalRange",
    "TemporalResolution",
    "TemporalResolutionEngine",
    "DateParserTemporalResolutionEngine",
    "LLMTemporalResolutionEngine",
    "get_temporal_resolution_engine",
    "register_temporal_resolution_engine",
]
