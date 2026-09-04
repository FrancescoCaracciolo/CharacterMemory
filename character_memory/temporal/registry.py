"""Registry for third-party temporal-resolution engines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import TemporalResolutionEngine

TemporalEngineFactory = Callable[..., TemporalResolutionEngine]
_REGISTRY: dict[str, TemporalEngineFactory] = {}


def register_temporal_resolution_engine(
    name: str, factory: TemporalEngineFactory
) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Temporal-resolution engine name cannot be empty")
    _REGISTRY[key] = factory


def get_temporal_resolution_engine(
    name: str, **kwargs: Any
) -> TemporalResolutionEngine:
    key = str(name).strip().lower()
    try:
        factory = _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown temporal-resolution engine: {name!r}") from exc
    return factory(**kwargs)
