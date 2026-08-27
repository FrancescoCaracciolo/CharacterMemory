"""Strict helpers for emotion vectors.

Emotion vectors are sparse mappings of configured emotion-axis names to
non-negative intensities in ``[0, 1]``.  Scalars are deliberately rejected:
legacy scalar conversion belongs exclusively to the one-off upgrade script.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any


def emotion_vector(
    value: Any,
    *,
    allowed_axes: Iterable[str] | None = None,
    ignore_unknown_axes: bool = False,
) -> dict[str, float]:
    """Validate and clamp an emotion vector.

    Missing axes remain absent (a sparse vector).  When ``allowed_axes`` is
    provided, unknown names are rejected rather than silently discarded unless
    ``ignore_unknown_axes`` is explicitly enabled at an untrusted boundary such
    as LLM extraction. Public setters should retain the strict default.
    """
    if not isinstance(value, Mapping):
        raise TypeError("emotional_shift must be an object mapping emotion names to numbers")
    allowed = set(allowed_axes) if allowed_axes is not None else None
    out: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("emotion-vector keys must be non-empty strings")
        if allowed is not None and key not in allowed:
            if ignore_unknown_axes:
                continue
            raise ValueError(
                f"unknown emotion axis {key!r}; allowed axes: {', '.join(sorted(allowed))}"
            )
        if isinstance(raw_value, bool):
            raise TypeError(f"emotion intensity for {key!r} must be a number")
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"emotion intensity for {key!r} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"emotion intensity for {key!r} must be finite")
        out[key] = max(0.0, min(1.0, number))
    return out


def decode_emotion_vector(
    value: Any,
    *,
    allowed_axes: Iterable[str] | None = None,
) -> dict[str, float]:
    """Decode a stored JSON object and validate it as an emotion vector."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("stored emotional_shift is not valid JSON") from exc
    return emotion_vector(value, allowed_axes=allowed_axes)


def encode_emotion_vector(
    value: Any,
    *,
    allowed_axes: Iterable[str] | None = None,
) -> str:
    """Validate and encode a vector using a deterministic JSON representation."""
    return json.dumps(
        emotion_vector(value, allowed_axes=allowed_axes),
        ensure_ascii=False,
        sort_keys=True,
    )


def emotional_impact(value: Mapping[str, float]) -> float:
    """Clamped L2 magnitude of an emotion vector."""
    return min(1.0, math.sqrt(sum(float(v) ** 2 for v in value.values())))


def emotion_similarity(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    """Cosine similarity of two non-negative emotion vectors.

    Missing axes are zero.  Empty or zero vectors have similarity ``0``.
    """
    axes = set(left) | set(right)
    if not axes:
        return 0.0
    dot = sum(float(left.get(k, 0.0)) * float(right.get(k, 0.0)) for k in axes)
    left_norm = math.sqrt(sum(float(left.get(k, 0.0)) ** 2 for k in axes))
    right_norm = math.sqrt(sum(float(right.get(k, 0.0)) ** 2 for k in axes))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


__all__ = [
    "decode_emotion_vector",
    "emotion_similarity",
    "emotion_vector",
    "emotional_impact",
    "encode_emotion_vector",
]
