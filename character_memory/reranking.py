"""Global memory selection interfaces, independent of retrieval backends."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from .memory.base import MemoryItem
    from .rag.base import Query


class _Unset(Enum):
    DEFAULT = "inherit"


UNSET = _Unset.DEFAULT
Budget = int | None | _Unset
TokenCounter = Callable[[str], int]


def validate_budget(budget: int | None) -> None:
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative integer or None")


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count ordinary text exactly with cl100k_base; never approximate."""
    return len(_encoding().encode(text, disallowed_special=()))


@dataclass(frozen=True)
class MemoryCandidate:
    """One selectable item; ``id`` is unique only within this recall.

    ``rendered_text`` and ``token_count`` include the standalone section's
    header/labels. The final cost is recomputed after grouping, since shared
    headers, numbering, and token boundaries make costs non-additive.
    ``item`` is None for an atomic, body-only custom recall result.
    """

    id: int
    memory_name: str
    item: MemoryItem | None
    rendered_text: str
    token_count: int

    @property
    def score(self) -> float:
        return self.item.score if self.item is not None else 0.0


class MemoryReranker(ABC):
    """Order/filter candidates; the character enforces the rendered budget.

    Return an ordered subset of the supplied candidates without changing the
    items. Returning a candidate twice or an unknown candidate is an error.
    Implementations should keep request-specific state local to this method.
    """

    @abstractmethod
    def rerank(
        self, query: Query, candidates: Sequence[MemoryCandidate], *, budget: int | None
    ) -> Sequence[MemoryCandidate]:
        """Return candidates in descending selection priority."""


class ScoreMemoryReranker(MemoryReranker):
    """Normalize positive scores per memory, then apply memory weights.

    This deterministic heuristic does not calibrate semantic relevance across
    backends. Ties retain retrieval order; nonpositive/nonfinite scores become
    zero. Weights default to 1 and must be finite and nonnegative.
    """

    def __init__(self, memory_weights: Mapping[str, float] | None = None) -> None:
        self.memory_weights = dict(memory_weights or {})
        for weight in self.memory_weights.values():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("memory weights must be finite and nonnegative")

    def rerank(
        self, query: Query, candidates: Sequence[MemoryCandidate], *, budget: int | None
    ) -> list[MemoryCandidate]:
        maxima: dict[str, float] = {}
        scores: dict[int, float] = {}
        for candidate in candidates:
            score = float(candidate.score)
            score = score if math.isfinite(score) and score > 0 else 0.0
            scores[candidate.id] = score
            maxima[candidate.memory_name] = max(maxima.get(candidate.memory_name, 0.0), score)
        return sorted(
            candidates,
            key=lambda c: (
                scores[c.id] / (maxima[c.memory_name] or 1.0)
                * self.memory_weights.get(c.memory_name, 1.0)
            ),
            reverse=True,
        )
