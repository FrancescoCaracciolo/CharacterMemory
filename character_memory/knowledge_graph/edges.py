"""Knowledge-graph edge types.

Edges carry the relationship-specific data described in the design doc. All
edges are directed (`src` -> `dst`); `TransitionEdge` and `CoOccurrenceEdge`
are *conceptually* undirected and the graph treats them symmetrically when
walking neighbours, but they are still stored once with a canonical
`(src, dst)` ordering so (de)serialisation is unambiguous.

Like nodes, edges are plain dataclasses with `to_dict` / `from_dict`.
"""

from __future__ import annotations

import dataclasses as _dc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    """Base class — common shape shared by every edge kind.

    All fields carry defaults so subclasses can add their own defaulted
    fields without ordering issues (Python dataclasses forbid non-default
    fields after defaulted ones in inheritance chains).
    """

    id: str = ""
    kind: str = "edge"
    src: str = ""
    dst: str = ""
    #: Generic strength in [0, 1] used by the spreading-activation weight
    #: `w_uv`. Edge kinds with their own richer signals (confidence,
    #: importance, emotional_shift) fold those into `weight` at retrieval
    #: time; `weight` itself is the persisted aggregate.
    weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        base = {
            "id": self.id,
            "kind": self.kind,
            "src": self.src,
            "dst": self.dst,
            "weight": self.weight,
        }
        base.update(self._extra_fields())
        return base

    def _extra_fields(self) -> dict[str, Any]:
        return {}


@dataclass
class RelationEdge(Edge):
    """Character (Self) <-> Person.

    Carries the same per-user dims as `EmotionStatus` (`valence`, `trust`,
    `affection`, all signed in [-1, 1]) plus a free-form relationship
    `comment` (colleague / friend / …).
    """

    kind: str = "relation"
    valence: float = 0.0
    trust: float = 0.0
    affection: float = 0.0
    comment: str = ""

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "valence": self.valence,
            "trust": self.trust,
            "affection": self.affection,
            "comment": self.comment,
        }


@dataclass
class FactEdge(Edge):
    """subject (Person/Entity/Self) -> Fact.

    Carries the fact's `confidence`, `importance` and a `timestamp` (the
    source row's `created_at`) so the activation weight can weigh recency.
    """

    kind: str = "fact"
    confidence: float = 0.5
    importance: float = 0.5
    timestamp: float = 0.0

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "importance": self.importance,
            "timestamp": self.timestamp,
        }


@dataclass
class TransitionEdge(Edge):
    """Person <-> Person. Symmetric relationship between two people.

    The dims are the same shape as `RelationEdge` (signed [-1, 1]) but
    describe how the *character* perceives the relationship *between* the two
    people, not toward either of them individually.
    """

    kind: str = "transition"
    valence: float = 0.0
    trust: float = 0.0
    affection: float = 0.0
    comment: str = ""

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "valence": self.valence,
            "trust": self.trust,
            "affection": self.affection,
            "comment": self.comment,
        }


@dataclass
class EpisodeEdge(Edge):
    """Person <-> Episode. One person's participation in an episode.

    Carries the per-participation `timestamp`, `emotional_shift` (signed) and
    `importance`, plus a `recall` flag set when this episode has been
    surfaced to the prompt at least once.
    """

    kind: str = "episode"
    timestamp: float = 0.0
    emotional_shift: float = 0.0
    importance: float = 0.5
    recall: bool = False

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "emotional_shift": self.emotional_shift,
            "importance": self.importance,
            "recall": bool(self.recall),
        }


@dataclass
class CoOccurrenceEdge(Edge):
    """any <-> any. Created when two nodes are made together or recalled together.

    `co_create=True` marks edges created at ingestion time (two nodes that
    appeared in the same source batch). `co_recall_count` is incremented by
    the Hebbian step each time both endpoints activate above threshold
    during a recall — the Hebbian "cells that fire together wire together".
    """

    kind: str = "co_occurrence"
    co_create: bool = False
    co_recall_count: int = 0

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "co_create": bool(self.co_create),
            "co_recall_count": int(self.co_recall_count),
        }


EDGE_CLASSES: dict[str, type[Edge]] = {
    "relation": RelationEdge,
    "fact": FactEdge,
    "transition": TransitionEdge,
    "episode": EpisodeEdge,
    "co_occurrence": CoOccurrenceEdge,
}

#: Edge kinds the graph treats as undirected when walking neighbours. A
#: transition between A and B is stored once; both A and B see each other as
#: neighbours regardless of which is `src`.
SYMMETRIC_KINDS: set[str] = {"transition", "co_occurrence"}


def edge_from_dict(data: dict[str, Any]) -> Edge:
    """Deserialise an edge dict, dispatching on its `kind`."""
    kind = data.get("kind", "edge")
    cls = EDGE_CLASSES.get(kind, Edge)
    if not _dc.is_dataclass(cls):
        return cls(**data)  # type: ignore[return-value]
    valid = {f.name for f in _dc.fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in data.items() if k in valid}
    return cls(**kwargs)  # type: ignore[call-arg]


__all__ = [
    "Edge",
    "RelationEdge",
    "FactEdge",
    "TransitionEdge",
    "EpisodeEdge",
    "CoOccurrenceEdge",
    "EDGE_CLASSES",
    "SYMMETRIC_KINDS",
    "edge_from_dict",
]
