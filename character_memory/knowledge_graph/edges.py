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

from ..emotion_vectors import emotion_vector


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

    Carries the per-participation `timestamp`, sparse vector
    `emotional_shift` and `importance`, plus a `recall` flag set when this episode has been
    surfaced to the prompt at least once.
    """

    kind: str = "episode"
    timestamp: float = 0.0
    emotional_shift: dict[str, float] = field(default_factory=dict)
    importance: float = 0.5
    recall: bool = False

    def __post_init__(self) -> None:
        self.emotional_shift = emotion_vector(self.emotional_shift)

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
    the Hebbian step each time both endpoints activate above threshold.
    Prompt-driven weight growth saturates at 0.10 above creation weight.
    """

    kind: str = "co_occurrence"
    co_create: bool = False
    co_recall_count: int = 0
    # Weight before prompt-driven Hebbian reinforcement. New edges persist the
    # exact creation value; legacy edges infer the historical defaults.
    creation_weight: float | None = None

    def __post_init__(self) -> None:
        if self.creation_weight is None:
            # A directly-authored edge treats its supplied weight as semantic
            # creation strength. Legacy persistence inference happens in
            # edge_from_dict, where absence of the new field is observable.
            self.creation_weight = float(self.weight)

    def effective_weight(self, max_boost: float = 0.10) -> float:
        """Weight used by retrieval, clamped for legacy overgrown edges."""
        base = float(self.creation_weight or 0.0)
        return min(float(self.weight), base + max(0.0, float(max_boost)))

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "co_create": bool(self.co_create),
            "co_recall_count": int(self.co_recall_count),
            "creation_weight": float(self.creation_weight or 0.0),
        }


@dataclass
class ChatEdge(Edge):
    """any <-> any (facts + episodes learned in the same conversation).

    A fixed, low-weight bridge that ties together the facts and episodes a
    single chat produced, so recalling one can spread to the others of that
    conversation. It carries only the generic ``weight`` (no per-kind dims), so
    ``_edge_strength`` leaves it low; and the Hebbian co-recall step only grows
    ``co_occurrence`` edges, so a ``ChatEdge`` stays at its low weight — it is
    a stable "same chat" tag, not a learned association.
    """

    kind: str = "chat"


EDGE_CLASSES: dict[str, type[Edge]] = {
    "relation": RelationEdge,
    "fact": FactEdge,
    "transition": TransitionEdge,
    "episode": EpisodeEdge,
    "co_occurrence": CoOccurrenceEdge,
    "chat": ChatEdge,
}

#: Edge kinds the graph treats as undirected when walking neighbours. A
#: transition between A and B is stored once; both A and B see each other as
#: neighbours regardless of which is `src`.
SYMMETRIC_KINDS: set[str] = {"transition", "co_occurrence", "chat"}


def edge_from_dict(data: dict[str, Any]) -> Edge:
    """Deserialise an edge dict, dispatching on its `kind`."""
    kind = data.get("kind", "edge")
    cls = EDGE_CLASSES.get(kind, Edge)
    if not _dc.is_dataclass(cls):
        return cls(**data)  # type: ignore[return-value]
    valid = {f.name for f in _dc.fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in data.items() if k in valid}
    if cls is CoOccurrenceEdge and "creation_weight" not in kwargs:
        # Historical recall-created edges began at 0.05; ingestion-created
        # edges began at 0.15. Their persisted weight may already be overgrown.
        kwargs["creation_weight"] = 0.15 if kwargs.get("co_create") else 0.05
    return cls(**kwargs)  # type: ignore[call-arg]


__all__ = [
    "Edge",
    "RelationEdge",
    "FactEdge",
    "TransitionEdge",
    "EpisodeEdge",
    "CoOccurrenceEdge",
    "ChatEdge",
    "EDGE_CLASSES",
    "SYMMETRIC_KINDS",
    "edge_from_dict",
]
