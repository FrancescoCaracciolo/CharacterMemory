"""Knowledge-graph node types.

The graph is per-character and shared across users. Every node carries a
common set of bookkeeping fields (decay, recall statistics, ACT-R practice
times, a back-reference to the source memory row) plus a `text` field — the
string that is embedded into the node-text hybrid index and that retrieval
matches against.

Node classes are plain dataclasses with `to_dict` / `from_dict` so the whole
graph serialises to JSON for persistence. `activation` is a transient field
populated at retrieval time and never persisted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..emotion_vectors import emotion_vector


def _now(clock: Optional[Callable[[], float]] = None) -> float:
    return (clock or time.time)()


@dataclass
class Node:
    """Base class — common bookkeeping shared by every node kind."""

    id: str
    kind: str
    text: str = ""
    created_at: float = 0.0
    last_recalled: Optional[float] = None
    recall_count: int = 0
    #: Exact list of practice-event timestamps (creation + every recall). The
    #: ACT-R base-level learning formula sums over these exactly.
    practice_times: list[float] = field(default_factory=list)
    #: `"<memory_name>:<row_id>"` — back-reference to the source row, so a
    #: dedup report can be mapped to graph mutations. Empty for nodes not
    #: derived from a single source row (e.g. the SelfNode).
    source: str = ""
    #: Transient activation populated at retrieval time; never persisted.
    activation: float = 0.0

    # ---- persistence helpers ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (drops the transient field)."""
        base = {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "created_at": self.created_at,
            "last_recalled": self.last_recalled,
            "recall_count": self.recall_count,
            "practice_times": list(self.practice_times),
            "source": self.source,
        }
        # Dataclass-only extra fields (the subclasses' data).
        for k, v in self._extra_fields().items():
            base[k] = v
        return base

    def _extra_fields(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        return cls(**data)

    # ---- runtime helpers ----------------------------------------------------
    def touch(self, now: Optional[float] = None) -> None:
        """Record a practice event (used after a recall that surfaces this node)."""
        t = now if now is not None else _now()
        self.last_recalled = t
        self.recall_count += 1
        self.practice_times.append(t)


class _NodeMixin:
    """Shared shape helpers for the concrete subclasses below.

    Each concrete node defines its typed dataclass fields and overrides
    `_extra_fields` / `from_dict` so (de)serialisation round-trips. The
    `kind` class attribute is what gets stamped on `Node.kind`.
    """

    kind: str = "node"


@dataclass
class SelfNode(Node, _NodeMixin):
    """The singular node representing the character.

    Holds the character's resting emotional `baseline` (the same shape as
    `EmotionStatus.baseline`). Every other node is at most two hops from
    this one, and it is always seeded with activation at retrieval time.
    """

    kind: str = "self"
    baseline: dict[str, float] = field(default_factory=dict)
    current_mood: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.baseline = emotion_vector(self.baseline)
        if not self.current_mood:
            self.current_mood = dict(self.baseline)
        self.current_mood = emotion_vector(
            self.current_mood, allowed_axes=self.baseline or None
        )
        self.current_mood = {
            axis: float(self.current_mood.get(axis, 0.0)) for axis in self.baseline
        }

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "baseline": dict(self.baseline),
            "current_mood": dict(self.current_mood),
        }


@dataclass
class PersonNode(Node, _NodeMixin):
    """A person the character knows. Mirrors one `user_summary` row."""

    kind: str = "person"
    user_id: str = ""
    name: str = ""
    aliases: list[str] = field(default_factory=list)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "aliases": list(self.aliases),
        }


@dataclass
class FactNode(Node, _NodeMixin):
    """A fact about an entity in the graph. Mirrors one `user_facts` row."""

    kind: str = "fact"
    content: str = ""
    type: str = "general"
    confidence: float = 0.5
    importance: float = 0.5
    #: The chat this fact was learned in (NULL ⇒ legacy / single-user / wiki).
    #: Pairs facts and episodes of the same chat via `ChatEdge`.
    chat_id: Optional[str] = None

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "importance": self.importance,
            "chat_id": self.chat_id,
        }


@dataclass
class EpisodeNode(Node, _NodeMixin):
    """Something that happened. Mirrors one `episodic` row."""

    kind: str = "episode"
    summary: str = ""
    emotional_shift: dict[str, float] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: float = 0.0
    participants: list[str] = field(default_factory=list)
    #: The chat this episode was learned in (NULL ⇒ legacy / single-user / wiki).
    #: Pairs facts and episodes of the same chat via `ChatEdge`.
    chat_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.emotional_shift = emotion_vector(self.emotional_shift)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "emotional_shift": self.emotional_shift,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "participants": list(self.participants),
            "chat_id": self.chat_id,
        }


@dataclass
class EntityNode(Node, _NodeMixin):
    """A special entity (place / object / thing) extracted from facts.

    Entities are extracted from `user_facts` by the LLM during ingestion and
    linked to the facts that mention them. `kind_label` (place/object/…) is
    the extraction-time category; `Node.kind` stays `"entity"`.
    """

    kind: str = "entity"
    name: str = ""
    kind_label: str = "thing"

    def _extra_fields(self) -> dict[str, Any]:
        return {"name": self.name, "kind_label": self.kind_label}


# Registry used by `Node.from_dict` to dispatch on `kind` when loading.
NODE_CLASSES: dict[str, type[Node]] = {
    "self": SelfNode,
    "person": PersonNode,
    "fact": FactNode,
    "episode": EpisodeNode,
    "entity": EntityNode,
}


def node_from_dict(data: dict[str, Any]) -> Node:
    """Deserialise a node dict, dispatching on its `kind`."""
    kind = data.get("kind", "node")
    cls = NODE_CLASSES.get(kind, Node)
    # Only pass keys the constructor accepts: keep the common ones and the
    # subclass-specific ones the class declared; drop anything unknown.
    import dataclasses as _dc

    if not _dc.is_dataclass(cls):
        return cls(**data)  # type: ignore[return-value]
    valid = {f.name for f in _dc.fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in data.items() if k in valid}
    obj = cls(**kwargs)  # type: ignore[call-arg]
    # Transient activation is never stored, but tolerate a stale value.
    obj.activation = float(data.get("activation") or 0.0)
    return obj


__all__ = [
    "Node",
    "SelfNode",
    "PersonNode",
    "FactNode",
    "EpisodeNode",
    "EntityNode",
    "NODE_CLASSES",
    "node_from_dict",
]
