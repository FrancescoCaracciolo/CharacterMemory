"""In-memory knowledge graph: nodes + adjacency.

`KnowledgeGraph` owns the node dict and the edge list, plus the small amount
of structure needed to walk it (forward and reverse adjacency, so symmetric
edge kinds are cheap). It is pure in-memory; persistence (SQLite + the
node-text hybrid index) lives in :mod:`persistence`.

Stable node IDs:
- ``self`` (singular).
- ``person:<user_id>``.
- ``entity:<slug>``.
- ``fact:<n>`` / ``episode:<n>`` — auto-incremented integers scoped to their
  kind. The counters are part of the graph's state so re-ingestion does not
  collide with previously-assigned ids.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .edges import (
    SYMMETRIC_KINDS,
    CoOccurrenceEdge,
    Edge,
    edge_from_dict,
)
from .nodes import (
    Node,
    SelfNode,
    PersonNode,
    FactNode,
    EpisodeNode,
    EntityNode,
    node_from_dict,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase, strip, collapse non-alphanumerics to single hyphens."""
    return _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-") or "entity"


class KnowledgeGraph:
    """Container for the nodes + edges of one character's knowledge graph."""

    SELF_ID = "self"

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        # `adj[u]` = list of edge ids whose src or dst is u. Symmetric kinds
        # appear under both endpoints even when stored once.
        self._adj: dict[str, list[str]] = {}
        # Auto-increment counters per node kind (fact / episode / entity).
        self._counters: dict[str, int] = {"fact": 0, "episode": 0, "entity": 0}

    # ------------------------------------------------------------------ nodes
    def add_node(self, node: Node) -> Node:
        """Insert `node`, or return the existing one if `node.id` is present.

        The SelfNode and PersonNode are upserts by identity; auto-incremented
        ids are assigned by the caller (via :meth:`next_id`) so collision
        handling is explicit.
        """
        existing = self.nodes.get(node.id)
        if existing is not None:
            return existing
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])
        return node

    def upsert_node(self, node: Node) -> Node:
        """Insert or replace `node` by id (refresh in place if present)."""
        if node.id in self.nodes:
            # Preserve transient activation across a refresh.
            node.activation = self.nodes[node.id].activation
            self.nodes[node.id] = node
            return node
        return self.add_node(node)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and every edge that touched it."""
        if node_id not in self.nodes:
            return
        for eid in list(self._adj.get(node_id, [])):
            self.remove_edge(eid)
        self._adj.pop(node_id, None)
        del self.nodes[node_id]

    def ensure_self(self, baseline: Optional[dict] = None) -> SelfNode:
        """Return the singular SelfNode, creating it with `baseline` if absent."""
        node = self.nodes.get(self.SELF_ID)
        if isinstance(node, SelfNode):
            if baseline:
                node.baseline.update({k: float(v) for k, v in baseline.items()})
            return node
        import time as _time
        now = _time.time()
        self_node = SelfNode(
            id=self.SELF_ID,
            kind="self",
            text="self",
            baseline=dict(baseline or {}),
            created_at=now,
            practice_times=[now],
        )
        self.add_node(self_node)
        return self_node

    def ensure_person(self, user_id: str, *, name: str = "", aliases: Optional[Iterable[str]] = None) -> PersonNode:
        """Return the PersonNode for `user_id`, creating it if absent."""
        if not user_id:
            user_id = "unknown"
        nid = f"person:{user_id}"
        node = self.nodes.get(nid)
        if isinstance(node, PersonNode):
            if name:
                node.name = name
            if aliases:
                merged = list(dict.fromkeys([*node.aliases, *aliases]))
                node.aliases = merged
            return node
        import time as _time
        now = _time.time()
        person = PersonNode(
            id=nid,
            kind="person",
            user_id=user_id,
            name=name or user_id,
            aliases=list(aliases or []),
            text=self._person_text(name or user_id, aliases or []),
            created_at=now,
            practice_times=[now],
        )
        self.add_node(person)
        return person

    def ensure_person_by_key(
        self,
        key: str,
        *,
        name: str = "",
        aliases: Optional[Iterable[str]] = None,
    ) -> PersonNode:
        """Return the PersonNode for a canonical `key`, merging on name/alias.

        Used by the wiki ingest, where a person has no `user_id` but a stable
        canonical key chosen by the LLM (e.g. ``okabe``) plus a name and a
        list of aliases (``Rintaro Okabe``, ``Hououin Kyouma``). Resolution:

        1. ``person:<key>`` if that id already exists;
        2. otherwise an existing PersonNode whose ``name`` or any alias matches
           ``name``/``aliases`` case-insensitively — so a wiki character and a
           ``user_summary`` row for the same person collapse into one node
           even when their keys differ;
        3. otherwise a fresh ``person:<key>`` node.

        On a hit, ``name`` and ``aliases`` are merged into the existing node.
        """
        key = (key or "").strip().lower() or "unknown"
        nid = f"person:{key}"
        node = self.nodes.get(nid)
        if node is None or not isinstance(node, PersonNode):
            # Fall back to name/alias matching against existing persons.
            node = self._find_person_by_name(name, aliases or [])
        if isinstance(node, PersonNode):
            if name:
                node.name = name if not node.name else node.name
                # Keep the more formal (longer) name as the display name.
                if len(name) > len(node.name):
                    node.name = name
            if aliases:
                merged = list(dict.fromkeys([*node.aliases, *aliases]))
                node.aliases = merged
            # Rebuild the indexed text so new aliases are searchable.
            node.text = self._person_text(node.name, node.aliases)
            return node
        import time as _time
        now = _time.time()
        aliases_list = [a for a in (aliases or []) if a and a != name]
        person = PersonNode(
            id=nid,
            kind="person",
            user_id=key,
            name=name or key,
            aliases=aliases_list,
            text=self._person_text(name or key, aliases_list),
            created_at=now,
            practice_times=[now],
        )
        self.add_node(person)
        return person

    def _find_person_by_name(
        self, name: str, aliases: Iterable[str]
    ) -> Optional[PersonNode]:
        """Return an existing PersonNode matching `name`/`aliases` (case-insensitive)."""
        labels = {c.lower() for c in [name, *aliases] if c}
        if not labels:
            return None
        for node in self.nodes.values():
            if not isinstance(node, PersonNode):
                continue
            node_labels = {c.lower() for c in [node.name, *node.aliases] if c}
            if node_labels & labels:
                return node
        return None

    @staticmethod
    def _person_text(name: str, aliases: Iterable[str]) -> str:
        parts = [name] + [a for a in aliases if a and a != name]
        return ". ".join(parts[:6])

    def ensure_entity(self, name: str, *, kind_label: str = "thing") -> EntityNode:
        """Return the EntityNode for `name` (slug-keyed), creating it if absent."""
        nid = f"entity:{slugify(name)}"
        node = self.nodes.get(nid)
        if isinstance(node, EntityNode):
            if kind_label and kind_label != "thing":
                node.kind_label = kind_label
            return node
        import time as _time
        now = _time.time()
        ent = EntityNode(
            id=nid,
            kind="entity",
            name=name,
            kind_label=kind_label,
            text=name,
            created_at=now,
            practice_times=[now],
        )
        self.add_node(ent)
        return ent

    def next_id(self, kind: str) -> str:
        """Return the next auto-incremented id for `kind` ('fact' / 'episode')."""
        if kind not in self._counters:
            raise ValueError(f"Unknown auto-increment kind: {kind!r}")
        self._counters[kind] += 1
        return f"{kind}:{self._counters[kind]}"

    # ------------------------------------------------------------------ edges
    def add_edge(self, edge: Edge) -> Edge:
        """Insert `edge`. Symmetric kinds are indexed under both endpoints."""
        existing = self.edges.get(edge.id)
        if existing is not None:
            return existing
        self.edges[edge.id] = edge
        self._adj.setdefault(edge.src, []).append(edge.id)
        self._adj.setdefault(edge.dst, []).append(edge.id)
        return edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self.edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        for endpoint in (edge.src, edge.dst):
            lst = self._adj.get(endpoint, [])
            if edge_id in lst:
                lst.remove(edge_id)

    def edge_id(self, kind: str, src: str, dst: str) -> str:
        """Canonical edge id for a `(kind, src, dst)` triple.

        For symmetric kinds the pair is sorted so a transition A->B and B->A
        share an id and the edge is stored exactly once.
        """
        if kind in SYMMETRIC_KINDS:
            a, b = sorted([src, dst])
            return f"e:{kind}:{a}::{b}"
        return f"e:{kind}:{src}->{dst}"

    def upsert_edge(self, edge: Edge) -> Edge:
        """Insert or merge `edge` by its canonical id."""
        eid = self.edge_id(edge.kind, edge.src, edge.dst)
        edge.id = eid
        existing = self.edges.get(eid)
        if existing is None:
            return self.add_edge(edge)
        # Merge scalar strengths by max so re-ingestion strengthens rather
        # than overwrites; co-occurrence counts are summed by the caller.
        self._merge_edge(existing, edge)
        return existing

    @staticmethod
    def _merge_edge(existing: Edge, new: Edge) -> None:
        existing.weight = max(existing.weight, new.weight)
        for f in ("valence", "trust", "affection", "importance", "confidence"):
            if hasattr(existing, f) and hasattr(new, f):
                # For signed dims take the value with the larger magnitude so
                # a clear signal is not averaged into neutrality.
                ev, nv = getattr(existing, f), getattr(new, f)
                if abs(nv) > abs(ev):
                    setattr(existing, f, nv)
        if hasattr(existing, "comment") and hasattr(new, "comment") and new.comment:
            existing.comment = new.comment  # type: ignore[attr-defined]
        if isinstance(existing, CoOccurrenceEdge) and isinstance(new, CoOccurrenceEdge):
            existing.co_create = existing.co_create or new.co_create
            # Re-ingesting the same co-create pair must be idempotent: do NOT
            # accumulate co_recall_count here. Only the Hebbian step (a real
            # co-recall event) increments it; take the max for safety.
            existing.co_recall_count = max(existing.co_recall_count, new.co_recall_count)

    def get_edge_between(self, kind: str, src: str, dst: str) -> Optional[Edge]:
        return self.edges.get(self.edge_id(kind, src, dst))

    # ------------------------------------------------------------- navigation
    def neighbors(self, node_id: str) -> list[tuple[Edge, Node]]:
        """Return `[(edge, neighbour_node)]` for every edge touching `node_id`.

        Symmetric kinds (transition / co_occurrence) are walked from either
        endpoint; directed kinds are walked from `src` only, so activation
        spreads along the semantic direction (Person -> Fact, etc.) while
        still allowing undirected traversal of the symmetric edges.
        """
        out: list[tuple[Edge, Node]] = []
        for eid in self._adj.get(node_id, []):
            edge = self.edges.get(eid)
            if edge is None:
                continue
            if edge.kind in SYMMETRIC_KINDS:
                other_id = edge.dst if edge.src == node_id else edge.src
            else:
                # Directed: only walk out of `src`.
                if edge.src != node_id:
                    continue
                other_id = edge.dst
            other = self.nodes.get(other_id)
            if other is None:
                continue
            out.append((edge, other))
        return out

    def degree(self, node_id: str) -> int:
        """Out-degree (for directed kinds) + both endpoints (symmetric kinds).

        This is the `fan(u)` used by spreading activation.
        """
        return len(self.neighbors(node_id))

    # ------------------------------------------------------- co-occurrence API
    def add_co_occurrence(
        self,
        a: str,
        b: str,
        *,
        co_create: bool = False,
        co_recall_count: int = 0,
        weight: float = 0.1,
    ) -> Optional[CoOccurrenceEdge]:
        """Create or strengthen the co-occurrence edge between `a` and `b`.

        No-op when `a == b` or either endpoint is missing.
        """
        if a == b or a not in self.nodes or b not in self.nodes:
            return None
        edge = CoOccurrenceEdge(
            id="",  # filled in by upsert_edge
            kind="co_occurrence",
            src=a,
            dst=b,
            weight=weight,
            co_create=co_create,
            co_recall_count=co_recall_count,
        )
        merged = self.upsert_edge(edge)
        return merged if isinstance(merged, CoOccurrenceEdge) else None

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        g = cls()
        for n in data.get("nodes", []):
            g.add_node(node_from_dict(n))
        for e in data.get("edges", []):
            g.add_edge(edge_from_dict(e))
        for k, v in (data.get("counters") or {}).items():
            if k in g._counters:
                g._counters[k] = int(v)
        return g

    # ------------------------------------------------------------------ stats
    def __len__(self) -> int:
        return len(self.nodes)

    def counts(self) -> dict[str, int]:
        """Per-kind node counts + total edge count, for overviews."""
        per_kind: dict[str, int] = {}
        for n in self.nodes.values():
            per_kind[n.kind] = per_kind.get(n.kind, 0) + 1
        per_kind["__edges__"] = len(self.edges)
        return per_kind

    def nodes_of_kind(self, kind: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]


__all__ = ["KnowledgeGraph", "slugify"]
