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
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Optional

from ..emotion_vectors import emotion_vector
from .edges import (
    SYMMETRIC_KINDS,
    CoOccurrenceEdge,
    Edge,
    EpisodeEdge,
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
        # Transient observer used only while a bulk build is running. It is
        # deliberately not part of the serialized graph state.
        self._on_node_added: Optional[Callable[[Node], None]] = None
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
        if self._on_node_added is not None:
            self._on_node_added(node)
        return node

    @contextmanager
    def _observe_node_additions(
        self, callback: Optional[Callable[[Node], None]]
    ) -> Iterator[None]:
        """Temporarily notify ``callback`` after each real node insertion."""
        previous = self._on_node_added
        self._on_node_added = callback
        try:
            yield
        finally:
            self._on_node_added = previous

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

    def ensure_self(
        self,
        baseline: Optional[dict] = None,
        current_mood: Optional[dict] = None,
    ) -> SelfNode:
        """Return the singular SelfNode, creating it with `baseline` if absent."""
        node = self.nodes.get(self.SELF_ID)
        if isinstance(node, SelfNode):
            if baseline:
                node.baseline = emotion_vector({**node.baseline, **baseline})
            if current_mood is not None:
                node.current_mood = emotion_vector(
                    current_mood, allowed_axes=node.baseline or None
                )
            return node
        import time as _time
        now = _time.time()
        self_node = SelfNode(
            id=self.SELF_ID,
            kind="self",
            text="self",
            baseline=dict(baseline or {}),
            current_mood=dict(current_mood or {}),
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

    # ------------------------------------------------------------ person merging
    # Two reusable, LLM-free merge passes. Both operate purely on the in-memory
    # graph and rewire edges so no information is lost; the merged node keeps
    # the union of aliases and the more formal (longer) display name. They are
    # the deterministic backstop for the alias-splitting problem: even when the
    # extraction LLM slips a synonym through, these collapse it on the next
    # ingest. See :meth:`collapse_into_self` for the character-self case.
    def merge_duplicate_persons(self) -> list[tuple[str, list[str]]]:
        """Fold PersonNodes that refer to the same real person into one node.

        Two PersonNodes are considered the same person when their label sets
        ``{name, *aliases, user_id}`` intersect case-insensitively — so
        ``person:okabe {Rintaro Okabe}`` and ``person:123 {Okabe}`` collapse.
        For each such group the survivor is chosen preferring a real
        ``user_id`` (non-slug) key then the longest name; everyone else's
        aliases are merged in, their edges are rewired onto the survivor via
        :meth:`upsert_edge` (which merges strengths idempotently), and the
        duplicate nodes are removed.

        Returns ``[(survivor_id, [removed_ids]), ...]`` for logging.
        """
        persons = [n for n in self.nodes.values() if isinstance(n, PersonNode)]
        # Union-find over persons keyed by lowercased label intersection.
        parent: dict[str, str] = {p.id: p.id for p in persons}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        label_to_ids: dict[str, list[str]] = {}
        for p in persons:
            labels = {
                str(c).lower()
                for c in [p.name, p.user_id, *(p.aliases or [])]
                if c
            }
            for lab in labels:
                label_to_ids.setdefault(lab, []).append(p.id)
        for ids in label_to_ids.values():
            for other in ids[1:]:
                union(ids[0], other)

        groups: dict[str, list[str]] = {}
        for p in persons:
            groups.setdefault(find(p.id), []).append(p.id)

        report: list[tuple[str, list[str]]] = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            survivor = self._pick_person_survivor(members)
            removed = [m for m in members if m != survivor]
            if not removed:
                continue
            self._merge_person_nodes(survivor, removed)
            report.append((survivor, removed))
        return report

    def collapse_into_self(self, labels: Iterable[str]) -> list[str]:
        """Merge any PersonNode referring to the character into the SelfNode.

        ``labels`` is the character's name + aliases (any case). Any
        PersonNode whose ``{name, *aliases, user_id}`` intersects that set
        case-insensitively is folded into the singular ``self`` node: edges
        are rewired onto ``self`` (strengths merged via :meth:`upsert_edge`),
        the aliases are recorded on the SelfNode's text so it stays
        searchable, and the duplicate person nodes are removed.

        Returns the ids of the removed person nodes, for logging.
        """
        target = {str(c).lower() for c in labels if c}
        if not target:
            return []
        # Make sure the SelfNode exists; without it there is nothing to fold into.
        if not isinstance(self.nodes.get(self.SELF_ID), SelfNode):
            return []
        removed: list[str] = []
        absorbed_aliases: list[str] = []
        for node in list(self.nodes.values()):
            if not isinstance(node, PersonNode):
                continue
            node_labels = {
                str(c).lower()
                for c in [node.name, node.user_id, *(node.aliases or [])]
                if c
            }
            if not (node_labels & target):
                continue
            # Remember the labels so the SelfNode text can mention them.
            for c in [node.name, *(node.aliases or [])]:
                if c and c.lower() not in target:
                    absorbed_aliases.append(c)
            self._rewire_edges(node.id, self.SELF_ID)
            self.remove_node(node.id)
            removed.append(node.id)
        if removed or absorbed_aliases:
            self_node = self.nodes.get(self.SELF_ID)
            if isinstance(self_node, SelfNode):
                # Rebuild a searchable text from whatever labels we now know.
                seen = list(dict.fromkeys(
                    [*(self_node.text.split(". ") if self_node.text else []), *absorbed_aliases]
                ))
                self_node.text = ". ".join(s for s in seen if s and s.lower() != "the character") or "the character"
        return removed

    def _pick_person_survivor(self, member_ids: list[str]) -> str:
        """Choose the canonical survivor id from a group of duplicate persons.

        Prefer a node anchored to a **real user identity** (one whose
        ``user_id`` came from a ``user_summary`` row, not an LLM canonical
        key) so that subsequent ingests' ``ensure_person(user_id)`` calls
        hit the same node instead of re-splitting the person. The heuristic:
        real platform ids almost always contain a digit, underscore, or
        uppercase letter, whereas LLM wiki keys are clean lowercase words
        (``okabe``, ``rintaro``). Break ties by the longest display name so
        the most formal variant wins.
        """
        import re as _re

        _real_id = _re.compile(r"[0-9_A-Z]")

        def score(nid: str) -> tuple[int, int]:
            node = self.nodes.get(nid)
            uid = getattr(node, "user_id", "") or ""
            # A real external id contains a digit/underscore/uppercase letter;
            # a clean lowercase slug (LLM canonical key) does not.
            has_real_id = bool(uid) and bool(_real_id.search(uid))
            name_len = len(getattr(node, "name", "") or "")
            return (1 if has_real_id else 0, name_len)

        return max(member_ids, key=score)

    def _merge_person_nodes(self, survivor_id: str, removed_ids: list[str]) -> None:
        """Fold ``removed_ids`` into ``survivor_id``: merge aliases, rewire, drop."""
        survivor = self.nodes.get(survivor_id)
        if not isinstance(survivor, PersonNode):
            return
        for rid in removed_ids:
            dup = self.nodes.get(rid)
            if not isinstance(dup, PersonNode):
                continue
            # Keep the more formal (longer) display name; union the aliases.
            if dup.name and len(dup.name) > len(survivor.name or ""):
                if survivor.name:
                    survivor.aliases = list(dict.fromkeys([*survivor.aliases, survivor.name]))
                survivor.name = dup.name
            else:
                survivor.aliases = list(dict.fromkeys([*survivor.aliases, dup.name]))
            survivor.aliases = list(dict.fromkeys([*survivor.aliases, *(dup.aliases or [])]))
            survivor.aliases = [a for a in survivor.aliases if a and a != survivor.name]
            self._rewire_edges(rid, survivor_id)
            self.remove_node(rid)
        survivor.text = self._person_text(survivor.name, survivor.aliases)

    def _rewire_edges(self, old_id: str, new_id: str) -> None:
        """Move every edge touching ``old_id`` onto ``new_id`` and dedupe.

        Each affected edge is re-inserted via :meth:`upsert_edge` so the merge
        logic in :meth:`_merge_edge` combines strengths (max for scalars,
        larger magnitude for signed dims) instead of overwriting. Self-loops
        created by the rewire (both endpoints now ``new_id``) are dropped.
        """
        if old_id == new_id:
            return
        affected = [eid for eid in list(self._adj.get(old_id, []))]
        for eid in affected:
            edge = self.edges.get(eid)
            if edge is None:
                continue
            new_src = new_id if edge.src == old_id else edge.src
            new_dst = new_id if edge.dst == old_id else edge.dst
            if new_src == new_dst:
                # Would become a self-loop on the survivor; drop it.
                self.remove_edge(eid)
                continue
            # Drop the old edge then re-insert under the new canonical id so
            # upsert_edge can merge it with any existing edge between the same
            # (kind, src, dst) triple.
            self.remove_edge(eid)
            edge.src = new_src
            edge.dst = new_dst
            edge.id = ""
            self.upsert_edge(edge)

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
        if isinstance(existing, EpisodeEdge) and isinstance(new, EpisodeEdge):
            # The vector is source-of-truth data, not a monotonic strength.
            # Re-ingestion must refresh it when an episodic row is edited.
            existing.emotional_shift = dict(new.emotional_shift)

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
