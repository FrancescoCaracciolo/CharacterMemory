"""Read-side adapters that turn each `Memory` into paged, searchable records.

The GUI never talks to a memory subclass directly. Instead it goes through a
`MemoryAdapter`, which normalises the wildly different back-ends — SQLite rows
for the structured memories, in-RAM chunk nodes for the RAG memories, the
per-user JSON blob for emotion — into one `MemoryRecord` shape the frontend can
render.

Binding is by **base class**, not by name: any new `StructuredMemory` /
`RAGMemory` / `EmotionStatus` subclass is rendered by the matching adapter for
free (matching the library's "one new subclass, nothing else changes" rule).
A per-name override hook (`register`) lets a specific memory type swap in a
tailored adapter without touching the rest.

The two public entry points are :func:`overview` (sidebar: every memory with a
count) and :func:`read_memory` (one memory, one page, optional search + user
filter). Search prefers the memory's own hybrid (semantic) retrieval and falls
back to lexical matching when the embedding server is unreachable, so the GUI
stays useful offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from character_memory import EmotionStatus, Memory, StructuredMemory
from character_memory.emotion_vectors import emotion_similarity, emotional_impact
from character_memory.memory.character_base import RAGMemory
from character_memory.memory.knowledge_graph_memory import KnowledgeGraphMemory

# Hard cap on how many hits search ever ranks, so a query against a huge memory
# stays snappy. Pagination slices within this ranked window.
SEARCH_CAP = 200


# --------------------------------------------------------------------------- #
# Normalised shapes the API returns.
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    """One row/chunk/state, normalised for the frontend.

    `text` is a default display string; `fields` carries the raw, type-specific
    columns so a custom per-memory renderer can show whatever it wants. `score`
    is the search relevance (search mode) or effective-importance (browse mode);
    `meta` holds decay/recency bookkeeping for the structured memories.
    """

    id: Any
    user_id: Optional[str]
    text: str
    score: Optional[float]
    fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter base + builtins.
# --------------------------------------------------------------------------- #
class MemoryAdapter:
    """Normalise one memory backend into paged/searchable records."""

    kind = "generic"

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    @property
    def title(self) -> str:
        return self.memory.title

    def count(self, user_id: Optional[str] = None) -> int:
        return 0

    def users(self) -> list[str]:
        return []

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        raise NotImplementedError

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        return [], 0


class StructuredAdapter(MemoryAdapter):
    """SQLite rows + hybrid recall (facts / directives / episodes / heartbeat)."""

    kind = "structured"

    @property
    def m(self) -> StructuredMemory:
        return self.memory  # type: ignore[return-type]

    def _where(self, user_id: Optional[str]) -> Optional[dict[str, Any]]:
        return {"user_id": user_id} if user_id else None

    def count(self, user_id: Optional[str] = None) -> int:
        stmt = f"SELECT COUNT(*) AS c FROM {self.m.table}"
        params: list[Any] = []
        if user_id:
            stmt += " WHERE user_id = ?"
            params.append(user_id)
        rows = self.m.store.execute(stmt, params)
        return int(rows[0]["c"]) if rows else 0

    def users(self) -> list[str]:
        rows = self.m.store.execute(
            f"SELECT DISTINCT user_id FROM {self.m.table} ORDER BY user_id"
        )
        return [r["user_id"] for r in rows]

    def _record(self, row: dict[str, Any], score: Optional[float] = None) -> MemoryRecord:
        # Use the memory's own effective-importance (episodic weights in emotion).
        eff = self._safe_effective(row)
        fields = dict(row)
        meta = {
            "effective": eff,
            "importance": row.get("importance"),
            "recall_count": row.get("recall_count"),
            "created_at": row.get("created_at"),
            "last_recalled": row.get("last_recalled"),
        }
        if self.m.name == "episodic":
            try:
                item = self.m.row_item(row, eff or 0.0)
                fields["emotional_shift"] = item.metadata["emotional_shift"]
                meta["raw_emotional_impact"] = item.metadata["raw_emotional_impact"]
                meta["emotion_similarity"] = item.metadata["emotion_similarity"]
                meta["impact"] = item.metadata["impact"]
                meta["similarity"] = item.metadata["similarity"]
            except Exception:
                pass
        return MemoryRecord(
            id=row.get("id"),
            user_id=row.get("user_id"),
            text=self.m.row_text(row),
            score=score if score is not None else eff,
            fields=fields,
            meta=meta,
        )

    def _safe_effective(self, row: dict[str, Any]) -> Optional[float]:
        try:
            return float(self.m._effective(row))
        except Exception:
            return None

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        total = self.count(user_id)
        rows = self.m.store.select(
            self.m.table,
            where=self._where(user_id),
            order_by="id DESC",
            limit=size,
            offset=(page - 1) * size,
        )
        return [self._record(r) for r in rows], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ranked = self._ranked_rows(q, user_id)
        total = len(ranked)
        start = (page - 1) * size
        return [self._record(r, score=s) for r, s in ranked[start : start + size]], total

    def _ranked_rows(self, q: str, user_id: Optional[str]) -> list[tuple[dict, float]]:
        q = (q or "").strip()
        if not q:
            return []
        where = self._where(user_id)
        # Prefer semantic + lexical hybrid recall (needs the embedding server).
        try:
            hits = self.m.hybrid.search(q, k=SEARCH_CAP, where=where)
        except Exception:
            hits = []
        if hits:
            rows_by_id = {r["id"]: r for r in self.m.store.select(self.m.table, where=where)}
            out: list[tuple[dict, float]] = []
            for h in hits:
                rid = h.metadata.get("id")
                row = rows_by_id.get(rid)
                if row:
                    out.append((row, float(h.metadata.get("similarity") or h.score)))
            if out:
                return out
        # Lexical fallback (embedding server down / no hits): substring + count.
        return self._lexical(q, user_id)

    def _lexical(self, q: str, user_id: Optional[str]) -> list[tuple[dict, float]]:
        ql = q.lower()
        rows = self.m.store.select(self.m.table, where=self._where(user_id))
        scored: list[tuple[float, dict]] = []
        for r in rows:
            hay = (self.m.row_text(r) or "").lower()
            if ql in hay:
                scored.append((float(hay.count(ql)), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(r, s) for s, r in scored]


class RAGAdapter(MemoryAdapter):
    """In-RAM chunk nodes (character_info / dialogue_style)."""

    kind = "rag"

    @property
    def m(self) -> RAGMemory:
        return self.memory  # type: ignore[return-type]

    def _docs(self):
        try:
            return self.m.hybrid.documents
        except Exception:
            return []

    def count(self, user_id: Optional[str] = None) -> int:
        try:
            return int(self.m.hybrid.count)
        except Exception:
            return len(self._docs())

    def users(self) -> list[str]:
        return []

    def _record(self, doc, idx: int, score: Optional[float] = None) -> MemoryRecord:
        meta = dict(getattr(doc, "metadata", None) or {})
        text = getattr(doc, "text", "") or ""
        return MemoryRecord(
            id=meta.get("_pos", idx),
            user_id=meta.get("user_id"),
            text=text,
            score=score,
            fields={"source": meta.get("source", ""), "metadata": meta},
            meta={"source": meta.get("source", ""), "pos": meta.get("_pos", idx)},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        docs = self._docs()
        total = len(docs)
        start = (page - 1) * size
        window = docs[start : start + size]
        return [self._record(d, start + i) for i, d in enumerate(window)], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        q = (q or "").strip()
        if not q:
            return [], 0
        try:
            hits = self.m.hybrid.search(q, k=SEARCH_CAP)
            ranked = [
                (h.text, float(h.score), dict(h.metadata or {}))
                for h in hits
            ]
        except Exception:
            ranked = []
        if not ranked:  # lexical fallback over the in-RAM nodes
            ql = q.lower()
            for i, d in enumerate(self._docs()):
                text = getattr(d, "text", "") or ""
                if ql in text.lower():
                    meta = dict(getattr(d, "metadata", None) or {})
                    meta.setdefault("_pos", i)
                    ranked.append((text, float(text.lower().count(ql)), meta))
        total = len(ranked)
        start = (page - 1) * size
        recs = [
            MemoryRecord(
                id=m.get("_pos", 0),
                user_id=m.get("user_id"),
                text=text,
                score=score,
                fields={"source": m.get("source", ""), "metadata": m},
                meta={"source": m.get("source", "")},
            )
            for text, score, m in ranked[start : start + size]
        ]
        return recs, total


class EmotionAdapter(MemoryAdapter):
    """Per-user emotion state blobs (+ baseline exposed via page extra)."""

    kind = "emotion"

    @property
    def m(self) -> EmotionStatus:
        return self.memory  # type: ignore[return-type]

    def _rows(self, user_id: Optional[str] = None) -> list[dict]:
        return self.m.store.select(
            self.m.table,
            where={"user_id": user_id} if user_id else None,
            order_by="user_id",
        )

    def count(self, user_id: Optional[str] = None) -> int:
        return len(self._rows(user_id))

    def users(self) -> list[str]:
        return [r["user_id"] for r in self._rows()]

    def baseline(self) -> dict[str, float]:
        try:
            return {k: float(v) for k, v in self.m.baseline.items()}
        except Exception:
            return {}

    def current_mood(self) -> dict[str, float]:
        try:
            return self.m.get_current_mood()
        except Exception:
            return {}

    def _record(self, row: dict) -> MemoryRecord:
        try:
            blob = json.loads(row.get("state") or "{}")
        except (TypeError, ValueError):
            blob = {}
        # The blob holds numeric dims plus a string `comment` (relationship
        # descriptor). Split them so the dims render as bars and the comment
        # surfaces as text.
        comment = ""
        state: dict[str, float] = {}
        for k, v in blob.items():
            if k == "comment":
                comment = str(v or "")
            else:
                try:
                    state[k] = float(v)
                except (TypeError, ValueError):
                    continue
        text = ", ".join(f"{k}={v:.2f}" for k, v in state.items())
        if comment:
            text = (text + " | " if text else "") + f"relationship: {comment}"
        return MemoryRecord(
            id=row.get("user_id"),
            user_id=row.get("user_id"),
            text=text,
            score=None,
            fields={"state": state, "comment": comment},
            meta={"comment": comment} if comment else {},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        rows = self.m.store.select(
            self.m.table,
            where={"user_id": user_id} if user_id else None,
            order_by="user_id",
            limit=size,
            offset=(page - 1) * size,
        )
        return [self._record(r) for r in rows], self.count(user_id)

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ql = (q or "").lower().strip()
        rows = [r for r in self._rows(user_id) if not ql or ql in r["user_id"].lower()]
        total = len(rows)
        start = (page - 1) * size
        return [self._record(r) for r in rows[start : start + size]], total


class KnowledgeGraphAdapter(MemoryAdapter):
    """Knowledge-graph memory: paged/searched nodes (graph view via /api/graph)."""

    kind = "graph"

    @property
    def m(self) -> KnowledgeGraphMemory:
        return self.memory  # type: ignore[return-type]

    def _nodes(self):
        try:
            return list(self.m.retriever.graph.nodes.values())
        except Exception:
            return []

    def count(self, user_id: Optional[str] = None) -> int:
        return len(self._nodes())

    def users(self) -> list[str]:
        try:
            return list(self.m.retriever._known_users)
        except Exception:
            return []

    def _record(self, node, score: Optional[float] = None) -> MemoryRecord:
        meta = {
            "node_id": getattr(node, "id", ""),
            "node_kind": getattr(node, "kind", ""),
            "created_at": getattr(node, "created_at", None),
            "last_recalled": getattr(node, "last_recalled", None),
            "recall_count": getattr(node, "recall_count", 0),
            "source": getattr(node, "source", ""),
            "activation": getattr(node, "activation", 0.0),
        }
        fields: dict[str, Any] = {}
        for k in ("name", "aliases", "user_id", "content", "type", "confidence",
                  "importance", "summary", "emotional_shift", "timestamp",
                  "participants", "kind_label", "baseline", "current_mood"):
            v = getattr(node, k, None)
            if v is not None:
                fields[k] = v
        shift = getattr(node, "emotional_shift", None)
        if isinstance(shift, dict):
            self_node = self.m.retriever.graph.nodes.get(self.m.retriever.graph.SELF_ID)
            current = getattr(self_node, "current_mood", {}) if self_node else {}
            meta["raw_emotional_impact"] = emotional_impact(shift)
            meta["emotion_similarity"] = emotion_similarity(shift, current or {})
            meta["impact"] = meta["raw_emotional_impact"]
            meta["similarity"] = meta["emotion_similarity"]
        return MemoryRecord(
            id=getattr(node, "id", None),
            user_id=getattr(node, "user_id", None),
            text=getattr(node, "text", "") or getattr(node, "id", ""),
            score=score if score is not None else float(getattr(node, "activation", 0.0)),
            fields=fields,
            meta=meta,
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        nodes = self._nodes()
        total = len(nodes)
        start = (page - 1) * size
        # Default browse order: by kind then activation-descending.
        kind_order = {"self": 0, "person": 1, "fact": 2, "episode": 3, "entity": 4}
        nodes = sorted(nodes, key=lambda n: (kind_order.get(n.kind, 9), -float(getattr(n, "activation", 0.0))))
        return [self._record(n) for n in nodes[start : start + size]], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        q = (q or "").strip()
        if not q:
            return self.page(page, size, user_id)
        # Run a read-only activation search so the results mirror what the
        # prompt would actually surface.
        try:
            items = self.m.retriever.retrieve(
                q, user_id=user_id, limit=SEARCH_CAP, state_changing=False
            )
        except Exception:
            items = []
        ranked = sorted(items, key=lambda it: -float(it.score or 0.0))
        total = len(ranked)
        start = (page - 1) * size
        recs: list[MemoryRecord] = []
        for it in ranked[start : start + size]:
            node = self.m.retriever.graph.nodes.get(it.metadata.get("node_id"))
            if node is not None:
                recs.append(self._record(node, score=float(it.score or 0.0)))
        return recs, total


class GenericAdapter(MemoryAdapter):
    """Fallback: anything exposing `get_memories()`."""

    kind = "generic"

    def _all(self) -> list:
        try:
            return list(self.memory.get_memories(0))
        except Exception:
            return []

    def count(self, user_id: Optional[str] = None) -> int:
        return len(self._all())

    def _record(self, item, idx: int, score: Optional[float] = None) -> MemoryRecord:
        meta = dict(getattr(item, "metadata", None) or {})
        return MemoryRecord(
            id=meta.get("id", idx),
            user_id=meta.get("user_id"),
            text=getattr(item, "text", "") or "",
            score=getattr(item, "score", None) if score is None else score,
            fields=meta,
            meta={"kind": getattr(item, "kind", "")},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        items = self._all()
        total = len(items)
        start = (page - 1) * size
        return [self._record(it, start + i) for i, it in enumerate(items[start : start + size])], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ql = (q or "").lower().strip()
        items = [it for it in self._all() if not ql or ql in (getattr(it, "text", "") or "").lower()]
        total = len(items)
        start = (page - 1) * size
        return [self._record(it, start + i) for i, it in enumerate(items[start : start + size])], total


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #
_BY_NAME: dict[str, type[MemoryAdapter]] = {}


def register(name: str, adapter_cls: type[MemoryAdapter]) -> None:
    """Override the adapter for one memory `name` (tailor a single type)."""
    _BY_NAME[name] = adapter_cls


def get_adapter(memory: Memory) -> MemoryAdapter:
    """Pick an adapter: explicit per-name override, else bind by base class."""
    cls = _BY_NAME.get(memory.name)
    if cls is not None:
        return cls(memory)
    if isinstance(memory, KnowledgeGraphMemory):
        return KnowledgeGraphAdapter(memory)
    if isinstance(memory, EmotionStatus):
        return EmotionAdapter(memory)
    if isinstance(memory, RAGMemory):
        return RAGAdapter(memory)
    if isinstance(memory, StructuredMemory):
        return StructuredAdapter(memory)
    return GenericAdapter(memory)


# --------------------------------------------------------------------------- #
# Public entry points used by the API.
# --------------------------------------------------------------------------- #
def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


def overview(agent) -> list[dict[str, Any]]:
    """One summary per memory (sidebar)."""
    out: list[dict[str, Any]] = []
    for name, mem in agent.memories.items():
        adapter = get_adapter(mem)
        out.append(
            {
                "name": name,
                "title": mem.title,
                "kind": adapter.kind,
                "enabled": bool(getattr(mem, "enabled", True)),
                "count": _safe(adapter.count, default=0) or 0,
                "users": _safe(adapter.users, default=[]) or [],
            }
        )
    return out


def read_memory(
    agent,
    name: str,
    *,
    page: int = 1,
    size: int = 25,
    user: Optional[str] = None,
    q: Optional[str] = None,
) -> dict[str, Any]:
    """One page of records for one memory, with optional search + user filter."""
    mem = agent.memories.get(name)
    if mem is None:
        raise KeyError(name)

    adapter = get_adapter(mem)
    page = max(1, int(page or 1))
    size = max(1, min(100, int(size or 25)))
    q = (q or "").strip()
    user = user or None

    if q:
        records, total = _safe(
            adapter.search, q, page, size, user, default=([], 0)
        ) or ([], 0)
        search = True
    else:
        records, total = _safe(
            adapter.page, page, size, user, default=([], 0)
        ) or ([], 0)
        search = False

    pages = max(1, math.ceil(total / size)) if size else 1
    extra: dict[str, Any] = {}
    if isinstance(adapter, EmotionAdapter):
        extra["baseline"] = adapter.baseline()
        extra["current_mood"] = adapter.current_mood()

    return {
        "character": agent.character_name,
        "memory": name,
        "title": mem.title,
        "kind": adapter.kind,
        "enabled": bool(getattr(mem, "enabled", True)),
        "page": page,
        "size": size,
        "total": total,
        "pages": pages,
        "search": search,
        "query": q,
        "user": user,
        "users": _safe(adapter.users, default=[]) or [],
        "records": [asdict(r) for r in records],
        "extra": extra,
    }


def read_graph(
    agent,
    *,
    q: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 50,
    hops_subgraph: int = 1,
    max_edges: int = 400,
    include_co_occurrence: bool = False,
    full: bool = False,
    retrieve_k: int = 30,
) -> dict[str, Any]:
    """Return the activation-weighted knowledge-graph for the viz.

    Two modes:

    * ``full=False`` (default) — the activation-weighted **subgraph**: keeps the
      top ``limit`` nodes by activation, growing a connected neighbourhood by
      ``hops_subgraph`` hops within the node budget so a single high-degree node
      can't pull in the whole graph. Edges are capped at ``max_edges`` and
      ``co_occurrence`` edges are dropped by default (visually noisy); set
      ``include_co_occurrence=True`` to include them.
    * ``full=True`` — returns (almost) the **entire** graph. Every node carries a
      ``retrieved`` flag marking whether it landed in the top ``retrieve_k`` by
      activation for the query. The GUI uses this to light up the retrieved
      nodes and gray out the rest while still drawing all the links.

    Each node carries both the raw ``activation`` (signed, ACT-R + spreading)
    and a ``activation_norm`` in ``[0, 1]`` mapped from the returned graph's
    min/max so the GUI can size/opacity nodes without assuming a positive
    range (ACT-R base-level activations are routinely negative for old or
    low-recall nodes).
    """
    mem = agent.memories.get("knowledge_graph")
    if not isinstance(mem, KnowledgeGraphMemory):
        raise KeyError("knowledge_graph")
    retriever = mem.retriever
    full = bool(full)
    if full:
        node_budget = max(1, min(6000, int(limit or 6000)))
        edge_budget = max(0, min(12000, int(max_edges or 10000)))
        # In full mode the whole graph is the point, so surface co-occurrence
        # links unless the caller explicitly opts out.
        include_co = bool(include_co_occurrence) if include_co_occurrence else True
    else:
        node_budget = max(1, min(200, int(limit or 50)))
        edge_budget = max(0, min(2000, int(max_edges or 0)))
        # Always include co_occurrence edges in the subgraph: many characters'
        # knowledge graphs are co_occurrence-dominant, and dropping them would
        # leave a graph of isolated dots with no visible structure.
        include_co = True

    # Compute activations (read-only).
    if q and q.strip():
        trace = retriever.test_activation(q.strip(), user_id=user)
    else:
        trace = retriever.test_activation("", user_id=user)

    if full:
        # Keep the whole graph (capped); the top-`retrieve_k` by activation are
        # flagged "retrieved" so the GUI can highlight them.
        ranked = sorted(trace.items(), key=lambda kv: kv[1], reverse=True)
        k = max(1, min(len(ranked), int(retrieve_k or 30)))
        retrieved_ids = {nid for nid, _ in ranked[:k]}
        keep_ids = [nid for nid in retriever.graph.nodes if nid in trace][:node_budget]
        keep_set = set(keep_ids)
    else:
        # Sort all nodes by activation desc; take the top slice as the seed set.
        ranked = sorted(trace.items(), key=lambda kv: kv[1], reverse=True)
        seed_ids: list[str] = [nid for nid, _act in ranked[: max(1, node_budget // 3)]]
        seed_set: set[str] = set(seed_ids)
        # Grow a *connected* subgraph from the seeds: repeatedly add the
        # neighbour (of anything already kept) that adds the most edges to the
        # kept set, breaking ties by activation. This yields an edge-rich,
        # connected viz rather than a bag of isolated high-activation nodes.
        keep_ids = list(seed_ids)
        keep_set = set(seed_set)
        if hops_subgraph > 0:
            def edge_yield(nid: str) -> int:
                return sum(
                    1 for _e, nb in retriever.graph.neighbors(nid) if nb.id in keep_set
                )

            while len(keep_set) < node_budget:
                cand: dict[str, float] = {}
                for nid in list(keep_set):
                    for _edge, neighbour in retriever.graph.neighbors(nid):
                        if neighbour.id not in keep_set:
                            cand[neighbour.id] = trace.get(neighbour.id, 0.0)
                if not cand:
                    break
                best_nid = max(cand, key=lambda nid: (edge_yield(nid), cand[nid]))
                keep_set.add(best_nid)
                keep_ids.append(best_nid)
        # In subgraph mode everything returned is "retrieved".
        retrieved_ids = set(keep_ids)

    # Activation normalization across the returned graph (not the whole graph)
    # so the GUI's radius/opacity math is robust to negative BLL.
    sub_acts = [trace.get(nid, 0.0) for nid in keep_ids]
    a_min = min(sub_acts) if sub_acts else 0.0
    a_max = max(sub_acts) if sub_acts else 0.0
    a_span = (a_max - a_min) or 1.0

    nodes = []
    for nid in keep_ids:
        node = retriever.graph.nodes.get(nid)
        if node is None:
            continue
        raw = float(trace.get(node.id, 0.0))
        d = {
            "id": node.id,
            "kind": node.kind,
            "text": node.text or node.id,
            "activation": raw,
            "activation_norm": float((raw - a_min) / a_span),
            "retrieved": node.id in retrieved_ids,
        }
        for k in ("name", "user_id", "aliases", "content", "type", "confidence",
                  "importance", "summary", "emotional_shift", "timestamp",
                  "kind_label", "baseline", "current_mood", "comment"):
            v = getattr(node, k, None)
            if v is not None and v != "":
                d[k] = v
        shift = getattr(node, "emotional_shift", None)
        if isinstance(shift, dict):
            self_node = retriever.graph.nodes.get(retriever.graph.SELF_ID)
            current = getattr(self_node, "current_mood", {}) if self_node else {}
            d["raw_emotional_impact"] = emotional_impact(shift)
            d["emotion_similarity"] = emotion_similarity(shift, current or {})
            d["impact"] = d["raw_emotional_impact"]
            d["similarity"] = d["emotion_similarity"]
        nodes.append(d)
    # Edges: only between kept nodes, skip co_occurrence unless asked, cap the
    # count. Prefer higher-weight / structural edges over weak ones.
    edge_objs = [
        e for e in retriever.graph.edges.values()
        if e.src in keep_set and e.dst in keep_set
        and (include_co or e.kind != "co_occurrence")
    ]
    edge_objs.sort(key=lambda e: float(e.weight or 0.0), reverse=True)
    edge_objs = edge_objs[:edge_budget]
    edges = [
        {"id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst, "weight": float(e.weight)}
        for e in edge_objs
    ]
    self_node = retriever.graph.SELF_ID if retriever.graph.SELF_ID in retriever.graph.nodes else None
    return {
        "character": agent.character_name,
        "query": q or "",
        "user": user,
        "mode": "full" if full else "subgraph",
        "self_node": self_node,
        "nodes": nodes,
        "edges": edges,
        "activation_range": {"min": a_min, "max": a_max},
        "truncated": len(retriever.graph.nodes) > len(nodes) or len(retriever.graph.edges) > len(edges),
        "overview": retriever.overview(),
    }
