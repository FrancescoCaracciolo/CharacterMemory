"""Numeric spreading activation over a version-cached sparse snapshot.

The weighted adjacency (edge strength × destination node strength, one row
per directed walk) is a pure function of graph state — it does not depend on
the query. :func:`cached_numeric_adjacency` therefore keeps one unscoped
snapshot plus a small LRU of privacy-scoped variants on the graph, keyed by
the graph's mutation clock, rebuilding lazily after structural changes. The
Hebbian step bumps co-occurrence strengths in place on every state-changing
retrieval; those edges are notified via ``graph.touch_edge_strength`` and
patched row-by-row here instead of triggering a full rebuild, so routine
retrieval never pays the snapshot cost.

Positions are temporary integers, distinct from durable node IDs. This module
is a private backend for :func:`spread_activation`. Custom
``KnowledgeGraph`` subclasses keep the scalar walker. Cache reads happen
under the retriever's lock or on single-threaded callers; snapshots are built
locally and assigned atomically.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AbstractSet, Optional

import numpy as np

from .edges import ChatEdge, CoOccurrenceEdge, SYMMETRIC_KINDS
from .._timing import time_phase
from .graph import KnowledgeGraph

_EMPTY_INT = np.empty(0, dtype=np.int64)
_EMPTY_FLOAT = np.empty(0, dtype=np.float64)

# Scoped privacy variants kept per graph version. One per participant union
# covers group chats; more exotic viewer sets simply cycle the LRU.
_MAX_SCOPED_SNAPSHOTS = 8


@dataclass(frozen=True, slots=True)
class NumericAdjacency:
    """Sparse directed walks for one scope, privacy-filtered."""

    node_ids: list[str]
    index: dict[str, int]
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    fan: np.ndarray
    # Rows per co-occurrence edge id (self-loops occupy two rows). Only this
    # kind has its strength patched in place (Hebbian step); everything else
    # invalidates through the graph version and forces a rebuild.
    edge_rows: dict[str, tuple[int, ...]] = field(default_factory=dict)


def _include_node(node_id: str, allowed: Optional[AbstractSet[str]]) -> bool:
    return allowed is None or node_id in allowed


def _edge_strengths(edges) -> np.ndarray:
    """Vectorize Chat/CoOccurrence weights; richer types use the helpers."""
    from .activation import (
        _clamped_weight,
        _edge_strength,
        _has_attached_strength_attrs,
    )

    n = len(edges)
    strengths = np.empty(n, dtype=np.float64)
    chat_i: list[int] = []
    chat_w: list[float] = []
    co_i: list[int] = []
    co_weight: list[float] = []
    co_created: list[float] = []
    for i, edge in enumerate(edges):
        exact = type(edge)
        if exact is ChatEdge and not _has_attached_strength_attrs(edge):
            chat_i.append(i)
            chat_w.append(_clamped_weight(getattr(edge, "weight", 0.5)))
        elif exact is CoOccurrenceEdge and not _has_attached_strength_attrs(edge):
            co_i.append(i)
            co_weight.append(float(edge.weight))
            co_created.append(float(edge.creation_weight or 0.0))
        else:
            strengths[i] = _edge_strength(edge)
    if chat_i:
        strengths[np.asarray(chat_i, dtype=np.int64)] = np.asarray(
            chat_w, dtype=np.float64
        )
    if co_i:
        weight = np.asarray(co_weight, dtype=np.float64)
        created = np.asarray(co_created, dtype=np.float64)
        effective = np.minimum(weight, created + 0.10)
        strengths[np.asarray(co_i, dtype=np.int64)] = np.clip(effective, 0.0, 1.0)
    return strengths


def build_numeric_adjacency(
    graph: KnowledgeGraph,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
) -> NumericAdjacency:
    """Materialise directed walks matching :meth:`KnowledgeGraph.neighbors`.

    Private nodes are dropped before fan-out is counted. Dangling endpoints
    are omitted. Self-loops are emitted twice, matching ``add_edge`` writing
    the same id into ``_adj`` twice. Parallel edges stay separate rows.
    """
    from .activation import _node_strength

    if allowed_node_ids is None:
        node_ids = list(graph.nodes)
    else:
        node_ids = [nid for nid in allowed_node_ids if nid in graph.nodes]
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    edges = list(graph.edges.values())
    if n == 0 or not edges:
        fan = np.zeros(n, dtype=np.float64)
        return NumericAdjacency(
            node_ids=node_ids,
            index=index,
            src=_EMPTY_INT,
            dst=_EMPTY_INT,
            weight=_EMPTY_FLOAT,
            fan=fan,
        )

    node_strengths = np.empty(n, dtype=np.float64)
    for i, nid in enumerate(node_ids):
        node_strengths[i] = _node_strength(graph.nodes[nid])
    edge_strengths = _edge_strengths(edges)

    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[float] = []
    append_src = srcs.append
    append_dst = dsts.append
    append_w = weights.append

    def emit(src_id: str, dst_id: str, edge_strength: float) -> Optional[int]:
        src_i = index.get(src_id)
        dst_i = index.get(dst_id)
        if src_i is None or dst_i is None:
            return None
        append_src(src_i)
        append_dst(dst_i)
        append_w(edge_strength * node_strengths[dst_i])
        return len(srcs) - 1

    co_rows: dict[str, list[int]] = {}
    for edge, edge_strength in zip(edges, edge_strengths):
        src_id = edge.src
        dst_id = edge.dst
        if not _include_node(src_id, allowed_node_ids) and not _include_node(
            dst_id, allowed_node_ids
        ):
            continue
        strength = float(edge_strength)
        symmetric = edge.kind in SYMMETRIC_KINDS
        rows = co_rows.setdefault(edge.id, []) if edge.kind == "co_occurrence" else None
        row = emit(src_id, dst_id, strength)
        if rows is not None and row is not None:
            rows.append(row)
        if src_id == dst_id:
            # add_edge appends the same id onto one adj list twice.
            row = emit(src_id, dst_id, strength)
            if rows is not None and row is not None:
                rows.append(row)
        elif symmetric:
            row = emit(dst_id, src_id, strength)
            if rows is not None and row is not None:
                rows.append(row)

    edge_rows = {eid: tuple(rows) for eid, rows in co_rows.items()}
    if not srcs:
        fan = np.zeros(n, dtype=np.float64)
        return NumericAdjacency(
            node_ids=node_ids,
            index=index,
            src=_EMPTY_INT,
            dst=_EMPTY_INT,
            weight=_EMPTY_FLOAT,
            fan=fan,
            edge_rows=edge_rows,
        )

    src = np.asarray(srcs, dtype=np.int64)
    dst = np.asarray(dsts, dtype=np.int64)
    weight = np.asarray(weights, dtype=np.float64)
    fan = np.zeros(n, dtype=np.float64)
    np.add.at(fan, src, 1.0)
    return NumericAdjacency(
        node_ids=node_ids,
        index=index,
        src=src,
        dst=dst,
        weight=weight,
        fan=fan,
        edge_rows=edge_rows,
    )


class _NumericAdjacencyCache:
    """Version-keyed snapshot store attached to one graph instance."""

    __slots__ = ("version", "unscoped", "scoped")

    def __init__(self) -> None:
        self.version = -1
        self.unscoped: Optional[NumericAdjacency] = None
        self.scoped: "OrderedDict[frozenset[str], NumericAdjacency]" = OrderedDict()


def _patch_edge_strengths(
    graph: KnowledgeGraph,
    dirty: set[str],
    snapshots: list[NumericAdjacency],
) -> None:
    """Rewrite the rows of edges whose strengths changed in place.

    Destination node strengths are unaffected by the Hebbian step, so each
    patched row is exactly ``_edge_strength(edge) * _node_strength(dst)``,
    the same product the snapshot builder stores.
    """
    from .activation import _edge_strength, _node_strength

    nodes = graph.nodes
    edges = graph.edges
    for adj in snapshots:
        for eid in dirty:
            rows = adj.edge_rows.get(eid)
            if not rows:
                continue
            edge = edges.get(eid)
            if edge is None:
                continue
            strength = _edge_strength(edge)
            node_ids = adj.node_ids
            dst = adj.dst
            weight = adj.weight
            for row in rows:
                node = nodes.get(node_ids[int(dst[row])])
                if node is None:
                    continue
                weight[row] = strength * _node_strength(node)


def cached_numeric_adjacency(
    graph: KnowledgeGraph,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
) -> NumericAdjacency:
    """Return the snapshot for ``allowed_node_ids``, rebuilding only if stale.

    Keyed by :attr:`KnowledgeGraph.version`; pending in-place strength
    notifications are drained and applied as row patches to every live
    snapshot. Fresh builds read the live graph, so draining before building
    keeps newly built snapshots exact as well. Rebuilds are recorded as a
    ``kg_snapshot`` phase so the meter can distinguish "rebuilding every
    query" from genuinely slow warm propagation.
    """
    cache: Optional[_NumericAdjacencyCache] = getattr(graph, "_numeric_adj_cache", None)
    if cache is None:
        cache = _NumericAdjacencyCache()
        graph._numeric_adj_cache = cache
    if graph.version != cache.version:
        cache.version = graph.version
        cache.unscoped = None
        cache.scoped.clear()
    dirty = graph.take_strength_dirty_edges()
    if dirty:
        live = [
            adj
            for adj in (cache.unscoped, *cache.scoped.values())
            if adj is not None
        ]
        if live:
            _patch_edge_strengths(graph, dirty, live)
    if allowed_node_ids is None:
        if cache.unscoped is None:
            with time_phase("kg_snapshot"):
                cache.unscoped = build_numeric_adjacency(graph, None)
        return cache.unscoped
    key = (
        allowed_node_ids
        if isinstance(allowed_node_ids, frozenset)
        else frozenset(allowed_node_ids)
    )
    adj = cache.scoped.get(key)
    if adj is None:
        with time_phase("kg_snapshot"):
            adj = build_numeric_adjacency(graph, key)
        cache.scoped[key] = adj
        if len(cache.scoped) > _MAX_SCOPED_SNAPSHOTS:
            cache.scoped.popitem(last=False)
    else:
        cache.scoped.move_to_end(key)
    return adj


def propagate_numeric(
    adj: NumericAdjacency,
    seeds: dict[str, float],
    *,
    gain: float = 0.35,
    hops: int = 2,
    decay_per_hop: float = 0.6,
    floor: float = 0.0,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
) -> dict[str, float]:
    """Anderson spreading on a numeric adjacency. Touches nothing on ``adj``."""
    n = len(adj.node_ids)
    act = np.zeros(n, dtype=np.float64)
    out: dict[str, float] = {}
    for nid, raw in seeds.items():
        if allowed_node_ids is not None and nid not in allowed_node_ids:
            continue
        value = max(floor, float(raw))
        i = adj.index.get(nid)
        if i is None:
            out[nid] = value
            continue
        act[i] = value

    frontier = act.copy()
    src = adj.src
    dst = adj.dst
    weight = adj.weight
    fan = adj.fan
    hop_count = max(1, int(hops))
    if src.size and n:
        inv_fan = np.zeros(n, dtype=np.float64)
        positive = fan > 0.0
        inv_fan[positive] = 1.0 / fan[positive]
        scaled = weight * inv_fan[src]
        for hop in range(1, hop_count + 1):
            g = gain * (decay_per_hop ** (hop - 1))
            src_act = frontier[src]
            contrib = g * src_act * scaled
            mask = (src_act > 0.0) & (contrib > 0.0)
            next_frontier = np.zeros(n, dtype=np.float64)
            if not np.any(mask):
                break
            np.add.at(next_frontier, dst[mask], contrib[mask])
            act += next_frontier
            if not np.any(next_frontier > 0.0):
                break
            frontier = next_frontier

    for nid, i in adj.index.items():
        value = float(act[i])
        if nid in out:
            continue
        seeded = nid in seeds and (
            allowed_node_ids is None or nid in allowed_node_ids
        )
        if seeded or value > 0.0:
            out[nid] = value
    return out


def spread_activation_numeric(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    *,
    gain: float = 0.35,
    hops: int = 2,
    decay_per_hop: float = 0.6,
    floor: float = 0.0,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
) -> dict[str, float]:
    """Propagate over the version-cached snapshot for this scope."""
    adj = cached_numeric_adjacency(graph, allowed_node_ids)
    return propagate_numeric(
        adj,
        seeds,
        gain=gain,
        hops=hops,
        decay_per_hop=decay_per_hop,
        floor=floor,
        allowed_node_ids=allowed_node_ids,
    )


__all__ = [
    "NumericAdjacency",
    "build_numeric_adjacency",
    "cached_numeric_adjacency",
    "propagate_numeric",
    "spread_activation_numeric",
]
