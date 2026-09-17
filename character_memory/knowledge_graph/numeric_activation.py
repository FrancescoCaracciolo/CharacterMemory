"""Query-local numeric spreading activation.

Builds a sparse directed walk (source/destination/strength/fan) from the
live graph, then propagates with float64 NumPy indexed reductions. The
snapshot is discarded at the end of the call; nothing is cached across
queries. Positions are temporary integers, distinct from durable node IDs.

This module is a private backend for :func:`spread_activation`. Custom
``KnowledgeGraph`` subclasses keep the scalar walker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Optional

import numpy as np

from .edges import ChatEdge, CoOccurrenceEdge, SYMMETRIC_KINDS
from .graph import KnowledgeGraph

_EMPTY_INT = np.empty(0, dtype=np.int64)
_EMPTY_FLOAT = np.empty(0, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NumericAdjacency:
    """Sparse directed walks for one query, privacy-filtered."""

    node_ids: list[str]
    index: dict[str, int]
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    fan: np.ndarray


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

    def emit(src_id: str, dst_id: str, edge_strength: float) -> None:
        src_i = index.get(src_id)
        dst_i = index.get(dst_id)
        if src_i is None or dst_i is None:
            return
        append_src(src_i)
        append_dst(dst_i)
        append_w(edge_strength * node_strengths[dst_i])

    for edge, edge_strength in zip(edges, edge_strengths):
        src_id = edge.src
        dst_id = edge.dst
        if not _include_node(src_id, allowed_node_ids) and not _include_node(
            dst_id, allowed_node_ids
        ):
            continue
        strength = float(edge_strength)
        symmetric = edge.kind in SYMMETRIC_KINDS
        emit(src_id, dst_id, strength)
        if src_id == dst_id:
            # add_edge appends the same id onto one adj list twice.
            emit(src_id, dst_id, strength)
        elif symmetric:
            emit(dst_id, src_id, strength)

    if not srcs:
        fan = np.zeros(n, dtype=np.float64)
        return NumericAdjacency(
            node_ids=node_ids,
            index=index,
            src=_EMPTY_INT,
            dst=_EMPTY_INT,
            weight=_EMPTY_FLOAT,
            fan=fan,
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
    )


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
    """Anderson spreading on a query-local numeric adjacency."""
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
    """Build a query-local snapshot and propagate. Public retrieval is unchanged."""
    adj = build_numeric_adjacency(graph, allowed_node_ids)
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
    "propagate_numeric",
    "spread_activation_numeric",
]
