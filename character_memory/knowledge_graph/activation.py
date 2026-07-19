"""Activation functions: ACT-R base-level learning + spreading activation.

Two pieces, kept pure (no I/O, no globals) so they are easy to test and so
the retriever can compose them:

- :func:`base_level_activation` — the exact ACT-R base-level learning (BLL)
  formula from a node's `practice_times`. Practice events are the node's
  creation timestamp and every recall.
- :func:`spread_activation` — Anderson-style spreading over the graph, up to
  a configurable number of hops, weighted by edge strength and the fan-out
  of the source node.

The retrieval-time weight of an edge `w_uv` blends the edge's own `weight`
with the destination node's importance/confidence/emotion magnitude. Those
factors live on the concrete edge/node classes; this module reads them via
small accessor helpers so it stays decoupled from the dataclasses.
"""

from __future__ import annotations

import math
from typing import Optional

from .edges import Edge, EpisodeEdge, FactEdge, RelationEdge
from .graph import KnowledgeGraph
from .nodes import Node


def base_level_activation(
    node: Node,
    *,
    now: float,
    decay: float = 0.5,
    decay_half_life: float = 0.0,
) -> float:
    """ACT-R base-level learning: ``B = ln(Σ_j t_j^-d)``.

    `t_j` is the elapsed time since practice event `j`. The formula is
    evaluated exactly from `node.practice_times`. When there are no practice
    events the node gets a small floor so it can still be activated by
    spreading. A secondary `decay_half_life` (seconds, optional) adds an
    exponential recency factor on top, mirroring the library's existing
    decay model so a stale-but-practiced node still fades between practices.
    """
    if not node.practice_times:
        return -2.0
    total = 0.0
    for t in node.practice_times:
        dt = max(1e-3, now - float(t))
        total += math.pow(dt, -decay)
    if total <= 0.0:
        return -2.0
    bll = math.log(total)
    # Recency nudges: a node practiced long ago and never since fades a bit.
    if decay_half_life and decay_half_life > 0.0:
        last = max(node.practice_times)
        age = max(0.0, now - last)
        bll *= math.exp(-age / (decay_half_life * 2.0))
    return bll


def _node_strength(node: Node) -> float:
    """How strongly a destination node pulls activation, in [0, 1]-ish.

    Combines importance/confidence/emotion-magnitude where present. Used as
    the `v` side of `w_uv`.
    """
    s = 0.4  # floor
    for attr, scale in (
        ("importance", 0.6),
        ("confidence", 0.4),
    ):
        v = getattr(node, attr, None)
        if isinstance(v, (int, float)):
            s = max(s, float(v) * scale + 0.2)
    # Episodic / emotional magnitude boosts retention (mirrors decay.py).
    shift = getattr(node, "emotional_shift", None)
    if isinstance(shift, (int, float)):
        s += 0.2 * abs(float(shift))
    return min(1.5, s)


def _edge_strength(edge: Edge) -> float:
    """The `u->v` edge weight in [0, 1]."""
    base = float(getattr(edge, "weight", 0.5) or 0.5)
    base = max(0.0, min(1.0, base))
    # Strong signed signals (relation/episode dims) push the weight up or
    # down from the centre.
    for attr in ("trust", "affection", "importance", "confidence"):
        v = getattr(edge, attr, None)
        if isinstance(v, (int, float)):
            base = max(base, min(1.0, 0.5 + 0.5 * abs(float(v))))
    if isinstance(edge, RelationEdge):
        # A clearly charged relationship (one strong dim) pulls harder.
        magnitude = max(abs(edge.valence), abs(edge.trust), abs(edge.affection))
        base = max(base, min(1.0, 0.4 + 0.6 * magnitude))
    if isinstance(edge, EpisodeEdge):
        base = max(base, min(1.0, 0.3 + 0.5 * abs(edge.emotional_shift)))
    if isinstance(edge, FactEdge):
        base = max(base, min(1.0, 0.3 + 0.5 * edge.confidence))
    return base


def spread_activation(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    *,
    gain: float = 0.35,
    hops: int = 2,
    decay_per_hop: float = 0.6,
    floor: float = 0.0,
) -> dict[str, float]:
    """Propagate activation from `seeds` over up to `hops` neighbours.

    The classic Anderson update, applied hop by hop:

        A_v += gain_h * (A_u * w_uv) / fan(u)

    where `gain_h = gain * decay_per_hop ** (hop - 1)` attenuates each
    successive hop and `fan(u)` is `u`'s degree (the fan effect: a node
    pointing at many things spreads less to each). `seeds` provides the
    initial activation (SelfNode base + matched nodes' RRF score).
    """
    activation: dict[str, float] = {nid: max(floor, float(a)) for nid, a in seeds.items()}

    frontier = dict(activation)
    for hop in range(1, max(1, hops) + 1):
        g = gain * (decay_per_hop ** (hop - 1))
        next_frontier: dict[str, float] = {}
        for u_id, u_act in frontier.items():
            if u_act <= 0.0:
                continue
            fan = graph.degree(u_id)
            if fan <= 0:
                continue
            for edge, neighbour in graph.neighbors(u_id):
                w = _edge_strength(edge) * _node_strength(neighbour)
                contributed = g * (u_act * w) / fan
                if contributed <= 0.0:
                    continue
                cur = activation.get(neighbour.id, 0.0) + contributed
                activation[neighbour.id] = cur
                next_frontier[neighbour.id] = next_frontier.get(neighbour.id, 0.0) + contributed
        if not next_frontier:
            break
        frontier = next_frontier
    return activation


def combined_activation(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    *,
    now: float,
    decay: float = 0.5,
    decay_half_life: float = 0.0,
    gain: float = 0.35,
    hops: int = 2,
    base_weight: float = 1.0,
    spread_weight: float = 1.0,
) -> dict[str, float]:
    """Final activation per node = `base_weight * BLL + spread_weight * spread`.

    Convenience wrapper used by the retriever. `seeds` is the RRF-derived
    seed map plus the SelfNode base; the BLL is added per-node so even a node
    the query did not match can surface if it is well-practiced and a
    neighbour matched.
    """
    bll: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        bll[nid] = base_level_activation(node, now=now, decay=decay, decay_half_life=decay_half_life)
    # The SelfNode is always "on" — seed it if the caller did not.
    seeds = dict(seeds)
    if graph.SELF_ID in graph.nodes and graph.SELF_ID not in seeds:
        seeds[graph.SELF_ID] = 0.5
    spread = spread_activation(graph, seeds, gain=gain, hops=hops)
    out: dict[str, float] = {}
    for nid in graph.nodes:
        out[nid] = base_weight * bll.get(nid, 0.0) + spread_weight * spread.get(nid, 0.0)
    return out


__all__ = [
    "base_level_activation",
    "spread_activation",
    "combined_activation",
]
