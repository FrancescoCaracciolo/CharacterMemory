"""Activation functions: bounded ACT-R-style retention + spreading activation.

Two pieces, kept pure (no I/O, no globals) so they are easy to test and so
the retriever can compose them:

- :func:`base_level_activation` — ACT-R power-law retention anchored to node
  creation, with bounded familiarity from prompt exposure.
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
from typing import AbstractSet, Optional

from ..emotion_vectors import emotion_similarity, emotional_impact
from ..memory.decay import MAX_EXPOSURE_BOOST, exposure_saturation
from .edges import ChatEdge, CoOccurrenceEdge, Edge, EpisodeEdge, FactEdge, RelationEdge
from .graph import KnowledgeGraph
from .nodes import Node

# Numeric spreading rebuilds the full privacy-filtered snapshot every query.
# On the prod-db snapshot that snapshot costs ~75 ms, so it only beats the
# scalar walker for high-degree multi-seed queries. ``auto`` therefore stays
# on the scalar path; tests and the microbenchmark force ``engine="numeric"``.
NUMERIC_MIN_NODES = 10_000
NUMERIC_MIN_EDGES = 200_000

_STRENGTH_ATTRS = ("trust", "affection", "importance", "confidence")


def base_level_activation(
    node: Node,
    *,
    now: float,
    decay: float = 0.5,
    decay_half_life: float = 0.0,
) -> float:
    """Return bounded ACT-R-style base-level activation.

    Creation contributes the usual power-law mass. Prompt exposure can
    multiply that mass by at most 1.10, reaching half of the allowance at ten
    recalls. Neither the last-recalled time nor later practice timestamps
    reset age. The optional exponential factor is applied to positive
    activation mass before taking its logarithm, so stale negative activation
    cannot become spuriously stronger.
    """
    created_at = float(node.created_at or 0.0)
    if created_at <= 0.0 and node.practice_times:
        created_at = min(float(t) for t in node.practice_times)
    if created_at <= 0.0:
        return -2.0

    age = max(1e-3, now - created_at)
    mass = math.pow(age, -decay)
    if decay_half_life and decay_half_life > 0.0:
        mass *= math.exp(-age / (decay_half_life * 2.0))
    # Keep the graph's established floor so direct query seeds and spreading
    # can still recover an old node. Familiarity is added above that floor;
    # applying it before the clamp would erase the bounded benefit for almost
    # every realistically-aged node.
    retained = max(-2.0, math.log(max(mass, 1e-300)))
    familiarity = 1.0 + (
        MAX_EXPOSURE_BOOST * exposure_saturation(node.recall_count)
    )
    return retained + math.log(familiarity)


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
    if isinstance(shift, dict):
        s += 0.2 * emotional_impact(shift)
    return min(1.5, s)


def _has_attached_strength_attrs(edge: Edge) -> bool:
    """True when extra strength fields were attached after construction.

    Chat/co-occurrence fast paths skip the generic attribute scan. A caller
    that dynamically sets ``trust`` / ``affection`` / ``importance`` /
    ``confidence`` must still take the generic path.
    """
    data = getattr(edge, "__dict__", None)
    if not data:
        return False
    for attr in _STRENGTH_ATTRS:
        if attr not in data:
            continue
        value = data[attr]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
    return False


def _clamped_weight(value: object, default: float = 0.5) -> float:
    """Clamp an edge weight, treating a missing/zero value as ``default``."""
    return max(0.0, min(1.0, float(value or default)))


def _edge_strength_generic(edge: Edge) -> float:
    """The `u->v` edge weight in [0, 1] (all edge kinds)."""
    if isinstance(edge, CoOccurrenceEdge):
        base = edge.effective_weight()
    else:
        base = float(getattr(edge, "weight", 0.5) or 0.5)
    base = max(0.0, min(1.0, base))
    # Strong relationship signals and episode-vector impact push the weight
    # up from the centre.
    for attr in _STRENGTH_ATTRS:
        v = getattr(edge, attr, None)
        if isinstance(v, (int, float)):
            base = max(base, min(1.0, 0.5 + 0.5 * abs(float(v))))
    if isinstance(edge, RelationEdge):
        # A clearly charged relationship (one strong dim) pulls harder.
        magnitude = max(abs(edge.valence), abs(edge.trust), abs(edge.affection))
        base = max(base, min(1.0, 0.4 + 0.6 * magnitude))
    if isinstance(edge, EpisodeEdge):
        base = max(
            base,
            min(1.0, 0.3 + 0.5 * emotional_impact(edge.emotional_shift)),
        )
    if isinstance(edge, FactEdge):
        base = max(base, min(1.0, 0.3 + 0.5 * edge.confidence))
    return base


def _edge_strength(edge: Edge) -> float:
    """The `u->v` edge weight in [0, 1].

    Exact ``ChatEdge`` / ``CoOccurrenceEdge`` instances take a fast path;
    subclasses and edges with dynamically attached strength attributes keep
    the generic evaluation.
    """
    exact = type(edge)
    if exact is ChatEdge and not _has_attached_strength_attrs(edge):
        return _clamped_weight(getattr(edge, "weight", 0.5))
    if exact is CoOccurrenceEdge and not _has_attached_strength_attrs(edge):
        return max(0.0, min(1.0, edge.effective_weight()))
    return _edge_strength_generic(edge)


def _filtered_graph_size(
    graph: KnowledgeGraph,
    allowed_node_ids: Optional[AbstractSet[str]],
) -> tuple[int, int]:
    """Node/edge counts after the privacy filter, used for engine dispatch."""
    if allowed_node_ids is None:
        return len(graph.nodes), len(graph.edges)
    n_nodes = 0
    for nid in allowed_node_ids:
        if nid in graph.nodes:
            n_nodes += 1
    n_edges = 0
    for edge in graph.edges.values():
        if edge.src in allowed_node_ids and edge.dst in allowed_node_ids:
            n_edges += 1
    return n_nodes, n_edges


def _use_numeric(
    graph: KnowledgeGraph,
    allowed_node_ids: Optional[AbstractSet[str]],
    engine: str,
) -> bool:
    """Choose NumPy spreading only when it is safe and large enough."""
    if engine == "scalar":
        return False
    # Custom graph subclasses may override neighbour walking; the numeric
    # snapshot reconstructs adjacency from stored edges and SYMMETRIC_KINDS.
    if type(graph) is not KnowledgeGraph:
        return False
    if engine != "numeric":
        # Cheap unfiltered check first so typical queries never scan edges.
        if len(graph.nodes) < NUMERIC_MIN_NODES or len(graph.edges) < NUMERIC_MIN_EDGES:
            return False
        n_nodes, n_edges = _filtered_graph_size(graph, allowed_node_ids)
        if n_nodes < NUMERIC_MIN_NODES or n_edges < NUMERIC_MIN_EDGES:
            return False
    return True


def _spread_activation_scalar(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    *,
    gain: float = 0.35,
    hops: int = 2,
    decay_per_hop: float = 0.6,
    floor: float = 0.0,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
) -> dict[str, float]:
    """Reference Python walker. Exact neighbour order and float sums."""
    activation: dict[str, float] = {
        nid: max(floor, float(a))
        for nid, a in seeds.items()
        if allowed_node_ids is None or nid in allowed_node_ids
    }

    # Query-local caches: reinforcement, emotions and privacy can change
    # between calls. Cache only visited nodes/edges, preserving neighbor order
    # and parallel edges so both fan-out and floating-point sums stay exact.
    node_strengths: dict[str, float] = {}
    edge_strengths: dict[str, float] = {}
    weighted_neighbors: dict[str, list[tuple[str, float]]] = {}

    frontier = dict(activation)
    for hop in range(1, max(1, hops) + 1):
        g = gain * (decay_per_hop ** (hop - 1))
        next_frontier: dict[str, float] = {}
        for u_id, u_act in frontier.items():
            if u_act <= 0.0:
                continue
            neighbours = weighted_neighbors.get(u_id)
            if neighbours is None:
                neighbours = []
                for edge, neighbour in graph.neighbors(u_id):
                    if allowed_node_ids is not None and neighbour.id not in allowed_node_ids:
                        continue
                    edge_strength = edge_strengths.get(edge.id)
                    if edge_strength is None:
                        edge_strength = _edge_strength(edge)
                        edge_strengths[edge.id] = edge_strength
                    node_strength = node_strengths.get(neighbour.id)
                    if node_strength is None:
                        node_strength = _node_strength(neighbour)
                        node_strengths[neighbour.id] = node_strength
                    neighbours.append((neighbour.id, edge_strength * node_strength))
                weighted_neighbors[u_id] = neighbours
            fan = len(neighbours)
            if fan <= 0:
                continue
            for neighbour_id, w in neighbours:
                contributed = g * (u_act * w) / fan
                if contributed <= 0.0:
                    continue
                cur = activation.get(neighbour_id, 0.0) + contributed
                activation[neighbour_id] = cur
                next_frontier[neighbour_id] = next_frontier.get(neighbour_id, 0.0) + contributed
        if not next_frontier:
            break
        frontier = next_frontier
    return activation


def spread_activation(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    *,
    gain: float = 0.35,
    hops: int = 2,
    decay_per_hop: float = 0.6,
    floor: float = 0.0,
    allowed_node_ids: Optional[AbstractSet[str]] = None,
    engine: str = "auto",
) -> dict[str, float]:
    """Propagate activation from `seeds` over up to `hops` neighbours.

    The classic Anderson update, applied hop by hop:

        A_v += gain_h * (A_u * w_uv) / fan(u)

    where `gain_h = gain * decay_per_hop ** (hop - 1)` attenuates each
    successive hop and `fan(u)` is `u`'s degree (the fan effect: a node
    pointing at many things spreads less to each). `seeds` provides the
    initial activation (SelfNode base + matched nodes' RRF score).

    ``engine`` is ``"auto"`` (scalar unless the graph is far larger than the
    current production snapshot), ``"scalar"`` (this module's walker, the
    correctness reference) or ``"numeric"``.
    """
    if _use_numeric(graph, allowed_node_ids, engine):
        from .numeric_activation import spread_activation_numeric

        return spread_activation_numeric(
            graph,
            seeds,
            gain=gain,
            hops=hops,
            decay_per_hop=decay_per_hop,
            floor=floor,
            allowed_node_ids=allowed_node_ids,
        )
    return _spread_activation_scalar(
        graph,
        seeds,
        gain=gain,
        hops=hops,
        decay_per_hop=decay_per_hop,
        floor=floor,
        allowed_node_ids=allowed_node_ids,
    )


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
    allowed_node_ids: Optional[AbstractSet[str]] = None,
    engine: str = "auto",
) -> dict[str, float]:
    """Final activation per node = `base_weight * BLL + spread_weight * spread`.

    Convenience wrapper used by the retriever. `seeds` is the RRF-derived
    seed map plus the SelfNode base; the BLL is added per-node so even a node
    the query did not match can surface if it is well-practiced and a
    neighbour matched. For the per-factor decomposition use
    :func:`combined_activation_breakdown`.
    """
    return {
        nid: b["score"]
        for nid, b in combined_activation_breakdown(
            graph, seeds,
            now=now, decay=decay, decay_half_life=decay_half_life,
            gain=gain, hops=hops,
            base_weight=base_weight, spread_weight=spread_weight,
            allowed_node_ids=allowed_node_ids,
            engine=engine,
        ).items()
    }


def combined_activation_breakdown(
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
    allowed_node_ids: Optional[AbstractSet[str]] = None,
    engine: str = "auto",
) -> dict[str, dict[str, float]]:
    """Same fusion as :func:`combined_activation`, but with per-factor detail.

    Returns ``{node_id: {bll, spread, seed, base, spread_w, emotion_mult, score}}``:

    - ``bll``          — raw ACT-R base-level learning (may be negative).
    - ``spread``       — raw spreading-activation contribution (includes the
      node's own seed plus everything propagated to it).
    - ``seed``         — the query/RRF + SelfNode seed this node started with
      (the slice of ``spread`` that is not propagated from neighbours). Useful
      to see "did this node match the query at all?".
    - ``base``         — ``base_weight * bll`` (weighted BLL contribution).
    - ``spread_w``     — ``spread_weight * spread`` (weighted spread contribution).
    - ``emotion_mult`` — the multiplicative mood-alignment boost actually
      applied (``1 + similarity`` when the node is emotional & the score is
      positive, else ``1.0``).
    - ``score``        — the final activation = ``base + spread_w`` then scaled
      by ``emotion_mult``.

    The GUI / `test_activation_details` use this to render a score-breakdown.
    """
    bll: dict[str, float] = {}
    nodes = (
        graph.nodes.items()
        if allowed_node_ids is None
        else (
            (nid, graph.nodes[nid])
            for nid in allowed_node_ids
            if nid in graph.nodes
        )
    )
    for nid, node in nodes:
        bll[nid] = base_level_activation(node, now=now, decay=decay, decay_half_life=decay_half_life)
    # The SelfNode is always "on" — seed it if the caller did not.
    seeds = dict(seeds)
    if (
        graph.SELF_ID in graph.nodes
        and graph.SELF_ID not in seeds
        and (allowed_node_ids is None or graph.SELF_ID in allowed_node_ids)
    ):
        seeds[graph.SELF_ID] = 0.5
    spread = spread_activation(
        graph,
        seeds,
        gain=gain,
        hops=hops,
        allowed_node_ids=allowed_node_ids,
        engine=engine,
    )
    out: dict[str, dict[str, float]] = {}
    self_node = graph.nodes.get(graph.SELF_ID)
    current_mood = getattr(self_node, "current_mood", {}) or {}
    output_nodes = (
        graph.nodes.items()
        if allowed_node_ids is None
        else (
            (nid, graph.nodes[nid])
            for nid in allowed_node_ids
            if nid in graph.nodes
        )
    )
    for nid, node in output_nodes:
        bll_val = bll.get(nid, 0.0)
        spread_val = spread.get(nid, 0.0)
        base_term = base_weight * bll_val
        spread_term = spread_weight * spread_val
        score = base_term + spread_term
        shift = getattr(node, "emotional_shift", None)
        emotion_mult = 1.0
        if score > 0.0 and isinstance(shift, dict):
            emotion_mult = 1.0 + emotion_similarity(shift, current_mood)
            score *= emotion_mult
        out[nid] = {
            "bll": float(bll_val),
            "spread": float(spread_val),
            "seed": float(seeds.get(nid, 0.0)),
            "base": float(base_term),
            "spread_w": float(spread_term),
            "emotion_mult": float(emotion_mult),
            "score": float(score),
        }
    return out


__all__ = [
    "NUMERIC_MIN_EDGES",
    "NUMERIC_MIN_NODES",
    "base_level_activation",
    "spread_activation",
    "combined_activation",
    "combined_activation_breakdown",
]
