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
from typing import Optional

from ..emotion_vectors import emotion_similarity, emotional_impact
from ..memory.decay import MAX_EXPOSURE_BOOST, exposure_saturation
from .edges import CoOccurrenceEdge, Edge, EpisodeEdge, FactEdge, RelationEdge
from .graph import KnowledgeGraph
from .nodes import Node


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


def _edge_strength(edge: Edge) -> float:
    """The `u->v` edge weight in [0, 1]."""
    if isinstance(edge, CoOccurrenceEdge):
        base = edge.effective_weight()
    else:
        base = float(getattr(edge, "weight", 0.5) or 0.5)
    base = max(0.0, min(1.0, base))
    # Strong relationship signals and episode-vector impact push the weight
    # up from the centre.
    for attr in ("trust", "affection", "importance", "confidence"):
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
    for nid, node in graph.nodes.items():
        bll[nid] = base_level_activation(node, now=now, decay=decay, decay_half_life=decay_half_life)
    # The SelfNode is always "on" — seed it if the caller did not.
    seeds = dict(seeds)
    if graph.SELF_ID in graph.nodes and graph.SELF_ID not in seeds:
        seeds[graph.SELF_ID] = 0.5
    spread = spread_activation(graph, seeds, gain=gain, hops=hops)
    out: dict[str, dict[str, float]] = {}
    self_node = graph.nodes.get(graph.SELF_ID)
    current_mood = getattr(self_node, "current_mood", {}) or {}
    for nid, node in graph.nodes.items():
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
    "base_level_activation",
    "spread_activation",
    "combined_activation",
    "combined_activation_breakdown",
]
