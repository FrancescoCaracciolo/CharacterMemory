"""Memory decay + effective-importance scoring.

Effective importance combines a fact's base importance, its age, a bounded
familiarity benefit from prior prompt exposure and the emotional impact of
that memory.

    Formula:
        intrinsic = base * E
        aged = intrinsic * exp(-age / (half_life * D))
        familiar = min(intrinsic, aged + intrinsic * 0.1 * recall/(recall + 10))

    where:
        E = 1 + impact         (emotionally intense memories are stronger)
        D = 1 + impact         (emotionally intense memories decay more slowly)

Prompt exposure is deliberately weak: it can restore at most ten percent of
intrinsic salience and it never refreshes the decay clock. Query relevance is
applied separately by :mod:`structured` so browsing/inspection can still show
the query-independent familiarity score.
"""

import math
from typing import Optional


MAX_EXPOSURE_BOOST = 0.10
EXPOSURE_HALF_SATURATION = 10.0
MAX_RELEVANCE_REPAIR = 1.10


def exposure_saturation(
    recall_count: int, half_saturation: float = EXPOSURE_HALF_SATURATION
) -> float:
    """Return a bounded exposure signal in ``[0, 1]``.

    ``count / (count + half_saturation)`` reaches one half at ten exposures
    with the default.  Recall counts remain useful telemetry without becoming
    an unbounded popularity multiplier.
    """
    count = max(0.0, float(recall_count))
    half = max(1e-9, float(half_saturation))
    return count / (count + half)


def relevance_repaired_score(
    intrinsic: float,
    familiar: float,
    relevance: float,
    *,
    max_repair: float = MAX_RELEVANCE_REPAIR,
) -> float:
    """Let relevance restore at most ``max_repair`` of lost salience.

    At maximum relevance, an intrinsic score of 1.0 that decayed to 0.4 is
    ranked at ``0.4 + 1.1 * 0.6 == 1.06``.  Familiarity above intrinsic is
    never manufactured here; malformed/custom scores are simply left intact.
    """
    intrinsic = max(0.0, float(intrinsic))
    familiar = max(0.0, float(familiar))
    rel = max(0.0, min(1.0, float(relevance)))
    lost = max(0.0, intrinsic - familiar)
    return familiar + rel * max(0.0, float(max_repair)) * lost


def intrinsic_score(
    base_importance: float, emotion_impact: Optional[float] = None
) -> float:
    """Return query-independent salience before time decay or familiarity."""
    base = max(0.0, float(base_importance))
    if emotion_impact is None:
        return base
    impact = max(0.0, min(1.0, float(emotion_impact)))
    return base * (1.0 + impact)


def decay_score(
    base_importance: float,
    recall_count: int,
    age_seconds: float,
    half_life: float,
    emotion_impact: Optional[float] = None,
) -> float:
    """Return non-negative, query-independent familiar salience."""
    # Emotion impact is a non-negative magnitude derived from a vector.
    if emotion_impact is not None:
        impact = max(0.0, min(1.0, float(emotion_impact)))
        decay_resistance = 1.0 + impact
    else:
        decay_resistance = 1
     
    adjusted_half_life = half_life * decay_resistance
    if adjusted_half_life <= 0:
        time_factor = 1.0
    else:
        time_factor = math.exp(-max(0.0, age_seconds) / adjusted_half_life)
    intrinsic = intrinsic_score(base_importance, emotion_impact)
    aged = intrinsic * time_factor
    familiarity = (
        intrinsic
        * MAX_EXPOSURE_BOOST
        * exposure_saturation(recall_count)
    )
    return min(intrinsic, aged + familiarity)


def age_seconds(created_at: float, last_recalled: float | None, now: float) -> float:
    """Seconds since creation.

    ``last_recalled`` remains in the signature for source compatibility, but
    is intentionally ignored: merely surfacing a memory must not renew it.
    """
    del last_recalled
    return max(0.0, now - created_at)
