"""Memory decay + effective-importance scoring.

Effective importance combines a fact's base importance, how long ago it was
last recalled, how often it has been recalled and the emotional impact of that memory:

    Formula:
        effective = base * E * exp(-age / (half_life * D)) * (1 + log1p(recall))

    where:
        E = 1 + impact         (emotionally intense memories are stronger)
        D = 1 + impact         (emotionally intense memories decay more slowly)

So frequently-recalled, recently-used facts stay strong, while unused ones
fade. `base` alone decides stickiness (always-in-prompt) so a deliberately
important fact never fully vanishes.
"""

import math
from typing import Optional


def decay_score(
    base_importance: float,
    recall_count: int,
    age_seconds: float,
    half_life: float,
    emotion_impact: Optional[float] = None,
) -> float:
    """Return a non-negative effective importance."""
    # Emotion impact is a non-negative magnitude derived from a vector.
    if emotion_impact is not None:
        impact = max(0.0, min(1.0, float(emotion_impact)))
        emotion_boost = 1.0 + impact
        decay_resistance = 1.0 + impact
    else:
        emotion_boost = 1  
        decay_resistance = 1
     
    adjusted_half_life = half_life * decay_resistance
    if adjusted_half_life <= 0:
        time_factor = 1.0
    else:
        time_factor = math.exp(-max(0.0, age_seconds) / adjusted_half_life)
    usage_factor = 1.0 + math.log1p(max(0, recall_count))
    return base_importance * time_factor * usage_factor * emotion_boost


def age_seconds(created_at: float, last_recalled: float | None, now: float) -> float:
    """Seconds since the fact was last touched (creation if never recalled)."""
    anchor = last_recalled if last_recalled else created_at
    return max(0.0, now - anchor)
