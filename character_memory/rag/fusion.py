"""Shared weighted RRF and confidence scoring for hybrid backends."""
from collections.abc import Hashable
from .base import Hit

def fuse(queries, candidates, k, rrf_k, min_dense_similarity, result_key, document):
    # Reciprocal Rank Fusion over result groups, weight-scaled. Normal
    # documents form one group per positional node. A memory may stamp
    # several index keys with the same `_result_id`; those keys then rank
    # as one result and cannot consume several top-k slots.
    scores: dict[Hashable, float] = {}
    representative: dict[Hashable, int] = {}
    sim_by_result: dict[Hashable, float] = {}
    lexical_by_result: dict[Hashable, float] = {}
    confidence_failure: dict[Hashable, float] = {}
    max_query_weight = max((weight for _, weight in queries), default=0.0)

    def add_confidence(key: Hashable, confidence: float, weight: float) -> None:
        if max_query_weight <= 0.0:
            return
        weighted = max(
            0.0,
            min(1.0, float(confidence) * float(weight) / max_query_weight),
        )
        confidence_failure[key] = confidence_failure.get(key, 1.0) * (
            1.0 - weighted
        )

    for (_, weight), (bm25_hits, dense_hits) in zip(queries, candidates):
        best_lexical = max((lexical for _, _, lexical in bm25_hits), default=0.0)
        for pos, rank, lexical in bm25_hits:
            key = result_key(pos)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)
            representative.setdefault(key, pos)
            lexical_by_result[key] = max(
                lexical_by_result.get(key, 0.0), lexical
            )
            if best_lexical > 0.0:
                add_confidence(key, lexical / best_lexical, weight)
        for pos, rank, sim in dense_hits:
            key = result_key(pos)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)
            representative.setdefault(key, pos)
            if sim > sim_by_result.get(key, -1.0):
                sim_by_result[key] = sim
            if min_dense_similarity is None:
                dense_confidence = max(0.0, min(1.0, sim))
            elif min_dense_similarity < 1.0:
                dense_confidence = max(
                    0.0,
                    min(
                        1.0,
                        (sim - min_dense_similarity)
                        / (1.0 - min_dense_similarity),
                    ),
                )
            else:
                dense_confidence = 0.0
            add_confidence(key, dense_confidence, weight)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    # Preserve rank-only RRF as the ordering score and expose it separately
    # from confidence-based relevance.  A positive lexical match can now
    # carry full confidence even when the dense model misses it, while
    # weak dense neighbors cannot masquerade as two-channel agreement.
    max_rrf = sum(weight for _, weight in queries) * (
        2.0 / (rrf_k + 1)
    )
    hits: list[Hit] = []
    for key, score in ranked:
        pos = representative[key]
        text, source, metadata = document(pos)
        meta = dict(metadata)
        meta["similarity"] = sim_by_result.get(key, 0.0)
        meta["dense_similarity"] = sim_by_result.get(key, 0.0)
        meta["bm25_score"] = lexical_by_result.get(key, 0.0)
        meta["rrf_relevance"] = (
            max(0.0, min(1.0, float(score) / max_rrf))
            if max_rrf > 0.0
            else 0.0
        )
        meta["normalized_relevance"] = max(
            0.0,
            min(1.0, 1.0 - confidence_failure.get(key, 1.0)),
        )
        hits.append(
            Hit(text=text, score=score, source=source, metadata=meta)
        )
    return hits

