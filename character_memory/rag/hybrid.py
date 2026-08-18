"""Hybrid retrieval: BM25 (lexical) + FAISS (dense similarity), fused with RRF.

Uses llama-index as a library (`TextNode` schema + `BM25Retriever`) and a
FAISS vector store for dense similarity. Scores are combined with Reciprocal
Rank Fusion (RRF), so the two signals need no calibration.

A single `HybridSearch` instance is reused by the RAG-backed memories
(character info, dialogue style) and by the structured memories (facts, etc.)
for their BM25+similarity recall.
"""

import json
import hashlib
import os
import warnings
from collections.abc import Hashable, Iterable
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # Windows / non-POSIX: fall back to atomic-rename only.
    fcntl = None

import faiss
import numpy as np

from ..chunking.base import Chunk
from ..llm.embedding_base import EmbeddingProvider
from .base import Hit, Query, RAGSystem, as_queries

# llama-index as a library: node schema + BM25 retriever.
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

# RRF constant (standard value).
DEFAULT_RRF_K = 60
DEFAULT_CLEANUP_MIN_DELETED = 64
DEFAULT_CLEANUP_DELETED_RATIO = 0.25
INDEX_META_SCHEMA_VERSION = 1
INDEX_META_FILENAME = "index_meta.json"


def _matches(where: dict | None, meta: dict) -> bool:
    if not where:
        return True
    return all(meta.get(k) == v for k, v in where.items())


class HybridSearch(RAGSystem):
    """BM25 + FAISS similarity search combined via reciprocal rank fusion."""

    name = "hybrid"

    def __init__(
        self,
        embedder: EmbeddingProvider,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = 30,
        min_dense_similarity: Optional[float] = None,
        cleanup_min_deleted: int = DEFAULT_CLEANUP_MIN_DELETED,
        cleanup_deleted_ratio: float = DEFAULT_CLEANUP_DELETED_RATIO,
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        if min_dense_similarity is not None and not -1.0 <= float(min_dense_similarity) <= 1.0:
            raise ValueError("min_dense_similarity must be between -1 and 1")
        self.min_dense_similarity = (
            None if min_dense_similarity is None else float(min_dense_similarity)
        )
        self.cleanup_min_deleted = max(1, int(cleanup_min_deleted))
        self.cleanup_deleted_ratio = max(
            0.0, min(1.0, float(cleanup_deleted_ratio))
        )
        self._nodes: list[TextNode] = []
        self._bm25: Optional[BM25Retriever] = None
        self._index: Optional[faiss.Index] = None

    # ------------------------------------------------------------------ build
    def _chunk_to_node(self, chunk: Chunk, idx: int) -> TextNode:
        meta = dict(chunk.metadata)
        meta.setdefault("source", chunk.source)
        # Internal positional key; never collides with app-supplied "id".
        meta["_pos"] = idx
        return TextNode(text=chunk.text, metadata=meta)

    def build(self, chunks: list[Chunk]) -> None:
        self._nodes = [self._chunk_to_node(c, i) for i, c in enumerate(chunks)]
        self._build_bm25()
        self._build_faiss()

    @staticmethod
    def _normalize_vectors(vecs: "np.ndarray") -> "np.ndarray":
        """Return contiguous row-wise unit vectors for cosine FAISS search."""
        arr = np.asarray(vecs, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.ascontiguousarray(arr / norms)

    def _embed_queries(self, texts: list[str]) -> "np.ndarray":
        method = getattr(self.embedder, "embed_queries", None)
        return (method or self.embedder.embed)(texts)

    def _embed_documents(self, texts: list[str]) -> "np.ndarray":
        method = getattr(self.embedder, "embed_documents", None)
        return (method or self.embedder.embed)(texts)

    def add_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        start = len(self._nodes)
        new_nodes = [self._chunk_to_node(c, start + i) for i, c in enumerate(chunks)]
        self._nodes.extend(new_nodes)
        # Append dense vectors directly; rebuild the lexical index cheaply.
        vecs = self._normalize_vectors(
            self._embed_documents([n.text for n in new_nodes])
        )
        if self._index is None:
            # Normal first-add path: use the vectors already calculated above
            # instead of routing through _build_faiss and embedding them twice.
            if start == 0:
                index = faiss.IndexFlatIP(int(vecs.shape[1]))
                index.add(np.ascontiguousarray(vecs))
                self._index = index
            else:
                # Defensive recovery for an inconsistent in-memory state with
                # nodes but no dense index. Embed only the pre-existing nodes;
                # the new batch's vectors are still reused.
                old_vecs = self._normalize_vectors(
                    self._embed_documents([n.text for n in self._nodes[:start]])
                )
                all_vecs = np.concatenate([old_vecs, vecs], axis=0)
                index = faiss.IndexFlatIP(int(all_vecs.shape[1]))
                index.add(np.ascontiguousarray(all_vecs))
                self._index = index
        else:
            self._index.add(np.ascontiguousarray(vecs))
        self._build_bm25()

    @staticmethod
    def _is_deleted(node: TextNode) -> bool:
        return bool(node.metadata.get("_deleted", False))

    @property
    def deleted_count(self) -> int:
        return sum(1 for node in self._nodes if self._is_deleted(node))

    def delete_documents(self, ids: Iterable[Any]) -> int:
        """Tombstone every document matching an application metadata id.

        The id is resolved against ``metadata['id']``.  It is never treated as
        a FAISS position; ``_pos`` remains the canonical dense-index key.
        """
        wanted = set(ids)
        if not wanted:
            return 0
        deleted = 0
        for node in self._nodes:
            if node.metadata.get("id") in wanted and not self._is_deleted(node):
                node.metadata["_deleted"] = True
                deleted += 1
        if not deleted:
            return 0
        self._build_bm25()
        total = len(self._nodes)
        tombstones = self.deleted_count
        if self.count == 0 or (
            tombstones >= self.cleanup_min_deleted
            and tombstones / max(1, total) >= self.cleanup_deleted_ratio
        ):
            self.cleanup()
        return deleted

    def cleanup(self) -> int:
        """Physically remove tombstones without re-embedding healthy vectors."""
        deleted = self.deleted_count
        if not deleted:
            return 0
        active_positions = [
            pos for pos, node in enumerate(self._nodes) if not self._is_deleted(node)
        ]
        if not active_positions:
            self._nodes = []
            self._bm25 = None
            self._index = None
            return deleted

        old_nodes = self._nodes
        can_reconstruct = (
            self._index is not None
            and int(self._index.ntotal) == len(old_nodes)
        )
        self._nodes = []
        for new_pos, old_pos in enumerate(active_positions):
            old = old_nodes[old_pos]
            meta = dict(old.metadata)
            meta.pop("_deleted", None)
            meta["_pos"] = new_pos
            self._nodes.append(TextNode(text=old.text, metadata=meta))

        self._build_bm25()
        if can_reconstruct:
            assert self._index is not None
            vectors = np.asarray(
                [self._index.reconstruct(pos) for pos in active_positions],
                dtype="float32",
            )
            index = faiss.IndexFlatIP(int(self._index.d))
            index.add(np.ascontiguousarray(vectors))
            self._index = index
        else:
            # A positional mismatch means existing vectors cannot safely be
            # associated with nodes. Re-embedding active text is the necessary
            # correctness recovery, not routine cleanup behavior.
            warnings.warn(
                "HybridSearch cleanup found an inconsistent dense index; "
                "re-embedding active documents.",
                stacklevel=2,
            )
            self._build_faiss()
        return deleted

    def _build_bm25(self) -> None:
        active = [node for node in self._nodes if not self._is_deleted(node)]
        if not active:
            self._bm25 = None
            return
        # Cap top_k at the node count so BM25 never emits its override warning.
        top_k = min(self.candidate_pool, len(active))
        self._bm25 = BM25Retriever.from_defaults(
            nodes=active,
            similarity_top_k=top_k,
            verbose=False,
        )

    def _build_faiss(self) -> None:
        if not self._nodes:
            self._index = None
            return
        vecs = self._normalize_vectors(
            self._embed_documents([n.text for n in self._nodes])
        )
        dim = int(vecs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(vecs))
        self._index = index

    # SEARCH
    def search(self, query: Query, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Return up to `k` hits for `query`, fusing across weighted queries.

        ``query`` may be a plain string or a list of ``(text, weight)`` pairs.
        Each query runs its own BM25 + dense pass; the positional result lists
        are fused with reciprocal rank fusion where a query of weight ``w``
        contributes ``w / (rrf_k + rank + 1)`` per hit. A single (or unweighted)
        query is the legacy path and ranks identically to before.
        """
        active_count = self.count
        if active_count == 0 or self._index is None:
            return []
        queries = [(text, weight) for text, weight in as_queries(query) if weight > 0.0]
        if not queries:
            return []
        pool = min(max(self.candidate_pool, k * 3), active_count)

        # Embed every query text in one batch (one round-trip for the dense
        # pass regardless of how many messages are in the window).
        q_texts = [q for q, _ in queries]
        q_vecs = self._normalize_vectors(self._embed_queries(q_texts))

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

        for (qtext, weight), qv in zip(queries, q_vecs):
            bm25_hits, dense_hits = self._query_candidates(qtext, qv, pool, where)
            best_lexical = max((lexical for _, _, lexical in bm25_hits), default=0.0)
            for pos, rank, lexical in bm25_hits:
                key = self._result_key(pos)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                representative.setdefault(key, pos)
                lexical_by_result[key] = max(
                    lexical_by_result.get(key, 0.0), lexical
                )
                if best_lexical > 0.0:
                    add_confidence(key, lexical / best_lexical, weight)
            for pos, rank, sim in dense_hits:
                key = self._result_key(pos)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                representative.setdefault(key, pos)
                if sim > sim_by_result.get(key, -1.0):
                    sim_by_result[key] = sim
                if self.min_dense_similarity is None:
                    dense_confidence = max(0.0, min(1.0, sim))
                elif self.min_dense_similarity < 1.0:
                    dense_confidence = max(
                        0.0,
                        min(
                            1.0,
                            (sim - self.min_dense_similarity)
                            / (1.0 - self.min_dense_similarity),
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
            2.0 / (self.rrf_k + 1)
        )
        hits: list[Hit] = []
        for key, score in ranked:
            pos = representative[key]
            node = self._nodes[pos]
            meta = dict(node.metadata)
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
                Hit(text=node.text, score=score, source=meta.get("source", ""), metadata=meta)
            )
        return hits

    def _result_key(self, pos: int) -> Hashable:
        """Return a retrieval-group key without changing positional IDs."""
        result_id = self._nodes[pos].metadata.get("_result_id")
        return ("result", result_id) if result_id is not None else ("_pos", pos)

    def _query_candidates(
        self, qtext: str, qv: "np.ndarray", pool: int, where: dict | None
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
        """Run one query's BM25 + dense passes, returning ranked positional hits.

        Returns ``(bm25_hits, dense_hits)`` where each entry is ``(positional,
        rank, score)``. Zero-score lexical results and dense results below the
        configured cosine floor are excluded before RRF. `where` filters on
        metadata equality.
        """
        # Lexical candidates, keyed by internal positional index.
        bm25_hits: list[tuple[int, int, float]] = []
        if self._bm25 is not None:
            nodes = self._bm25.retrieve(qtext)
            rank = 0
            seen_results: set[Hashable] = set()
            for n in nodes:
                lexical_score = float(n.score or 0.0)
                if lexical_score <= 0.0:
                    continue
                if not _matches(where, n.metadata):
                    continue
                pos = int(n.metadata.get("_pos", -1))
                if pos < 0 or pos >= len(self._nodes):
                    continue
                key = self._result_key(pos)
                if key in seen_results:
                    continue
                seen_results.add(key)
                bm25_hits.append((pos, rank, lexical_score))
                rank += 1
                if len(bm25_hits) >= pool:
                    break

        # Dense search (ranked by cosine similarity via inner product).
        # `qv` is a single 1-D row vector; FAISS needs a 2-D (1, dim) array.
        # Tombstones can occupy the dense top-k even though they are filtered
        # below. Asking for ``pool + deleted_count`` guarantees enough room for
        # ``pool`` active candidates when that many active nodes exist.
        dense_pool = min(len(self._nodes), pool + self.deleted_count)
        sims, idxs = self._index.search(
            np.ascontiguousarray(qv.reshape(1, -1)), dense_pool
        )
        dense_hits: list[tuple[int, int, float]] = []
        n_nodes = len(self._nodes)
        rank = 0
        seen_results: set[Hashable] = set()
        for nid, sim in zip(idxs[0], sims[0]):
            if nid < 0:
                continue
            if (
                self.min_dense_similarity is not None
                and float(sim) <= self.min_dense_similarity
            ):
                continue
            # Guard against a stale dense index that has more vectors than the
            # current `_nodes` list (e.g. rows deleted from SQLite while the
            # index dir still holds ghost vectors, or a dim-mismatch rebuild
            # that left the on-disk faiss.index out of sync with nodes.json).
            # The positional id is the canonical key per AGENTS.md gotcha #1;
            # a ghost id is simply dropped rather than crashing recall.
            if nid >= n_nodes:
                continue
            node = self._nodes[nid]
            if self._is_deleted(node):
                continue
            if not _matches(where, node.metadata):
                continue
            key = self._result_key(int(nid))
            if key in seen_results:
                continue
            seen_results.add(key)
            dense_hits.append((int(nid), rank, float(sim)))
            rank += 1
        return bm25_hits, dense_hits

    # --------------------------------------------------------------- persist
    @property
    def count(self) -> int:
        return len(self._nodes) - self.deleted_count

    @property
    def documents(self) -> list[Chunk]:
        """Return all active indexed chunks."""
        return [
            Chunk(
                text=n.text,
                source=n.metadata.get("source", ""),
                metadata=dict(n.metadata),
            )
            for n in self._nodes
            if not self._is_deleted(n)
        ]

    def _index_fingerprint(self) -> str:
        """Return a safe fingerprint for vector-producing retrieval behavior."""
        payload = {
            "provider": (
                f"{type(self.embedder).__module__}."
                f"{type(self.embedder).__qualname__}"
            ),
            "provider_fingerprint": getattr(
                self.embedder, "index_fingerprint", None
            ),
            "dense_normalization": "l2-v1",
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _index_metadata(self) -> dict[str, Any]:
        # An empty index has no vector dimension.  Persisting one must stay an
        # offline operation instead of probing the embedding endpoint.
        dim = int(self._index.d) if self._index is not None else None
        return {
            "schema_version": INDEX_META_SCHEMA_VERSION,
            "embedding_fingerprint": self._index_fingerprint(),
            "dimension": dim,
            "dense_normalization": "l2-v1",
            "min_dense_similarity": self.min_dense_similarity,
        }

    def persist(self, path: str) -> None:
        """Persist nodes + dense index to `path` atomically and cross-process safe.

        Writes both files via temp + ``os.replace`` (POSIX-atomic), under an
        exclusive ``flock`` on ``<path>/.lock``. This prevents torn writes and
        interleaved writes when the cm_server and the Discord bot persist the
        same index concurrently. The lock is per index directory, matching the
        existing per-character in-process lock convention; it is auto-released
        by the OS on close/exit, so a crash can't leave it stuck.
        """
        os.makedirs(path, exist_ok=True)
        nodes_json = [
            {"id": n.metadata.get("id", i), "text": n.text,
             "source": n.metadata.get("source", ""), "metadata": n.metadata}
            for i, n in enumerate(self._nodes)
        ]
        index_meta = self._index_metadata()
        lock_path = os.path.join(path, ".lock")
        # "a+" keeps an existing lock file without truncating it; the file's
        # contents are never read, it only anchors the flock.
        with open(lock_path, "a+") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            nodes_tmp = os.path.join(path, "nodes.json.tmp")
            with open(nodes_tmp, "w", encoding="utf-8") as f:
                json.dump(nodes_json, f, ensure_ascii=False)
            meta_tmp = os.path.join(path, f"{INDEX_META_FILENAME}.tmp")
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(index_meta, f, ensure_ascii=False, sort_keys=True)
            if self._index is not None:
                idx_tmp = os.path.join(path, "faiss.index.tmp")
                faiss.write_index(self._index, idx_tmp)
                os.replace(idx_tmp, os.path.join(path, "faiss.index"))
            else:
                # An empty logical/physical index must not leave a ghost dense
                # file that triggers repair work on every subsequent load.
                try:
                    os.remove(os.path.join(path, "faiss.index"))
                except FileNotFoundError:
                    pass
            os.replace(meta_tmp, os.path.join(path, INDEX_META_FILENAME))
            # ``nodes.json`` is the publication/commit marker watched by
            # MemorySync. Publish it only after the matching dense state is in
            # place; otherwise a poller can pair new nodes with the old FAISS
            # file and mistake the transient count mismatch for corruption.
            os.replace(nodes_tmp, os.path.join(path, "nodes.json"))

    def load(self, path: str) -> None:
        idx_file = os.path.join(path, "faiss.index")
        loaded: Optional[faiss.Index] = None
        load_error: Optional[Exception] = None
        stored_meta: Optional[dict[str, Any]] = None
        meta_error: Optional[Exception] = None
        meta_file = os.path.join(path, INDEX_META_FILENAME)
        lock_path = os.path.join(path, ".lock")
        # Read nodes + vectors as one publication. Atomic rename protects each
        # file individually; this shared lock protects the relationship
        # between the two files while another process is persisting them.
        with open(lock_path, "a+") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                with open(
                    os.path.join(path, "nodes.json"), encoding="utf-8"
                ) as f:
                    data = json.load(f)
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            raw_meta = json.load(f)
                        if isinstance(raw_meta, dict):
                            stored_meta = raw_meta
                        else:
                            raise ValueError("index metadata is not an object")
                    except Exception as exc:  # noqa: BLE001 - repair below
                        meta_error = exc
                if os.path.exists(idx_file):
                    try:
                        loaded = faiss.read_index(idx_file)
                    except Exception as exc:  # noqa: BLE001 - repair below
                        load_error = exc
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

        self._nodes = []
        # Never let a previously-loaded dense index survive a failed repair of
        # the new path. It would refer to a different positional node set.
        self._index = None
        for i, entry in enumerate(data):
            meta = dict(entry.get("metadata") or {})
            if "id" in entry:
                meta["id"] = entry["id"]
            meta.setdefault("source", entry.get("source", ""))
            meta["_pos"] = i  # recompute canonical positional index
            self._nodes.append(TextNode(text=entry["text"], metadata=meta))
        rebuild = True
        if loaded is not None:
            expected = getattr(self.embedder, "dim", None)
            current_fingerprint = self._index_fingerprint()
            metadata_matches = bool(
                stored_meta is not None
                and stored_meta.get("schema_version") == INDEX_META_SCHEMA_VERSION
                and stored_meta.get("embedding_fingerprint") == current_fingerprint
                and stored_meta.get("dense_normalization") == "l2-v1"
                and stored_meta.get("dimension") == expected
            )
            if (
                expected is not None
                and loaded.d == expected
                and int(loaded.ntotal) == len(self._nodes)
                and metadata_matches
            ):
                self._index = loaded
                rebuild = False
            else:
                # Embedding behavior/dim drifted, the metadata is legacy, or
                # nodes/index cardinality diverged.
                print(
                    f"[rag] dense-index mismatch (index_dim={loaded.d}, "
                    f"embedder_dim={expected}, index_count={loaded.ntotal}, "
                    f"node_count={len(self._nodes)}, "
                    f"metadata_matches={metadata_matches}); rebuilding active nodes."
                )
        elif load_error is not None:
            print(
                f"[rag] could not load dense index {idx_file!r}: "
                f"{load_error!r}; rebuilding active nodes."
            )
        elif meta_error is not None:
            print(
                f"[rag] could not load dense metadata {meta_file!r}: "
                f"{meta_error!r}; rebuilding active nodes."
            )
        if rebuild:
            # Tombstones do not need new embeddings during a repair. Drop them
            # physically before recreating the dense index.
            active = [node for node in self._nodes if not self._is_deleted(node)]
            self._nodes = []
            for pos, node in enumerate(active):
                meta = dict(node.metadata)
                meta.pop("_deleted", None)
                meta["_pos"] = pos
                self._nodes.append(TextNode(text=node.text, metadata=meta))
            self._build_bm25()
            self._build_faiss()
            # Persist both corrected nodes and index so later loads are fast,
            # including the previously-missing-index case.
            self.persist(path)
        else:
            self._build_bm25()
