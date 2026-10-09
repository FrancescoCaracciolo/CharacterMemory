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
import threading
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
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
# Embedding runs outside the state lock against a snapshot of the nodes; when
# a concurrent write invalidates that snapshot the work is redone. After this
# many lost races the final attempt embeds under the lock so progress is
# guaranteed.
_MAX_SNAPSHOT_ATTEMPTS = 3


def _synchronized(method):
    """Serialize access to one HybridSearch instance's positional state."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapped


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
        self._index_revision = 0
        self._scoped_bm25_cache: OrderedDict[
            tuple[int, frozenset[int]], Optional[BM25Retriever]
        ] = OrderedDict()
        # Positions of live nodes per allowed-id set, memoized by object
        # identity: retrieval reuses one frozenset per visibility scope, so
        # the O(N) scan below runs once per scope per revision instead of
        # once per query.
        self._allowed_positions_memo: dict[
            tuple[int, int], tuple[object, frozenset[int]]
        ] = {}
        # Nodes, BM25 and FAISS are one positional data structure.  A write
        # must not interleave with another write (or a search): compaction can
        # otherwise clear/remap nodes while an incremental add is embedding,
        # leaving vectors and `_pos` values referring to different snapshots.
        # Embedding requests are never made while holding this lock: vectors
        # are computed first and installed only if `_index_revision` shows the
        # snapshot they were computed from is still current.
        self._state_lock = threading.RLock()
        # (realpath, revision, fingerprint) of the last state written to or
        # read from disk; `persist` is a no-op while it still matches.
        self._persisted_marker: Optional[tuple[str, int, str]] = None
        # realpath -> st_mtime_ns of the ``nodes.json`` this instance last
        # published or loaded; lets a cache poller tell its own writes apart
        # from another process's.
        self._synced_mtime_ns: dict[str, int] = {}
        # Per-thread query vectors computed ahead of time by
        # ``prefetched_queries`` so callers holding their own locks around
        # ``search`` do no network I/O inside them.
        self._prefetch = threading.local()

    def synced_mtime_ns(self, path: str) -> Optional[int]:
        """``st_mtime_ns`` of ``<path>/nodes.json`` as this instance last
        wrote or read it (``None`` if it never touched ``path``)."""
        return self._synced_mtime_ns.get(os.path.realpath(path))

    def _record_synced(self, path: str, mtime_ns: Optional[int] = None) -> None:
        if mtime_ns is None:
            try:
                mtime_ns = os.stat(os.path.join(path, "nodes.json")).st_mtime_ns
            except OSError:
                return
        self._synced_mtime_ns[os.path.realpath(path)] = mtime_ns

    @contextmanager
    def prefetched_queries(self, query: Query) -> Iterator[None]:
        """Embed ``query`` now; a ``search`` for the same texts on this thread
        inside the block reuses the vectors instead of calling the provider.

        Lets a caller that serializes its own state (the knowledge-graph
        retriever) keep embedding latency outside its lock. A failed embedding
        is ignored here: ``search`` then embeds (and raises) as usual.
        """
        texts = tuple(text for text, weight in as_queries(query) if weight > 0.0)
        previous = getattr(self._prefetch, "entry", None)
        if texts:
            try:
                vectors = self._normalize_vectors(
                    self._embed_queries(list(texts)), expected_count=len(texts)
                )
            except Exception:  # noqa: BLE001 - search() surfaces the error
                vectors = None
            if vectors is not None:
                self._prefetch.entry = (texts, vectors)
        try:
            yield
        finally:
            self._prefetch.entry = previous

    def _query_vectors(self, texts: list[str]) -> "np.ndarray":
        entry = getattr(self._prefetch, "entry", None)
        if entry is not None and entry[0] == tuple(texts):
            return entry[1]
        return self._normalize_vectors(
            self._embed_queries(texts), expected_count=len(texts)
        )

    def _invalidate_scoped_search(self) -> None:
        self._index_revision += 1
        self._scoped_bm25_cache.clear()
        self._allowed_positions_memo.clear()

    def _allowed_positions_for(self, allowed_ids: object, wanted: set) -> frozenset[int]:
        """Positions of live nodes whose app id is in ``wanted``, memoized.

        Keyed by ``(revision, id(allowed_ids))``; the entry keeps the key
        object referenced so a recycled id cannot alias a live entry, and an
        identity check on hit guards against pathological recycling. Any
        index mutation clears the memo via :meth:`_invalidate_scoped_search`.
        """
        key = (self._index_revision, id(allowed_ids))
        hit = self._allowed_positions_memo.get(key)
        if hit is not None and hit[0] is allowed_ids:
            return hit[1]
        positions = frozenset(
            pos
            for pos, node in enumerate(self._nodes)
            if not self._is_deleted(node) and node.metadata.get("id") in wanted
        )
        if len(self._allowed_positions_memo) >= 64:
            self._allowed_positions_memo.clear()
        self._allowed_positions_memo[key] = (allowed_ids, positions)
        return positions

    # ------------------------------------------------------------------ build
    def _chunk_to_node(self, chunk: Chunk, idx: int) -> TextNode:
        meta = dict(chunk.metadata)
        meta.setdefault("source", chunk.source)
        # Internal positional key; never collides with app-supplied "id".
        meta["_pos"] = idx
        return TextNode(text=chunk.text, metadata=meta)

    def build(self, chunks: list[Chunk]) -> None:
        nodes = [self._chunk_to_node(c, i) for i, c in enumerate(chunks)]
        index = self._dense_index(self._embed_document_vectors([n.text for n in nodes]))
        with self._state_lock:
            self._nodes = nodes
            self._index = index
            self._build_bm25()
            self._invalidate_scoped_search()

    @staticmethod
    def _normalize_vectors(
        vecs: "np.ndarray", *, expected_count: Optional[int] = None
    ) -> "np.ndarray":
        """Return contiguous row-wise unit vectors for cosine FAISS search."""
        arr = np.asarray(vecs, dtype="float32")
        if arr.ndim == 1:
            if expected_count not in (None, 1):
                raise ValueError(
                    "Embedding provider returned one vector for "
                    f"{expected_count} texts"
                )
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                "Embedding provider must return a two-dimensional array; "
                f"got shape {arr.shape}"
            )
        if expected_count is not None and arr.shape[0] != expected_count:
            raise ValueError(
                "Embedding provider returned "
                f"{arr.shape[0]} vectors for {expected_count} texts"
            )
        if arr.shape[1] == 0:
            raise ValueError("Embedding provider returned zero-width vectors")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.ascontiguousarray(arr / norms)

    def _embed_queries(self, texts: list[str]) -> "np.ndarray":
        method = getattr(self.embedder, "embed_queries", None)
        return (method or self.embedder.embed)(texts)

    def _embed_documents(self, texts: list[str]) -> "np.ndarray":
        method = getattr(self.embedder, "embed_documents", None)
        return (method or self.embedder.embed)(texts)

    def _embed_document_vectors(self, texts: list[str]) -> Optional["np.ndarray"]:
        """Embed and normalise document texts; ``None`` for an empty batch.

        Network I/O: callers must not hold ``_state_lock``.
        """
        if not texts:
            return None
        return self._normalize_vectors(
            self._embed_documents(texts), expected_count=len(texts)
        )

    @staticmethod
    def _dense_index(vecs: Optional["np.ndarray"]) -> Optional[faiss.Index]:
        if vecs is None:
            return None
        index = faiss.IndexFlatIP(int(vecs.shape[1]))
        index.add(np.ascontiguousarray(vecs))
        return index

    def _rebuild_dense(self) -> None:
        """Re-embed every node and install a positionally aligned dense index.

        Embeds a snapshot outside the state lock and installs it only if no
        write happened meanwhile; see ``_MAX_SNAPSHOT_ATTEMPTS``.
        """
        for _ in range(_MAX_SNAPSHOT_ATTEMPTS - 1):
            with self._state_lock:
                revision = self._index_revision
                texts = [n.text for n in self._nodes]
            index = self._dense_index(self._embed_document_vectors(texts))
            with self._state_lock:
                if self._index_revision == revision:
                    self._index = index
                    self._invalidate_scoped_search()
                    return
        with self._state_lock:
            self._build_faiss()
            self._invalidate_scoped_search()

    def _dense_accepts_append(self, vecs: "np.ndarray") -> bool:
        """Whether ``vecs`` can be appended to the current index in place."""
        if self._index is None:
            return not self._nodes
        return (
            int(self._index.d) == int(vecs.shape[1])
            and int(self._index.ntotal) == len(self._nodes)
        )

    def _append_locked(
        self, chunks: list[Chunk], index: Optional[faiss.Index], vecs: Optional["np.ndarray"]
    ) -> None:
        """Append ``chunks`` and either extend the index with ``vecs`` or
        replace it with ``index`` (which already covers every node)."""
        start = len(self._nodes)
        self._nodes.extend(self._chunk_to_node(c, start + i) for i, c in enumerate(chunks))
        if vecs is None:
            self._index = index
        elif self._index is None:
            self._index = self._dense_index(vecs)
        else:
            self._index.add(np.ascontiguousarray(vecs))
        self._build_bm25()
        self._invalidate_scoped_search()

    def add_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new_texts = [c.text for c in chunks]
        # A failed embedding leaves the in-memory index untouched; SQLite
        # remains the source of truth and a later rebuild recovers the row.
        vecs = self._embed_document_vectors(new_texts)
        assert vecs is not None
        for attempt in range(_MAX_SNAPSHOT_ATTEMPTS):
            with self._state_lock:
                if self._dense_accepts_append(vecs):
                    self._append_locked(chunks, None, vecs)
                    return
                if attempt == 0:
                    # The live embedding service may change dimensions after
                    # an index was loaded, and cardinality can be stale after
                    # an interrupted mutation (or nodes may exist without a
                    # dense index). Positional alignment then requires
                    # re-embedding every node as one snapshot; concatenating
                    # separate responses is unsafe if the dimension changed.
                    warnings.warn(
                        "HybridSearch dense index is incompatible with the new "
                        "document vectors; rebuilding all document vectors.",
                        stacklevel=2,
                    )
                if attempt == _MAX_SNAPSHOT_ATTEMPTS - 1:
                    start = len(self._nodes)
                    self._nodes.extend(
                        self._chunk_to_node(c, start + i) for i, c in enumerate(chunks)
                    )
                    try:
                        # `_build_faiss` assigns the index only on success.
                        self._build_faiss()
                    except BaseException:
                        del self._nodes[start:]
                        raise
                    self._build_bm25()
                    self._invalidate_scoped_search()
                    return
                revision = self._index_revision
                texts = [n.text for n in self._nodes] + new_texts
            index = self._dense_index(self._embed_document_vectors(texts))
            with self._state_lock:
                if self._index_revision == revision:
                    self._append_locked(chunks, index, None)
                    return

    @staticmethod
    def _is_deleted(node: TextNode) -> bool:
        return bool(node.metadata.get("_deleted", False))

    @property
    @_synchronized
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
        needs_dense_rebuild = False
        with self._state_lock:
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
                _, needs_dense_rebuild = self._cleanup_locked()
            self._invalidate_scoped_search()
        if needs_dense_rebuild:
            self._rebuild_dense()
        return deleted

    def cleanup(self) -> int:
        """Physically remove tombstones without re-embedding healthy vectors."""
        with self._state_lock:
            deleted, needs_dense_rebuild = self._cleanup_locked()
        if needs_dense_rebuild:
            self._rebuild_dense()
        return deleted

    def _cleanup_locked(self) -> tuple[int, bool]:
        """Compact tombstones; return ``(removed, needs_dense_rebuild)``.

        When the dense index cannot be reconstructed positionally it is
        dropped, and the caller re-embeds via :meth:`_rebuild_dense` after
        releasing the lock.
        """
        deleted = self.deleted_count
        if not deleted:
            return 0, False
        active_positions = [
            pos for pos, node in enumerate(self._nodes) if not self._is_deleted(node)
        ]
        if not active_positions:
            self._nodes = []
            self._bm25 = None
            self._index = None
            self._invalidate_scoped_search()
            return deleted, False

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
        needs_dense_rebuild = False
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
            # correctness recovery, not routine cleanup behavior. Until it
            # completes, dense search is unavailable rather than misaligned.
            warnings.warn(
                "HybridSearch cleanup found an inconsistent dense index; "
                "re-embedding active documents.",
                stacklevel=3,
            )
            self._index = None
            needs_dense_rebuild = True
        self._invalidate_scoped_search()
        return deleted, needs_dense_rebuild

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

    def _scoped_bm25(
        self, allowed_positions: frozenset[int]
    ) -> Optional[BM25Retriever]:
        """Return a bounded cached lexical retriever for an allowed subset."""
        key = (self._index_revision, allowed_positions)
        cached = self._scoped_bm25_cache.get(key)
        if key in self._scoped_bm25_cache:
            self._scoped_bm25_cache.move_to_end(key)
            return cached
        nodes = [
            self._nodes[pos]
            for pos in sorted(allowed_positions)
            if 0 <= pos < len(self._nodes)
            and not self._is_deleted(self._nodes[pos])
        ]
        retriever = (
            BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=min(self.candidate_pool, len(nodes)),
                verbose=False,
            )
            if nodes
            else None
        )
        self._scoped_bm25_cache[key] = retriever
        self._scoped_bm25_cache.move_to_end(key)
        while len(self._scoped_bm25_cache) > 16:
            self._scoped_bm25_cache.popitem(last=False)
        return retriever

    def _build_faiss(self) -> None:
        # Embeds while the caller holds the state lock: only the last-resort
        # fallback after `_MAX_SNAPSHOT_ATTEMPTS` lost races uses this.
        if not self._nodes:
            self._index = None
            return
        vecs = self._normalize_vectors(
            self._embed_documents([n.text for n in self._nodes]),
            expected_count=len(self._nodes),
        )
        self._index = self._dense_index(vecs)

    # SEARCH
    def search(
        self,
        query: Query,
        k: int = 5,
        where: dict | None = None,
        allowed_ids: Optional[Iterable[Any]] = None,
    ) -> list[Hit]:
        """Return up to `k` hits for `query`, fusing across weighted queries.

        ``query`` may be a plain string or a list of ``(text, weight)`` pairs.
        Each query runs its own BM25 + dense pass; the positional result lists
        are fused with reciprocal rank fusion where a query of weight ``w``
        contributes ``w / (rrf_k + rank + 1)`` per hit. A single (or unweighted)
        query is the legacy path and ranks identically to before.
        ``allowed_ids`` optionally restricts both BM25 and FAISS to documents
        whose application-level ``metadata['id']`` is allowed. The lexical
        subset is cached and FAISS uses an ID selector, so disallowed entries
        are not merely removed after consuming the candidate pool.
        """
        if k <= 0:
            return []
        queries = [(text, weight) for text, weight in as_queries(query) if weight > 0.0]
        if not queries:
            return []
        wanted: Optional[set] = None
        if allowed_ids is not None:
            wanted = (
                {allowed_ids}
                if isinstance(allowed_ids, (str, bytes))
                else set(allowed_ids)
            )
        with self._state_lock:
            if self._searchable_count(allowed_ids, wanted) == 0:
                return []

        # Embed every query text in one batch (one round-trip for the dense
        # pass regardless of how many messages are in the window). This is
        # network I/O, so it happens before taking the state lock.
        q_texts = [q for q, _ in queries]
        q_vecs = self._query_vectors(q_texts)
        from .fusion import fuse
        for attempt in range(_MAX_SNAPSHOT_ATTEMPTS):
            with self._state_lock:
                active_count = self._searchable_count(allowed_ids, wanted)
                if active_count == 0:
                    return []
                if (
                    self._index is not None
                    and int(self._index.d) == int(q_vecs.shape[1])
                    and int(self._index.ntotal) == len(self._nodes)
                ):
                    allowed_positions = (
                        self._allowed_positions_for(allowed_ids, wanted)
                        if wanted is not None
                        else None
                    )
                    pool = min(max(self.candidate_pool, k * 3), active_count)
                    candidates = [
                        self._query_candidates(text, vector, pool, where, allowed_positions)
                        for (text, _), vector in zip(queries, q_vecs)
                    ]
                    return fuse(queries, candidates, k, self.rrf_k, self.min_dense_similarity,
                                self._result_key,
                                lambda pos: (self._nodes[pos].text,
                                             self._nodes[pos].metadata.get("source", ""),
                                             self._nodes[pos].metadata))
                if attempt == _MAX_SNAPSHOT_ATTEMPTS - 1:
                    index_dim = None if self._index is None else int(self._index.d)
                    break
                if attempt == 0:
                    # Cover dimension drift that happens after load but before
                    # the next write, interrupted legacy indexes with a stale
                    # vector count, and nodes left without a dense index by a
                    # failed recovery. Query and document vectors must share
                    # one live dimensionality before FAISS can search them.
                    warnings.warn(
                        "HybridSearch dense index is incompatible with the live "
                        "query vectors; rebuilding all document vectors.",
                        stacklevel=2,
                    )
            self._rebuild_dense()
            with self._state_lock:
                index_dim = None if self._index is None else int(self._index.d)
            if index_dim is not None and index_dim != int(q_vecs.shape[1]):
                # An asymmetric provider may update its model state during
                # document embedding. Refresh queries before declaring the
                # provider contract inconsistent.
                q_vecs = self._normalize_vectors(
                    self._embed_queries(q_texts), expected_count=len(q_texts)
                )
        raise ValueError(
            "Embedding provider returned incompatible query and "
            f"document dimensions ({q_vecs.shape[1]} and {index_dim})"
        )

    def _searchable_count(self, allowed_ids: object, wanted: Optional[set]) -> int:
        """Active documents visible to a search; caller holds the state lock."""
        if wanted is None:
            return self.count
        return len(self._allowed_positions_for(allowed_ids, wanted))

    def _result_key(self, pos: int) -> Hashable:
        """Return a retrieval-group key without changing positional IDs."""
        result_id = self._nodes[pos].metadata.get("_result_id")
        return ("result", result_id) if result_id is not None else ("_pos", pos)

    def _query_candidates(
        self,
        qtext: str,
        qv: "np.ndarray",
        pool: int,
        where: dict | None,
        allowed_positions: Optional[frozenset[int]] = None,
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
        """Run one query's BM25 + dense passes, returning ranked positional hits.

        Returns ``(bm25_hits, dense_hits)`` where each entry is ``(positional,
        rank, score)``. Zero-score lexical results and dense results below the
        configured cosine floor are excluded before RRF. `where` filters on
        metadata equality.
        """
        # Lexical candidates, keyed by internal positional index.
        bm25_hits: list[tuple[int, int, float]] = []
        bm25 = (
            self._scoped_bm25(allowed_positions)
            if allowed_positions is not None
            else self._bm25
        )
        if bm25 is not None:
            nodes = bm25.retrieve(qtext)
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
        if allowed_positions is not None:
            dense_pool = min(len(allowed_positions), pool)
            selected = np.asarray(sorted(allowed_positions), dtype="int64")
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(selected)
            sims, idxs = self._index.search(
                np.ascontiguousarray(qv.reshape(1, -1)),
                dense_pool,
                params=params,
            )
        else:
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
    @_synchronized
    def count(self) -> int:
        return len(self._nodes) - self.deleted_count

    @property
    @_synchronized
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

    @_synchronized
    def persist(self, path: str) -> None:
        """Persist nodes + dense index to `path` atomically and cross-process safe.

        Writes both files via temp + ``os.replace`` (POSIX-atomic), under an
        exclusive ``flock`` on ``<path>/.lock``. This prevents torn writes and
        interleaved writes when the cm_server and the Discord bot persist the
        same index concurrently. The lock is per index directory, matching the
        existing per-character in-process lock convention; it is auto-released
        by the OS on close/exit, so a crash can't leave it stuck.

        A no-op when nothing changed since this instance last wrote or loaded
        `path` and its ``nodes.json`` still exists, so callers may persist
        every index after any write without rewriting untouched ones.
        """
        marker = self._persist_key(path)
        if marker == self._persisted_marker and os.path.exists(
            os.path.join(path, "nodes.json")
        ):
            return
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
            # Still under the exclusive lock: no other writer can have
            # replaced the file since, so this is exactly our publication.
            self._record_synced(path)
        self._persisted_marker = marker

    def _persist_key(self, path: str) -> tuple[str, int, str]:
        return (os.path.realpath(path), self._index_revision, self._index_fingerprint())

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
                    loaded_mtime_ns = os.fstat(f.fileno()).st_mtime_ns
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

        # Build the new snapshot privately and install it in one step at the
        # end: searches keep serving the previous (self-consistent) nodes and
        # vectors while a repair is embedding, and a failed repair leaves
        # them untouched instead of pairing new nodes with an old index.
        nodes: list[TextNode] = []
        for i, entry in enumerate(data):
            meta = dict(entry.get("metadata") or {})
            if "id" in entry:
                meta["id"] = entry["id"]
            meta.setdefault("source", entry.get("source", ""))
            meta["_pos"] = i  # recompute canonical positional index
            nodes.append(TextNode(text=entry["text"], metadata=meta))
        rebuild = True
        if loaded is not None:
            # May probe the embedding endpoint; no lock is held here.
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
                and int(loaded.ntotal) == len(nodes)
                and metadata_matches
            ):
                rebuild = False
            else:
                # Embedding behavior/dim drifted, the metadata is legacy, or
                # nodes/index cardinality diverged.
                print(
                    f"[rag] dense-index mismatch (index_dim={loaded.d}, "
                    f"embedder_dim={expected}, index_count={loaded.ntotal}, "
                    f"node_count={len(nodes)}, "
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
        index: Optional[faiss.Index] = loaded
        if rebuild:
            # Tombstones do not need new embeddings during a repair. Drop them
            # physically before recreating the dense index.
            active = [node for node in nodes if not self._is_deleted(node)]
            nodes = []
            for pos, node in enumerate(active):
                meta = dict(node.metadata)
                meta.pop("_deleted", None)
                meta["_pos"] = pos
                nodes.append(TextNode(text=node.text, metadata=meta))
            index = self._dense_index(
                self._embed_document_vectors([n.text for n in nodes])
            )
        with self._state_lock:
            self._nodes = nodes
            self._index = index
            self._build_bm25()
            self._invalidate_scoped_search()
            if rebuild:
                # Persist both corrected nodes and index so later loads are
                # fast, including the previously-missing-index case.
                self.persist(path)
            else:
                self._persisted_marker = self._persist_key(path)
                self._record_synced(path, loaded_mtime_ns)
