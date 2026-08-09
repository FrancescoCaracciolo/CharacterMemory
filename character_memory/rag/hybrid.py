"""Hybrid retrieval: BM25 (lexical) + FAISS (dense similarity), fused with RRF.

Uses llama-index as a library (`TextNode` schema + `BM25Retriever`) and a
FAISS vector store for dense similarity. Scores are combined with Reciprocal
Rank Fusion (RRF), so the two signals need no calibration.

A single `HybridSearch` instance is reused by the RAG-backed memories
(character info, dialogue style) and by the structured memories (facts, etc.)
for their BM25+similarity recall.
"""

import json
import os
from collections.abc import Hashable
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
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
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

    def add_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        start = len(self._nodes)
        new_nodes = [self._chunk_to_node(c, start + i) for i, c in enumerate(chunks)]
        self._nodes.extend(new_nodes)
        # Append dense vectors directly; rebuild the lexical index cheaply.
        vecs = self.embedder.embed([n.text for n in new_nodes]).astype("float32")
        if self._index is None:
            self._nodes = new_nodes
            self._build_faiss()
        else:
            self._index.add(np.ascontiguousarray(vecs))
        self._build_bm25()

    def _build_bm25(self) -> None:
        if not self._nodes:
            self._bm25 = None
            return
        # Cap top_k at the node count so BM25 never emits its override warning.
        top_k = min(self.candidate_pool, len(self._nodes))
        self._bm25 = BM25Retriever.from_defaults(nodes=self._nodes, similarity_top_k=top_k, verbose=False)

    def _build_faiss(self) -> None:
        if not self._nodes:
            self._index = None
            return
        vecs = self.embedder.embed([n.text for n in self._nodes]).astype("float32")
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
        if not self._nodes or self._index is None:
            return []
        queries = as_queries(query)
        if not queries:
            return []
        pool = min(max(self.candidate_pool, k * 3), len(self._nodes))

        # Embed every query text in one batch (one round-trip for the dense
        # pass regardless of how many messages are in the window).
        q_texts = [q for q, _ in queries]
        q_vecs = self.embedder.embed(q_texts).astype("float32")
        if q_vecs.ndim == 1:  # embed() squeezed a single query to (dim,).
            q_vecs = q_vecs.reshape(1, -1)

        # Reciprocal Rank Fusion over result groups, weight-scaled. Normal
        # documents form one group per positional node. A memory may stamp
        # several index keys with the same `_result_id`; those keys then rank
        # as one result and cannot consume several top-k slots.
        scores: dict[Hashable, float] = {}
        representative: dict[Hashable, int] = {}
        sim_by_result: dict[Hashable, float] = {}
        for (qtext, weight), qv in zip(queries, q_vecs):
            bm25_hits, dense_hits = self._query_candidates(qtext, qv, pool, where)
            for pos, rank in bm25_hits:
                key = self._result_key(pos)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                representative.setdefault(key, pos)
            for pos, rank, sim in dense_hits:
                key = self._result_key(pos)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                representative.setdefault(key, pos)
                if sim > sim_by_result.get(key, -1.0):
                    sim_by_result[key] = sim

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits: list[Hit] = []
        for key, score in ranked:
            pos = representative[key]
            node = self._nodes[pos]
            meta = dict(node.metadata)
            meta["similarity"] = sim_by_result.get(key, 0.0)
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
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
        """Run one query's BM25 + dense passes, returning ranked positional hits.

        Returns ``(bm25_hits, dense_hits)`` where each entry is ``(positional,
        rank)`` (BM25) or ``(positional, rank, similarity)`` (dense). `where`
        filters on metadata equality. Used by :meth:`search` for each weighted
        query; the caller fuses the lists with weight-scaled RRF.
        """
        # Lexical candidates, keyed by internal positional index.
        bm25_hits: list[tuple[int, int]] = []
        if self._bm25 is not None:
            nodes = self._bm25.retrieve(qtext)
            rank = 0
            seen_results: set[Hashable] = set()
            for n in nodes:
                if not _matches(where, n.metadata):
                    continue
                pos = int(n.metadata.get("_pos", -1))
                if pos < 0 or pos >= len(self._nodes):
                    continue
                key = self._result_key(pos)
                if key in seen_results:
                    continue
                seen_results.add(key)
                bm25_hits.append((pos, rank))
                rank += 1
                if len(bm25_hits) >= pool:
                    break

        # Dense search (ranked by cosine similarity via inner product).
        # `qv` is a single 1-D row vector; FAISS needs a 2-D (1, dim) array.
        sims, idxs = self._index.search(np.ascontiguousarray(qv.reshape(1, -1)), pool)
        dense_hits: list[tuple[int, int, float]] = []
        n_nodes = len(self._nodes)
        rank = 0
        seen_results: set[Hashable] = set()
        for nid, sim in zip(idxs[0], sims[0]):
            if nid < 0:
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
        return len(self._nodes)

    @property
    def documents(self) -> list[Chunk]:
        """Return all indexed chunks."""
        return [
            Chunk(text=n.text, source=n.metadata.get("source", ""), metadata=dict(n.metadata))
            for n in self._nodes
        ]

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
        lock_path = os.path.join(path, ".lock")
        # "a+" keeps an existing lock file without truncating it; the file's
        # contents are never read, it only anchors the flock.
        with open(lock_path, "a+") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            nodes_tmp = os.path.join(path, "nodes.json.tmp")
            with open(nodes_tmp, "w", encoding="utf-8") as f:
                json.dump(nodes_json, f, ensure_ascii=False)
            os.replace(nodes_tmp, os.path.join(path, "nodes.json"))
            if self._index is not None:
                idx_tmp = os.path.join(path, "faiss.index.tmp")
                faiss.write_index(self._index, idx_tmp)
                os.replace(idx_tmp, os.path.join(path, "faiss.index"))

    def load(self, path: str) -> None:
        with open(os.path.join(path, "nodes.json"), encoding="utf-8") as f:
            data = json.load(f)
        self._nodes = []
        for i, entry in enumerate(data):
            meta = dict(entry.get("metadata") or {})
            if "id" in entry:
                meta["id"] = entry["id"]
            meta.setdefault("source", entry.get("source", ""))
            meta["_pos"] = i  # recompute canonical positional index
            self._nodes.append(TextNode(text=entry["text"], metadata=meta))
        self._build_bm25()
        idx_file = os.path.join(path, "faiss.index")
        rebuild = True
        if os.path.exists(idx_file):
            loaded = faiss.read_index(idx_file)
            expected = getattr(self.embedder, "dim", None)
            if expected is not None and loaded.d == expected:
                self._index = loaded
                rebuild = False
            else:
                # Embedding dim drifted (e.g. server changed model) since the
                # index was built. Rebuild the dense index from stored text so
                # query vectors line up
                print(
                    f"[rag] dense-index dim mismatch (index={loaded.d}, "
                    f"embedder={expected}); rebuilding from stored nodes."
                )
        if rebuild:
            self._build_faiss()
            # Persist the corrected index so later loads are fast.
            if os.path.exists(idx_file):
                idx_tmp = idx_file + ".tmp"
                faiss.write_index(self._index, idx_tmp)
                os.replace(idx_tmp, idx_file)
