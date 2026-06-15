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
from typing import Any, Optional

import faiss
import numpy as np

from ..chunking.base import Chunk
from ..llm.embeddings_base import EmbeddingProvider
from .base import Hit, RAGSystem

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
    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        if not self._nodes or self._index is None:
            return []
        pool = min(max(self.candidate_pool, k * 3), len(self._nodes))

        # Lexical candidates (ranked), keyed by internal positional index.
        bm25_hits: list[tuple[int, int]] = []  # (positional, rank)
        if self._bm25 is not None:
            nodes = self._bm25.retrieve(query)
            rank = 0
            for n in nodes:
                if not _matches(where, n.metadata):
                    continue
                bm25_hits.append((int(n.metadata.get("_pos", -1)), rank))
                rank += 1
                if len(bm25_hits) >= pool:
                    break

        # Dense saerch (ranked by cosine similarity via inner product).
        qv = self.embedder.embed(query).astype("float32")
        sims, idxs = self._index.search(np.ascontiguousarray(qv), pool)
        dense_hits: list[tuple[int, int, float]] = []  # (positional, rank, sim)
        rank = 0
        for nid, sim in zip(idxs[0], sims[0]):
            if nid < 0:
                continue
            node = self._nodes[nid]
            if not _matches(where, node.metadata):
                continue
            dense_hits.append((int(nid), rank, float(sim)))
            rank += 1

        # Reciprocal Rank Fusion over positional indices.
        scores: dict[int, float] = {}
        for pos, rank in bm25_hits:
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for pos, rank, _sim in dense_hits:
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Dense similarity is kept on the hit for display/debugging.
        sim_by_pos = {pos: sim for pos, _r, sim in dense_hits}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits: list[Hit] = []
        for pos, score in ranked:
            node = self._nodes[pos]
            meta = dict(node.metadata)
            meta["similarity"] = sim_by_pos.get(pos, 0.0)
            hits.append(
                Hit(text=node.text, score=score, source=meta.get("source", ""), metadata=meta)
            )
        return hits

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
        os.makedirs(path, exist_ok=True)
        nodes_json = [
            {"id": n.metadata.get("id", i), "text": n.text,
             "source": n.metadata.get("source", ""), "metadata": n.metadata}
            for i, n in enumerate(self._nodes)
        ]
        with open(os.path.join(path, "nodes.json"), "w", encoding="utf-8") as f:
            json.dump(nodes_json, f, ensure_ascii=False)
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(path, "faiss.index"))

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
                faiss.write_index(self._index, idx_file)
