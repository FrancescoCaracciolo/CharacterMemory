"""OpenAI-compatible embedding client (works with any /v1/embeddings server)."""

import hashlib
import json
from typing import Optional

import numpy as np
from openai import OpenAI
from .._timing import time_phase
from ..config import EmbeddingConfig
from .embedding_base import EmbeddingProvider

# Small query batches are cached per text; the bound keeps a few days of a
# sliding history window resident (texts repeat across turns) without ever
# retaining bulk indexing batches.
_TEXT_CACHE_LIMIT = 512


class OpenAICompatibleEmbeddings(EmbeddingProvider):
    """Embedding provider over an OpenAI-compatible ``/v1`` endpoint."""

    def __init__(self, config: Optional[EmbeddingConfig] = None, **overrides) -> None:
        cfg = config or EmbeddingConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.config = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
        self._dim: Optional[int] = cfg.dim
        # Context assembly asks several independent indexes to embed the same
        # small query batch, and the history-aware window slides by one
        # message per turn — so most texts were embedded on the previous
        # request. Cache per (role, prefix, text); only misses reach the
        # embedding server. Bulk indexing batches are never retained.
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed("hi").shape[1]
        return self._dim

    def embed(self, texts) -> np.ndarray:
        return self._embed_role(texts, role="symmetric", prefix="")

    def embed_queries(self, texts) -> np.ndarray:
        return self._embed_role(
            texts,
            role="query",
            prefix=self.config.retrieval_query_prefix,
        )

    def embed_documents(self, texts) -> np.ndarray:
        return self._embed_role(
            texts,
            role="document",
            prefix=self.config.retrieval_document_prefix,
        )

    @property
    def index_fingerprint(self) -> str:
        """Hash only settings that affect retrieval/index compatibility."""
        payload = {
            "provider": f"{type(self).__module__}.{type(self).__qualname__}",
            "base_url": str(self.config.base_url).rstrip("/"),
            "model": self.config.model,
            # `_dim` is populated by every successful embedding request.  Do
            # not force a network request merely to fingerprint an empty
            # index; a loaded non-empty FAISS index is dimension-checked by
            # HybridSearch before this property is read.
            "dim": self._dim,
            "retrieval_query_prefix": self.config.retrieval_query_prefix,
            "retrieval_document_prefix": self.config.retrieval_document_prefix,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _embed_role(self, texts, *, role: str, prefix: str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        cacheable = len(texts) <= 8
        rows: list[Optional[np.ndarray]] = [None] * len(texts)
        misses: list[int] = list(range(len(texts)))
        if cacheable:
            misses = []
            for i, text in enumerate(texts):
                cached = self._cache.get((role, prefix, text))
                if cached is not None:
                    rows[i] = cached
                else:
                    misses.append(i)

        if misses:
            pending = [prefix + texts[i] for i in misses]
            fetched: list[np.ndarray] = []
            bs = self.config.batch_size
            for j in range(0, len(pending), bs):
                batch = pending[j : j + bs]
                with time_phase("embed"):
                    resp = self._client.embeddings.create(
                        model=self.config.model, input=batch
                    )
                fetched.append(
                    np.asarray(
                        [d.embedding for d in resp.data], dtype=np.float32
                    )
                )
            vectors = fetched[0] if len(fetched) == 1 else np.concatenate(fetched, axis=0)
            for offset, i in enumerate(misses):
                rows[i] = vectors[offset]
            if cacheable:
                for i in misses:
                    self._cache[(role, prefix, texts[i])] = rows[i]
                while len(self._cache) > _TEXT_CACHE_LIMIT:
                    self._cache.pop(next(iter(self._cache)))

        if not texts:
            return np.asarray([], dtype=np.float32)
        # np.stack copies into a fresh array, so callers can mutate the
        # result without corrupting the cached rows.
        arr = np.stack(rows)
        if arr.ndim == 2 and arr.shape[0] > 0:
            self._dim = int(arr.shape[1])
        return arr
