"""OpenAI-compatible embedding client (works with any /v1/embeddings server)."""

import hashlib
import json
from typing import Optional

import numpy as np
from openai import OpenAI
from ..config import EmbeddingConfig
from .embedding_base import EmbeddingProvider


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
        # small query batch. Keep a tiny process-local cache so only the first
        # one reaches the embedding server; never retain bulk indexing batches.
        self._cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

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
        key = (role, tuple(texts))
        cacheable = len(texts) <= 8
        if cacheable:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy()

        vecs: list[list[float]] = []
        bs = self.config.batch_size
        for i in range(0, len(texts), bs):
            batch = [prefix + text for text in texts[i : i + bs]]
            resp = self._client.embeddings.create(model=self.config.model, input=batch)
            vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > 0:
            self._dim = int(arr.shape[1])
        if cacheable:
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = arr.copy()
        return arr
