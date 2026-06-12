"""OpenAI-compatible embedding client (works with any /v1/embeddings server)."""

from typing import Optional
import numpy as np
from openai import OpenAI
from ..config import EmbeddingConfig
from .embeddings_base import EmbeddingProvider


class OpenAICompatibleEmbeddings(EmbeddingProvider):
    """Embedding provider over an OpenAI-compatible ``/v1`` endpoint."""

    def __init__(self, config: Optional[EmbeddingConfig] = None, **overrides) -> None:
        cfg = config or EmbeddingConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.config = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
        self._dim: Optional[int] = cfg.dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed("hi").shape[1]
        return self._dim

    def embed(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        texts = list(texts)
        vecs: list[list[float]] = []
        bs = self.config.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            resp = self._client.embeddings.create(model=self.config.model, input=batch)
            vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype=np.float32)
        if single:
            arr = arr[:1]
        return arr
