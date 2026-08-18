"""Abstract embedding provider.

Subclass to plug in a different embedding backend. The RAG layer and the
structured memories talk only to this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Maps text to fixed-dimension float vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vectors produced."""

    @abstractmethod
    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Return a ``(n, dim)`` float32 array."""

    def embed_queries(self, texts: str | list[str]) -> np.ndarray:
        """Embed retrieval queries.

        Symmetric embedding models need no special handling, so the default
        delegates to :meth:`embed`.  Asymmetric providers may override this
        method without leaking model-specific behavior into the RAG layer.
        """
        return self.embed(texts)

    def embed_documents(self, texts: str | list[str]) -> np.ndarray:
        """Embed documents stored in a retrieval index.

        The default preserves the legacy symmetric behavior. Providers whose
        models distinguish queries from passages can override this method.
        """
        return self.embed(texts)

    @property
    def index_fingerprint(self) -> Optional[str]:
        """Stable, credential-free identity for persisted dense vectors.

        Returning ``None`` opts a custom provider out of provider-specific
        compatibility checks; HybridSearch still validates schema, dimension,
        vector count and its own normalization policy.
        """
        return None
