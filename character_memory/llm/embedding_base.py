"""Abstract embedding provider.

Subclass to plug in a different embedding backend. The RAG layer and the
structured memories talk only to this interface.
"""
from abc import ABC, abstractmethod
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
