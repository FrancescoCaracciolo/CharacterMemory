"""Abstract RAG system.

A `RAGSystem` indexes text and returns ranked hits for a query.
Subclass it to implement a different retrieval strategy
The memory layer and the agent only ever talk to this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..chunking.base import Chunk


@dataclass
class Hit:
    """A single retrieval result."""

    text: str
    score: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> Any:
        return self.metadata.get("id")


class RAGSystem(ABC):
    """Base class for retrieval-augmentation back-ends."""

    name: str = "base"

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """Build the index from `chunks` (replacing any existing index)."""

    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> None:
        """Add `chunks` to an existing index."""

    @abstractmethod
    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Return up to `k` hits for `query`.

        `where` optionally filters on metadata equality (e.g.
        `{"user_id": "alice"}`).
        """

    @abstractmethod
    def persist(self, path: str) -> None:
        """Write the index to `path` (a directory)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index previously written by :meth:`persist`."""

    @property
    def count(self) -> int:
        """Number of indexed documents (best-effort)."""
        return 0

    @property
    def documents(self) -> list[Chunk]:
        """Return all indexed documents (best-effort)."""
        return []
