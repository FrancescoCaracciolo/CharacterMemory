"""Abstract RAG system.

A `RAGSystem` indexes text and returns ranked hits for a query.
Subclass it to implement a different retrieval strategy
The memory layer and the agent only ever talk to this interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Union

from ..chunking.base import Chunk

# A retrieval query. Either a plain string (single query, weight 1.0) or a
# list of ``(text, weight)`` pairs — e.g. one per recent chat message, with
# older messages weighted less. Weights scale each query's contribution to the
# fused ranking. A single bare string is the legacy/common path.
WeightedQuery = tuple[str, float]
Query = Union[str, list[WeightedQuery]]


def as_queries(query: Query) -> list[WeightedQuery]:
    """Normalize a ``Query`` into a ``[(text, weight), ...]`` list.

    A bare string becomes ``[(query, 1.0)]``; a weighted list is copied with
    empty/whitespace-only texts dropped. Weights are clamped to be
    non-negative (a zero-weight query is a harmless no-op; a negative weight
    would invert ranking, which we never want).
    """
    if isinstance(query, str):
        return [(query, 1.0)] if query.strip() else []
    out: list[WeightedQuery] = []
    for q, w in query:
        if not isinstance(q, str) or not q.strip():
            continue
        out.append((q, max(0.0, float(w))))
    return out


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

    def delete_documents(self, ids: Iterable[Any]) -> int:
        """Logically delete documents whose application ``metadata['id']`` matches.

        Mutable backends may implement this without physically rebuilding the
        index.  The default keeps existing third-party RAG implementations
        source-compatible; callers that need deletion must fall back to
        :meth:`build` when this method raises ``NotImplementedError``.

        Returns the number of documents newly marked as deleted.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support deletion")

    def cleanup(self) -> int:
        """Physically compact logically-deleted documents, returning the count.

        Implementations should preserve existing embeddings where possible.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support cleanup")

    @abstractmethod
    def search(self, query: Query, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Return up to `k` hits for `query`.

        `query` may be a plain string or a list of ``(text, weight)`` pairs;
        a weighted list runs one search per query and fuses the results with
        weight-scaled reciprocal rank fusion. `where` optionally filters on
        metadata equality (e.g. `{"user_id": "alice"}`).
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
