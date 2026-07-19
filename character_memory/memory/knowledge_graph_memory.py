"""`Memory` wrapper around the knowledge-graph retriever.

This is the adapter that lets the :class:`KnowledgeGraphRetriever` plug into
the agent like any other memory: it implements :class:`Memory`, renders an
"Activated knowledge" prompt section, and routes persistence through the
retriever. The retriever itself stays free of agent concerns.

The memory keeps a back-reference to the *other* memories on the agent so
the incremental :meth:`KnowledgeGraphRetriever.update` and
:meth:`apply_deduplication` paths can resolve freshly-added items and
dedupped source rows to graph nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..chunking import Chunk
from ..knowledge_graph import (
    KnowledgeGraphConfig,
    KnowledgeGraphRetriever,
)
from ..memory.store import SQLiteStore
from ..rag.hybrid import HybridSearch
from .base import Memory, MemoryItem, MemoryScope

if TYPE_CHECKING:  # avoid a circular import at runtime
    from ..memory.dedup import DedupReport


class KnowledgeGraphMemory(Memory):
    """A memory whose recall is a knowledge-graph activation search.

    `scope = PER_USER` so the agent recalls once per participant; the
    queried user's PersonNode gets an activation bias in the retriever.
    """

    name = "knowledge_graph"
    scope = MemoryScope.PER_USER

    def __init__(
        self,
        store: SQLiteStore,
        hybrid: HybridSearch,
        *,
        enabled: bool = True,
        config: Optional[KnowledgeGraphConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(enabled=enabled, name=name)
        self.store = store
        self.hybrid = hybrid
        self.retriever = KnowledgeGraphRetriever(config=config)
        # Source memories are wired by the agent after construction (see
        # :meth:`wire_sources`); the retriever's update/dedup paths need them.
        self._sources: dict[str, Any] = {}

    # ----------------------------------------------------------- wiring hooks
    def wire_sources(self, memories: dict[str, Any]) -> None:
        """Register the agent's other memories so incremental ingest works.

        The agent calls this once during build; it lets the retriever resolve
        a freshly-added `MemoryItem` (carrying `metadata.id`) back to its
        source row for the `update` / `apply_deduplication` paths.
        """
        self._sources = {k: v for k, v in memories.items() if k != self.name}

    def _source_memory_for(self, name: str) -> Optional[Any]:
        return self._sources.get(name)

    # --------------------------------------------------------------- lifecycle
    def attach_backends(self, llm: Any, embedder: Any) -> None:
        """Wire the LLM/embedder into the retriever (called by the agent)."""
        self.retriever.load(llm, embedder, self.hybrid, store=self.store)
        # Re-bind the source-memory resolver so update/dedup work post-attach.
        self.retriever._source_memory_for = self._source_memory_for  # type: ignore[assignment]

    # ---------------------------------------------------------------- Memory API
    def recall(
        self,
        query: str,
        user_id: str,
        limit: int,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        if not self.enabled:
            return []
        return self.retriever.retrieve(
            query, user_id=user_id, limit=limit, state_changing=state_changing
        )

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        """Every node, rendered as an item (for the GUI's generic browse mode)."""
        items: list[MemoryItem] = []
        for node in list(self.retriever.graph.nodes.values())[: limit or None]:
            items.append(self.retriever._node_to_item(node, node.activation))
        return items

    def format(self, items: list[MemoryItem]) -> str:
        if not items:
            return ""
        # Group by node kind so the prompt reads cleanly: people first, then
        # facts, then episodes, then entities.
        order = {"person": 0, "fact": 1, "episode": 2, "entity": 3, "self": 4}
        lines = sorted(items, key=lambda it: (order.get(it.metadata.get("node_kind"), 9), -it.score))
        return "\n".join(f"- {it.text}" for it in lines)

    # ----------------------------------------------- build / persist / load stubs
    def build(self, info_chunks: list[Chunk]) -> None:
        # The graph is built from the source memories, not from info chunks;
        # the agent drives ingestion explicitly. This is a no-op for the
        # Memory ABC contract.
        return None

    def persist(self, path: str) -> None:
        self.retriever.save(path)

    def load(self, path: str) -> None:
        self.retriever.load_persisted(path)

    @property
    def has_persisted(self) -> bool:
        # Delegated to the retriever once it knows the index path; the agent
        # passes that path explicitly at build time.
        return False


__all__ = ["KnowledgeGraphMemory"]
