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
from .base import Memory, MemoryItem, MemoryScope, RecallResult

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
        token_budget: int = 1_000,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(enabled=enabled, name=name)
        self.store = store
        self.hybrid = hybrid
        self.token_budget = max(0, int(token_budget))
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
        """Recall nodes within ``limit`` prompt tokens.

        ``Memory`` names its generic per-memory bound ``limit``; for the
        knowledge graph that value is deliberately a token budget rather than
        a node count.
        """
        if not self.enabled:
            return []
        return self.retriever.retrieve(
            query,
            user_id=user_id,
            token_budget=limit,
            state_changing=state_changing,
        )

    def _activation_snapshot(self, items: list[MemoryItem]) -> dict[str, Any]:
        """Copy the activation state produced by the just-finished recall.

        The copy is request-local: the graph's transient node fields may be
        overwritten by a later group participant or request, so the live
        monitor must never read them after the context call returns.
        """
        activation = {
            str(nid): float(getattr(node, "activation", 0.0))
            for nid, node in self.retriever.graph.nodes.items()
        }
        breakdowns = {}
        for nid, node in self.retriever.graph.nodes.items():
            comp = getattr(node, "score_breakdown", None)
            if isinstance(comp, dict) and comp:
                breakdowns[str(nid)] = dict(comp)
        surfaced = [
            str(item.metadata.get("node_id"))
            for item in items
            if item.metadata.get("node_id") is not None
        ]
        return {
            "activation": activation,
            "breakdowns": breakdowns,
            "retrieved_ids": list(dict.fromkeys(surfaced)),
        }

    def build_section_result(
        self,
        query: str,
        user_id: str,
        limit: int,
        state_changing: bool = True,
    ) -> RecallResult:
        result = super().build_section_result(
            query, user_id, limit, state_changing=state_changing
        )
        if result.items:
            result.diagnostics = self._activation_snapshot(result.items)
        return result

    def build_section_participants_result(
        self,
        query: str,
        participants: list[str],
        limit: int,
        state_changing: bool = True,
    ) -> RecallResult:
        if not participants:
            return RecallResult()
        if len(participants) == 1:
            return self.build_section_result(
                query, participants[0], limit, state_changing=state_changing
            )
        if not self.enabled:
            return RecallResult()

        items: list[MemoryItem] = []
        by_participant: dict[str, dict[str, Any]] = {}
        for uid in participants:
            result = self.build_section_result(
                query, uid, limit, state_changing=state_changing
            )
            items.extend(result.items)
            if result.items:
                by_participant[uid] = result.diagnostics
        if not items:
            return RecallResult()
        body = self.format_grouped(items, participants)
        return RecallResult(
            items=items,
            body=body,
            diagnostics={"participants": by_participant},
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
        return self.retriever.format_items(items)

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
