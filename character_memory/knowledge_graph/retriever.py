"""The high-level retriever that wires ingestion + activation + retrieval.

This is the class the example API in the design doc drives::

    kg = KnowledgeGraphRetriever()
    kg.load(llm, embedder, HybridSearch(embedder))
    kg.ingest([UserFactMemory, EmotionMemory, ...])
    kg.update(extracted_items)
    kg.apply_deduplication(dedup_report)
    trace = kg.test_activation("message")
    items = kg.retrieve("message", user_id="...", limit=6)
    kg.save(save_dir)

It owns a :class:`KnowledgeGraph`, a node-text :class:`HybridSearch`, and an
optional :class:`SQLiteStore` + :class:`LLMClient`. Every public method is a
thin orchestrator over :mod:`ingest`, :mod:`activation`, and
:mod:`persistence`; the heavy lifting lives there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..llm.base import LLMClient
from ..llm.embedding_base import EmbeddingProvider
from ..memory.base import MemoryItem
from ..memory.dedup import DedupReport
from ..memory.emotion import EmotionStatus
from ..memory.episodic import EpisodicMemory
from ..memory.store import SQLiteStore
from ..memory.user_facts import UserFactMemory
from ..memory.user_summary import UserSummaryMemory
from ..rag.hybrid import HybridSearch
from .activation import combined_activation
from .edges import CoOccurrenceEdge
from .graph import KnowledgeGraph
from .ingest import (
    ingest_emotion,
    ingest_episodes,
    ingest_facts,
    ingest_summaries,
    ingest_wiki,
    ingest_wiki_llm,
)
from .nodes import Node
from .persistence import has_persisted, load_graph, save_graph


@dataclass
class KnowledgeGraphConfig:
    """Tunables for retrieval. Defaults are conservative."""

    #: ACT-R BLL decay parameter (d). Higher -> faster forgetting.
    decay: float = 0.5
    #: When > 0, an extra recency factor on top of BLL (seconds half-life).
    decay_half_life: float = 60 * 60 * 24 * 7  # one week
    #: Spreading activation gain (how much of `A_u` flows to neighbours).
    gain: float = 0.35
    #: Spreading hops. The design pins this at 2.
    hops: int = 2
    #: Per-hop attenuation.
    hop_decay: float = 0.6
    #: Weight of the BLL term in the combined score.
    base_weight: float = 1.0
    #: Weight of the spreading term in the combined score.
    spread_weight: float = 1.2
    #: Activation floor; nodes below this are not returned.
    min_activation: float = 0.0
    #: Hebbian: nodes both above this activation reinforce their co-edge.
    hebbian_threshold: float = 0.15
    #: Hebbian: how much a co-recall strengthens the edge weight.
    hebbian_lr: float = 0.05
    #: SelfNode seed activation.
    self_seed: float = 0.8
    #: Fixed activation base added per query match (rank-scaled). Makes matched
    #: nodes clearly outrank unmatched ones; RRF scores alone are too small.
    match_base: float = 4.0
    #: RRF-style multiplier on the hybrid score when seeding matches.
    match_gain: float = 3.0


def _default_clock() -> float:
    return time.time()


class KnowledgeGraphRetriever:
    """Orchestrates ingestion, activation, retrieval and persistence."""

    def __init__(
        self,
        *,
        graph: Optional[KnowledgeGraph] = None,
        config: Optional[KnowledgeGraphConfig] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.graph = graph or KnowledgeGraph()
        self.config = config or KnowledgeGraphConfig()
        self.llm: Optional[LLMClient] = None
        self.embedder: Optional[EmbeddingProvider] = None
        self.hybrid: Optional[HybridSearch] = None
        self.store: Optional[SQLiteStore] = None
        self._now = clock or _default_clock
        # user_ids the graph currently knows about (drives person resolution
        # and the SelfNode relation edges). Refreshed on every ingest.
        self._known_users: list[str] = []
        # Character identity (name + persona) used to make entity extraction
        # context-aware and relevance-filtered. Set by the agent at build time.
        self.character: Optional[dict[str, str]] = None

    # --------------------------------------------------------------- backends
    def load(
        self,
        llm: Optional[LLMClient],
        embedder: EmbeddingProvider,
        hybrid: HybridSearch,
        *,
        store: Optional[SQLiteStore] = None,
    ) -> "KnowledgeGraphRetriever":
        """Wire the LLM / embedder / hybrid index (and optional SQLite store).

        Mirrors the example API in the design doc::

            kg.load(llm, embedding, HybridSearch(embedding))
        """
        self.llm = llm
        self.embedder = embedder
        self.hybrid = hybrid
        self.store = store
        return self

    # --------------------------------------------------------------- ingestion
    def ingest(self, memories: Iterable[Any]) -> "KnowledgeGraphRetriever":
        """Full ingest from the given source memories.

        Reads each source memory via its public API and (re)builds the graph.
        Idempotent for the same inputs: PersonNode/SelfNode are upserts, and
        auto-increment ids (`fact:<n>`, `episode:<n>`) make collisions across
        re-ingests unlikely (call :meth:`reset` first for a clean rebuild).
        """
        mems = {getattr(m, "name", None): m for m in memories}

        # Summaries first so person resolution works during fact ingestion.
        known_users: list[str] = []
        summary = mems.get("user_summary")
        if isinstance(summary, UserSummaryMemory):
            known_users = ingest_summaries(self.graph, summary)

        emotion = mems.get("emotion")
        if isinstance(emotion, EmotionStatus):
            # Make sure every known user has a node before wiring relations.
            for uid in known_users:
                self.graph.ensure_person(uid)
            ingest_emotion(self.graph, emotion, known_users=known_users)

        facts = mems.get("user_facts")
        if isinstance(facts, UserFactMemory):
            ingest_facts(
                self.graph, facts, llm=self.llm,
                known_users=known_users,
                character=self.character,
            )

        episodic = mems.get("episodic")
        if isinstance(episodic, EpisodicMemory):
            ingest_episodes(self.graph, episodic)

        self._known_users = known_users or [n.user_id for n in self.graph.nodes_of_kind("person")]
        # Keep the node-text hybrid index in sync with whatever we just built.
        self._rebuild_index()
        return self

    def _has_wiki_nodes(self) -> bool:
        """True if any wiki-derived node is already in the graph."""
        return any(
            (n.source or "").startswith("wiki:") for n in self.graph.nodes.values()
        )

    def ingest_wiki(self, sections: Iterable[dict[str, Any]]) -> "KnowledgeGraphRetriever":
        """Wiki ingest into the graph.

        With an LLM wired (``attach_backends``), extracts characters, entities
        and story events per section and builds a typed subgraph
        (``ingest_wiki_llm``). Without one, falls back to flat ``FactNode``
        chunks (``ingest_wiki``). In both cases previously-ingested ``wiki:*``
        fact/episode nodes are dropped first (entity nodes dedupe by name and
        are preserved, so re-running is idempotent), then the node-text index
        is rebuilt.
        """
        for node in list(self.graph.nodes.values()):
            if (node.source or "").startswith("wiki:") and node.kind in ("fact", "episode"):
                self.graph.remove_node(node.id)
        if self.llm is not None:
            ingest_wiki_llm(self.graph, self.llm, sections, character=self.character)
        else:
            ingest_wiki(self.graph, sections)
        self._rebuild_index()
        return self

    def update(self, extracted_items: dict[str, list]) -> "KnowledgeGraphRetriever":
        """Incremental ingest for a freshly-extracted batch.

        `extracted_items` is the `{memory_name: [MemoryItem, ...]}` dict the
        agent carries around after extraction (the `added` map). Only the
        memory families the graph cares about are consumed.
        """
        facts_items = extracted_items.get("user_facts") or []
        episodes_items = extracted_items.get("episodic") or []
        summary_items = extracted_items.get("user_summary") or []
        # Summaries upsert PersonNodes directly.
        if summary_items:
            for it in summary_items:
                meta = it.metadata or {}
                uid = str(meta.get("user_id") or "")
                if uid:
                    aliases = meta.get("aliases")
                    if isinstance(aliases, str):
                        try:
                            import json as _json
                            aliases = _json.loads(aliases)
                        except (ValueError, TypeError):
                            aliases = []
                    name = str(meta.get("name") or uid)
                    p = self.graph.ensure_person(uid, name=name, aliases=aliases or [])
                    p.text = it.text or p.text
                    if uid not in self._known_users:
                        self._known_users.append(uid)
        # Facts: filter to rows the items reference, then ingest.
        facts_mem = self._source_memory_for("user_facts")
        if facts_mem is not None and facts_items:
            ids = [int((it.metadata or {}).get("id")) for it in facts_items if (it.metadata or {}).get("id") is not None]
            if ids:
                rows = [r for r in facts_mem.store.select(facts_mem.table) if int(r.get("id") or -1) in ids]
                ingest_facts(
                    self.graph, facts_mem, llm=self.llm,
                    known_users=self._known_users, rows=rows,
                    character=self.character,
                )
        # Episodes: same pattern.
        ep_mem = self._source_memory_for("episodic")
        if ep_mem is not None and episodes_items:
            ids = [int((it.metadata or {}).get("id")) for it in episodes_items if (it.metadata or {}).get("id") is not None]
            if ids:
                rows = [r for r in ep_mem.store.select(ep_mem.table) if int(r.get("id") or -1) in ids]
                ingest_episodes(self.graph, ep_mem, rows=rows)
        self._rebuild_index()
        return self

    def apply_deduplication(self, report: dict[str, DedupReport]) -> "KnowledgeGraphRetriever":
        """Mirror a per-memory dedup report into graph mutations.

        For each memory's report, ``removed_ids`` drop the matching nodes
        (by `source` tag) and ``updated_ids`` refresh the matching nodes'
        text + carried fields by re-reading the source row. The KG never
        writes back to the source memory.
        """
        for mem_name, rep in report.items():
            for rid in rep.removed_ids or []:
                self._remove_by_source(mem_name, rid)
            for rid in rep.updated_ids or []:
                self._refresh_by_source(mem_name, rid)
        if any(rep.removed_ids or rep.updated_ids for rep in report.values()):
            self._rebuild_index()
        return self

    def _remove_by_source(self, mem_name: str, row_id: Any) -> None:
        tag = f"{mem_name}:{row_id}"
        for node in list(self.graph.nodes.values()):
            if node.source == tag:
                self.graph.remove_node(node.id)

    def _refresh_by_source(self, mem_name: str, row_id: Any) -> None:
        tag = f"{mem_name}:{row_id}"
        src = self._source_memory_for(mem_name)
        if src is None:
            return
        rows = src.store.select(src.table, {"id": row_id})
        if not rows:
            return
        row = rows[0]
        for node in list(self.graph.nodes.values()):
            if node.source != tag or not isinstance(node, Node):
                continue
            # Refresh text + carried fields from the source row.
            try:
                node.text = src.row_text(row)
            except Exception:
                pass
            if mem_name == "user_facts":
                node.content = str(row.get("content") or getattr(node, "content", ""))  # type: ignore[attr-defined]
                node.confidence = float(row.get("confidence") or 0.5)  # type: ignore[attr-defined]
                node.importance = float(row.get("importance") or 0.5)  # type: ignore[attr-defined]
            elif mem_name == "episodic":
                node.summary = str(row.get("summary") or getattr(node, "summary", ""))  # type: ignore[attr-defined]
                node.emotional_shift = float(row.get("emotional_shift") or 0.0)  # type: ignore[attr-defined]
                node.importance = float(row.get("importance") or 0.5)  # type: ignore[attr-defined]
            elif mem_name == "user_summary":
                aliases = row.get("aliases")
                if isinstance(aliases, str):
                    try:
                        import json as _json
                        aliases = _json.loads(aliases)
                    except (ValueError, TypeError):
                        aliases = []
                if isinstance(node, type(self.graph.ensure_person("x"))):
                    pass
                node.name = str(row.get("name") or getattr(node, "name", ""))  # type: ignore[attr-defined]
                node.aliases = list(aliases or [])  # type: ignore[attr-defined]

    def _source_memory_for(self, name: str) -> Optional[Any]:
        """The retriever does not hold source memories; the agent resolves them.

        Returns None here; the agent subclass overrides retrieval-time hooks
        via :meth:`KnowledgeGraphMemory` which holds a back-reference. For
        the standalone API the caller wires `update` / dedup with rows.
        """
        return None

    # ----------------------------------------------------------- retrieval core
    def _seed_activations(self, query: str) -> dict[str, float]:
        """RRF-score the query against the node-text index + seed the SelfNode.

        Each hit contributes two things: a fixed base (so a lexical/semantic
        match always meaningfully lifts a node above the ACT-R decay floor)
        plus the RRF score scaled by ``match_gain`` (so better matches rank
        higher within the matched set). Without the fixed base the tiny RRF
        scores (~0.02-0.05) are drowned by the BLL term and query relevance
        is invisible in the ranking.
        """
        seeds: dict[str, float] = {}
        if self.graph.SELF_ID in self.graph.nodes:
            seeds[self.graph.SELF_ID] = self.config.self_seed
        if self.hybrid is None or not query.strip():
            return seeds
        try:
            hits = self.hybrid.search(query, k=max(10, self.config.hops * 8))
        except Exception:
            hits = []
        # Fixed per-hit base scaled by rank: the top hit gets the full base,
        # trailing hits get less. This makes "matched, ranked" nodes clearly
        # outrank "unmatched but not decayed" ones.
        n_hits = max(1, len(hits))
        for rank, h in enumerate(hits):
            nid = h.metadata.get("id")
            if not (isinstance(nid, str) and nid in self.graph.nodes):
                continue
            rank_base = self.config.match_base * (1.0 - 0.6 * rank / n_hits)
            seeds[nid] = seeds.get(nid, 0.0) + rank_base + float(h.score) * self.config.match_gain
        return seeds

    def test_activation(self, query: str, *, user_id: Optional[str] = None) -> dict[str, float]:
        """Return the full `{node_id: activation}` trace for `query`.

        Read-only: does not bump practice times or run the Hebbian step.
        """
        if not self.graph.nodes:
            return {}
        seeds = self._seed_activations(query)
        # Bias the user's own PersonNode so "about me" wins ties.
        if user_id:
            pid = f"person:{user_id}"
            if pid in self.graph.nodes:
                seeds[pid] = seeds.get(pid, 0.0) + self.config.self_seed * 0.6
        act = combined_activation(
            self.graph, seeds,
            now=self._now(),
            decay=self.config.decay,
            decay_half_life=self.config.decay_half_life,
            gain=self.config.gain,
            hops=self.config.hops,
            base_weight=self.config.base_weight,
            spread_weight=self.config.spread_weight,
        )
        # Stash on the nodes for the GUI / debugging.
        for nid, node in self.graph.nodes.items():
            node.activation = float(act.get(nid, 0.0))
        return act

    def retrieve(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 6,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        """Return the top-`limit` nodes by activation as MemoryItems.

        When `state_changing` is True (the default) the surfaced nodes get a
        practice event appended and the Hebbian step strengthens the
        co-occurrence edges between co-activated nodes. Read-only previews
        pass `state_changing=False`.
        """
        if not self.graph.nodes:
            return []
        act = self.test_activation(query, user_id=user_id)
        ranked = sorted(
            ((a, nid) for nid, a in act.items() if a >= self.config.min_activation),
            reverse=True,
        )
        top = [(a, nid) for a, nid in ranked[: max(0, limit)]]
        if not top:
            return []
        items: list[MemoryItem] = []
        surfaced_ids: list[str] = []
        for a, nid in top:
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            items.append(self._node_to_item(node, a))
            surfaced_ids.append(nid)
        if state_changing:
            now = self._now()
            for nid in surfaced_ids:
                node = self.graph.nodes.get(nid)
                if node is not None:
                    node.touch(now)
            self._hebbian_step(set(surfaced_ids))
        return items

    def _hebbian_step(self, surfaced: set[str]) -> None:
        """Strengthen co-occurrence edges between co-activated nodes.

        Uses the last computed `activation` (populated by
        :meth:`test_activation`) to decide which nodes "fired together".
        """
        threshold = self.config.hebbian_threshold
        lr = self.config.hebbian_lr
        fired = {
            nid for nid in surfaced
            if self.graph.nodes.get(nid) is not None
            and self.graph.nodes[nid].activation >= threshold
        }
        if len(fired) < 2:
            return
        fired_list = list(fired)
        for i, a in enumerate(fired_list):
            for b in fired_list[i + 1 :]:
                edge = self.graph.get_edge_between("co_occurrence", a, b)
                if edge is None or not isinstance(edge, CoOccurrenceEdge):
                    edge = self.graph.add_co_occurrence(a, b, co_create=False, weight=0.05)
                if edge is None:
                    continue
                edge.co_recall_count += 1
                edge.weight = min(1.0, float(edge.weight) + lr)

    # --------------------------------------------------------------- rendering
    @staticmethod
    def _node_to_item(node: Node, activation: float) -> MemoryItem:
        """Render a node into a prompt-friendly MemoryItem."""
        kind = node.kind
        if kind == "self":
            baseline = getattr(node, "baseline", {}) or {}
            text = "Self state: " + ", ".join(f"{k}={v:.2f}" for k, v in baseline.items())
        elif kind == "person":
            text = f"{getattr(node, 'name', node.id)} (user)"
        elif kind == "fact":
            text = str(getattr(node, "content", node.text) or node.text)
        elif kind == "episode":
            text = str(getattr(node, "summary", node.text) or node.text)
        elif kind == "entity":
            text = f"{getattr(node, 'name', node.text)} ({getattr(node, 'kind_label', 'thing')})"
        else:
            text = node.text or node.id
        return MemoryItem(
            text=text,
            score=float(activation),
            kind="knowledge_graph",
            metadata={"node_id": node.id, "node_kind": kind, "activation": float(activation)},
        )

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> "KnowledgeGraphRetriever":
        if self.hybrid is None or self.store is None:
            return self
        save_graph(self.graph, self.store, self.hybrid, path)
        return self

    def load_persisted(self, path: str) -> "KnowledgeGraphRetriever":
        if self.hybrid is None or self.store is None:
            return self
        if not has_persisted(self.store, path):
            return self
        self.graph = load_graph(self.store)
        try:
            self.hybrid.load(path)
        except Exception:
            # The SQLite tables are the source of truth for nodes/edges; a
            # stale/missing FAISS index is rebuilt on the next save().
            pass
        self._known_users = [n.user_id for n in self.graph.nodes_of_kind("person")]
        return self

    def has_persisted(self, path: str) -> bool:
        if self.store is None:
            return False
        return has_persisted(self.store, path)

    def reset(self) -> "KnowledgeGraphRetriever":
        """Wipe the in-memory graph (counters included)."""
        self.graph = KnowledgeGraph()
        self._known_users = []
        return self

    # ----------------------------------------------------------------- helpers
    def _rebuild_index(self) -> None:
        """Rebuild the node-text hybrid index from the current nodes."""
        if self.hybrid is None:
            return
        from ..chunking.base import Chunk

        chunks = []
        for node in self.graph.nodes.values():
            text = (node.text or "").strip()
            if not text or node.id == self.graph.SELF_ID:
                continue
            chunks.append(
                Chunk(text=text, source=node.kind, metadata={"id": node.id, "kind": node.kind})
            )
        try:
            self.hybrid.build(chunks)
        except Exception:
            # Embeddings unreachable: skip silently; rows are durable in SQLite.
            pass

    # ----------------------------------------------------------- introspection
    def overview(self) -> dict[str, Any]:
        """A compact summary used by the GUI sidebar / MCP `graph_overview`."""
        c = self.graph.counts()
        return {
            "nodes": len(self.graph),
            "edges": c.pop("__edges__", 0),
            "by_kind": c,
            "users": list(self._known_users),
        }


# Typo-friendly alias so the example API in the design doc works verbatim:
#     kg = KnowledgeGraphRetrivier()
KnowledgeGraphRetrivier = KnowledgeGraphRetriever

__all__ = [
    "KnowledgeGraphRetriever",
    "KnowledgeGraphRetrivier",
    "KnowledgeGraphConfig",
]
