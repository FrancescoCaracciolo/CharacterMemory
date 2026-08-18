"""The high-level retriever that wires ingestion + activation + retrieval.

This is the class the example API in the design doc drives::

    kg = KnowledgeGraphRetriever()
    kg.load(llm, embedder, HybridSearch(embedder))
    kg.ingest([UserFactMemory, EmotionMemory, ...])
    kg.update(extracted_items)
    kg.apply_deduplication(dedup_report)
    trace = kg.test_activation("message")
    items = kg.retrieve("message", user_id="...", token_budget=1000)
    kg.save(save_dir)

It owns a :class:`KnowledgeGraph`, a node-text :class:`HybridSearch`, and an
optional :class:`SQLiteStore` + :class:`LLMClient`. Every public method is a
thin orchestrator over :mod:`ingest`, :mod:`activation`, and
:mod:`persistence`; the heavy lifting lives there.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

from ..config import KnowledgeGraphConfig
from ..llm.base import LLMClient
from ..llm.embedding_base import EmbeddingProvider
from ..emotion_vectors import (
    decode_emotion_vector,
    emotion_similarity,
    emotional_impact,
)
from ..memory.base import MemoryItem, item_bullet
from ..memory.dedup import DedupReport
from ..memory.emotion import EmotionStatus
from ..memory.episodic import EpisodicMemory
from ..memory.store import SQLiteStore
from ..memory.user_facts import UserFactMemory
from ..memory.user_summary import UserSummaryMemory
from ..rag.base import as_queries
from ..rag.hybrid import HybridSearch
from .activation import combined_activation, combined_activation_breakdown
from .edges import CoOccurrenceEdge, EpisodeEdge
from .graph import KnowledgeGraph
from .ingest import (
    _EXTRACTION_TOKEN_LIMIT,
    _FACT_BATCH_SIZE,
    _EPISODE_BATCH_SIZE,
    _WIKI_BATCH_SIZE,
    _batch_by_limits,
    _count_tokens,
    ingest_emotion,
    ingest_episodes,
    ingest_facts,
    ingest_summaries,
    ingest_wiki,
    ingest_wiki_llm,
    wire_chat_edges,
)
from .nodes import Node, PersonNode
from .persistence import (
    graph_index_chunks,
    has_persisted,
    load_graph,
    save_graph,
    sync_hybrid_index,
)


_NodeAddedCallback = Callable[[Node], None]
_LLMProgressCallback = Callable[[int, int], None]


class _BuildProgress:
    """Transient callback state shared by both bulk-ingestion phases."""

    def __init__(
        self,
        total: int,
        *,
        on_node_added: Optional[_NodeAddedCallback],
        on_llm_progress: Optional[_LLMProgressCallback],
    ) -> None:
        self.completed = 0
        self.total = total
        self.on_node_added = on_node_added
        self.on_llm_progress = on_llm_progress

    def start(self) -> None:
        if self.on_llm_progress is not None:
            self.on_llm_progress(0, self.total)

    def request_done(self) -> None:
        self.completed += 1
        if self.on_llm_progress is not None:
            self.on_llm_progress(self.completed, self.total)


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
        # A brand-new/rebuilt retriever owns an authoritative full snapshot;
        # once loaded or saved, routine extraction persists as a merge so a
        # stale worker cannot erase wiki rows written by another process.
        self._replace_on_next_save = True
        self._now = clock or _default_clock
        # user_ids the graph currently knows about (drives person resolution
        # and the SelfNode relation edges). Refreshed on every ingest.
        self._known_users: list[str] = []
        # Character identity (name + persona + aliases) used to make entity
        # extraction context-aware and relevance-filtered, and to drive the
        # self-dedup pass. Set by the agent at build time.
        self.character: Optional[dict[str, Any]] = None

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
    def ingest(
        self,
        memories: Iterable[Any],
        *,
        _on_llm_request_done: Optional[Callable[[], None]] = None,
        _sync_index: bool = True,
    ) -> "KnowledgeGraphRetriever":
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
            known_users = list(dict.fromkeys([
                *known_users,
                *[
                    str(r.get("user_id") or "")
                    for r in emotion.store.select(emotion.table)
                    if r.get("user_id")
                ],
            ]))
            # Make sure every known user has a node before wiring relations.
            for uid in known_users:
                self.graph.ensure_person(uid)
            ingest_emotion(self.graph, emotion, known_users=known_users, character=self.character)

        facts = mems.get("user_facts")
        if isinstance(facts, UserFactMemory):
            ingest_facts(
                self.graph, facts, llm=self.llm,
                known_users=known_users,
                character=self.character,
                batch_size=getattr(self.config, "fact_batch_size", _FACT_BATCH_SIZE),
                token_limit=getattr(
                    self.config, "extraction_token_limit", _EXTRACTION_TOKEN_LIMIT
                ),
                _on_llm_request_done=_on_llm_request_done,
            )

        episodic = mems.get("episodic")
        if isinstance(episodic, EpisodicMemory):
            ingest_episodes(
                self.graph,
                episodic,
                batch_size=getattr(
                    self.config, "episode_batch_size", _EPISODE_BATCH_SIZE
                ),
                token_limit=getattr(
                    self.config, "extraction_token_limit", _EXTRACTION_TOKEN_LIMIT
                ),
            )

        self._known_users = known_users or [n.user_id for n in self.graph.nodes_of_kind("person")]
        # Deterministic self-healing: collapse any person node that is actually
        # the character into the SelfNode, then fold duplicate persons sharing
        # a name/alias. Runs after every ingest so an extraction slip never
        # leaves a duplicate node behind. No LLM cost.
        self._dedup_persons()
        # Link facts and episodes learned in the same chat (low-weight bridges).
        wire_chat_edges(self.graph)
        if _sync_index:
            self._sync_index()
        return self

    def _has_wiki_nodes(self) -> bool:
        """True if any wiki-derived node is already in the graph."""
        return any(
            (n.source or "").startswith("wiki:") for n in self.graph.nodes.values()
        )

    def ingest_wiki(
        self,
        sections: Iterable[dict[str, Any]],
        *,
        _on_llm_request_done: Optional[Callable[[], None]] = None,
        _sync_index: bool = True,
    ) -> "KnowledgeGraphRetriever":
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
            ingest_wiki_llm(
                self.graph,
                self.llm,
                sections,
                batch_size=getattr(self.config, "wiki_batch_size", _WIKI_BATCH_SIZE),
                token_limit=getattr(
                    self.config, "extraction_token_limit", _EXTRACTION_TOKEN_LIMIT
                ),
                character=self.character,
                _on_llm_request_done=_on_llm_request_done,
            )
        else:
            ingest_wiki(self.graph, sections)
        self._dedup_persons()
        if _sync_index:
            self._sync_index()
        return self

    def _ingest_for_build(
        self,
        memories: Iterable[Any],
        wiki_sections: Iterable[dict[str, Any]],
        *,
        on_node_added: Optional[_NodeAddedCallback] = None,
        on_llm_progress: Optional[_LLMProgressCallback] = None,
    ) -> "KnowledgeGraphRetriever":
        """Run both bulk-ingestion phases with one exact progress counter."""
        memory_list = list(memories)
        section_list = list(wiki_sections)
        total = self._build_llm_request_count(memory_list, section_list)
        progress = _BuildProgress(
            total,
            on_node_added=on_node_added,
            on_llm_progress=on_llm_progress,
        )
        progress.start()
        with self.graph._observe_node_additions(progress.on_node_added):
            self.ingest(
                memory_list,
                _on_llm_request_done=progress.request_done,
                _sync_index=False,
            )
            self.ingest_wiki(
                section_list,
                _on_llm_request_done=progress.request_done,
                _sync_index=False,
            )
        # A combined full ingest used to embed once after memories and again
        # after wiki. Build the authoritative final snapshot exactly once.
        self._rebuild_index()
        return self

    def _build_llm_request_count(
        self,
        memories: list[Any],
        wiki_sections: list[dict[str, Any]],
    ) -> int:
        """Return the structured LLM calls the matching bulk ingest will make."""
        if self.llm is None:
            return 0
        mems = {getattr(memory, "name", None): memory for memory in memories}
        facts = mems.get("user_facts")
        fact_rows = (
            facts.store.select(facts.table)
            if isinstance(facts, UserFactMemory)
            else []
        )
        token_limit = getattr(
            self.config, "extraction_token_limit", _EXTRACTION_TOKEN_LIMIT
        )
        fact_requests = len(
            _batch_by_limits(
                fact_rows,
                max_items=getattr(
                    self.config, "fact_batch_size", _FACT_BATCH_SIZE
                ),
                max_tokens=token_limit,
                text_of=lambda row: str(
                    row.get("content") or row.get("text") or ""
                ),
            )
        )
        wiki_requests = len(
            _batch_by_limits(
                list(enumerate(wiki_sections)),
                max_items=getattr(
                    self.config, "wiki_batch_size", _WIKI_BATCH_SIZE
                ),
                max_tokens=token_limit,
                text_of=lambda item: str(item[1].get("text") or ""),
            )
        )
        return fact_requests + wiki_requests

    def update(
        self,
        extracted_items: dict[str, list],
        *,
        sync_index: bool = True,
    ) -> "KnowledgeGraphRetriever":
        """Incremental ingest for a freshly-extracted batch.

        `extracted_items` is the `{memory_name: [MemoryItem, ...]}` dict the
        agent carries around after extraction (the `added` map). Only the
        memory families the graph cares about are consumed.
        """
        facts_items = extracted_items.get("user_facts") or []
        episodes_items = extracted_items.get("episodic") or []
        summary_items = extracted_items.get("user_summary") or []
        emotion_items = extracted_items.get("emotion") or []
        if not any((facts_items, episodes_items, summary_items, emotion_items)):
            return self
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
        # Emotion is mutable state rather than append-only rows. A change
        # marker tells us to re-read the authoritative current mood and
        # relationship blobs, then upsert Self/Relation graph data.
        emotion_mem = self._source_memory_for("emotion")
        if emotion_items and isinstance(emotion_mem, EmotionStatus):
            for row in emotion_mem.store.select(emotion_mem.table):
                uid = str(row.get("user_id") or "")
                if uid and uid not in self._known_users:
                    self._known_users.append(uid)
            ingest_emotion(
                self.graph,
                emotion_mem,
                known_users=self._known_users,
                character=self.character,
            )
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
                    batch_size=getattr(
                        self.config, "fact_batch_size", _FACT_BATCH_SIZE
                    ),
                    token_limit=getattr(
                        self.config,
                        "extraction_token_limit",
                        _EXTRACTION_TOKEN_LIMIT,
                    ),
                )
        # Episodes: same pattern.
        ep_mem = self._source_memory_for("episodic")
        if ep_mem is not None and episodes_items:
            ids = [int((it.metadata or {}).get("id")) for it in episodes_items if (it.metadata or {}).get("id") is not None]
            if ids:
                rows = [r for r in ep_mem.store.select(ep_mem.table) if int(r.get("id") or -1) in ids]
                ingest_episodes(
                    self.graph,
                    ep_mem,
                    rows=rows,
                    batch_size=getattr(
                        self.config, "episode_batch_size", _EPISODE_BATCH_SIZE
                    ),
                    token_limit=getattr(
                        self.config,
                        "extraction_token_limit",
                        _EXTRACTION_TOKEN_LIMIT,
                    ),
                )
        self._dedup_persons()
        # Re-link same-chat facts/episodes across the whole graph: a freshly
        # ingested fact should bridge to pre-existing episodes of that chat.
        wire_chat_edges(self.graph)
        if sync_index:
            self._sync_index()
        return self

    def apply_deduplication(
        self,
        report: dict[str, DedupReport],
        *,
        sync_index: bool = True,
    ) -> "KnowledgeGraphRetriever":
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
        if sync_index and any(
            rep.removed_ids or rep.updated_ids for rep in report.values()
        ):
            self._sync_index()
        return self

    def deduplicate_persons(self) -> dict[str, list]:
        """Run the deterministic person-dedup passes and return a report.

        Public entry point (used by the ``deduplicate_knowledge_graph`` MCP
        tool) for a one-time cleanup of an already-built graph: collapses
        any PersonNode that is actually the character into the SelfNode,
        then folds duplicate PersonNodes sharing a name/alias. The hybrid
        index is synchronized incrementally; the caller persists it.
        """
        into_self = self.graph.collapse_into_self(self._character_labels())
        merged = self.graph.merge_duplicate_persons()
        if into_self or merged:
            self._sync_index()
        return {"collapsed_into_self": into_self, "merged_persons": merged}

    def _dedup_persons(self) -> None:
        """Internal post-ingest hook: run the dedup passes (no index rebuild)."""
        self.graph.collapse_into_self(self._character_labels())
        self.graph.merge_duplicate_persons()

    def _character_labels(self) -> list[str]:
        """The character's name + aliases from the wired identity, for dedup."""
        if not self.character:
            return []
        labels = [self.character.get("name") or ""]
        labels.extend(self.character.get("aliases") or [])
        return [str(c).strip() for c in labels if str(c or "").strip()]

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
                node.emotional_shift = decode_emotion_vector(  # type: ignore[attr-defined]
                    row.get("emotional_shift", "{}"),
                    allowed_axes=getattr(src, "emotion_baseline", None),
                )
                node.importance = float(row.get("importance") or 0.5)  # type: ignore[attr-defined]
                for edge in self.graph.edges.values():
                    if isinstance(edge, EpisodeEdge) and edge.dst == node.id:
                        edge.emotional_shift = dict(node.emotional_shift)  # type: ignore[attr-defined]
            elif mem_name == "user_summary":
                aliases = row.get("aliases")
                if isinstance(aliases, str):
                    try:
                        import json as _json
                        aliases = _json.loads(aliases)
                    except (ValueError, TypeError):
                        aliases = []
                if isinstance(node, PersonNode):
                    node.name = str(row.get("name") or node.name)
                    node.aliases = list(aliases or [])
                    node.text = self.graph._person_text(node.name, node.aliases)

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

        Each hit contributes two things: a relevance-scaled base (so a strong
        lexical/semantic match meaningfully lifts a node above the ACT-R decay
        floor without amplifying marginal candidates) plus the RRF score
        scaled by ``match_gain``. Without the base the tiny RRF scores
        (~0.02-0.05) are drowned by the BLL term.
        """
        seeds: dict[str, float] = {}
        if self.graph.SELF_ID in self.graph.nodes:
            seeds[self.graph.SELF_ID] = self.config.self_seed
        # `query` may be a bare string or a list of (text, weight) pairs
        # (history-aware retrieval). Skip only when there is no usable text at
        # all; hybrid.search already fuses the weighted list itself.
        if self.hybrid is None or not as_queries(query):
            return seeds
        try:
            hits = self.hybrid.search(query, k=max(10, self.config.hops * 8))
        except Exception:
            hits = []
        # Per-hit base scaled by both rank and qualified retrieval relevance.
        n_hits = max(1, len(hits))
        for rank, h in enumerate(hits):
            nid = h.metadata.get("id")
            if not (isinstance(nid, str) and nid in self.graph.nodes):
                continue
            relevance = max(
                0.0,
                min(1.0, float(h.metadata.get("normalized_relevance", 1.0))),
            )
            if relevance <= 0.0:
                continue
            rank_base = self.config.match_base * (1.0 - 0.6 * rank / n_hits)
            seeds[nid] = seeds.get(nid, 0.0) + (
                rank_base * relevance + float(h.score) * self.config.match_gain
            )
        return seeds

    def test_activation(self, query: str, *, user_id: Optional[str] = None) -> dict[str, float]:
        """Return the full `{node_id: activation}` trace for `query`.

        Read-only: does not bump practice times or run the Hebbian step. Also
        stashes a per-node ``score_breakdown`` (BLL / spread / seed / emotion
        multiplier / final) on every node for the GUI and for
        :meth:`test_activation_details`.
        """
        if not self.graph.nodes:
            return {}
        seeds = self._seed_activations(query)
        # Bias the user's own PersonNode so "about me" wins ties.
        if user_id:
            person = self.graph.find_person_by_user_id(user_id)
            pid = person.id if person is not None else f"person:{user_id}"
            if pid in self.graph.nodes:
                seeds[pid] = seeds.get(pid, 0.0) + self.config.self_seed * 0.6
        breakdown = combined_activation_breakdown(
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
            comp = breakdown.get(nid, {})
            node.activation = float(comp.get("score", 0.0))
            node.score_breakdown = dict(comp)
        return {nid: comp["score"] for nid, comp in breakdown.items()}

    def test_activation_details(
        self, query: str, *, user_id: Optional[str] = None
    ) -> dict[str, dict[str, float]]:
        """Read-only activation trace with episode emotion components."""
        activation = self.test_activation(query, user_id=user_id)
        self_node = self.graph.nodes.get(self.graph.SELF_ID)
        current_mood = getattr(self_node, "current_mood", {}) or {}
        details: dict[str, dict[str, float]] = {}
        for nid, score in activation.items():
            node = self.graph.nodes[nid]
            shift = getattr(node, "emotional_shift", None)
            impact = emotional_impact(shift) if isinstance(shift, dict) else 0.0
            similarity = (
                emotion_similarity(shift, current_mood)
                if isinstance(shift, dict)
                else 0.0
            )
            details[nid] = {
                "activation": float(score),
                "raw_emotional_impact": impact,
                "emotion_similarity": similarity,
                "impact": impact,
                "similarity": similarity,
            }
            # Per-factor decomposition of the activation score, stashed by
            # test_activation. Forwarded verbatim so the GUI/MCP can show how
            # much each factor (BLL, spreading, seed, emotion) contributed.
            comp = getattr(node, "score_breakdown", None)
            if isinstance(comp, dict) and comp:
                details[nid]["score_breakdown"] = dict(comp)
        return details

    def retrieve(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        token_budget: int = 1_000,
        state_changing: bool = True,
        timestamp_style: str = "both",
    ) -> list[MemoryItem]:
        """Return activation-ranked nodes that fit within ``token_budget``.

        The budget covers the exact bullet-list body injected into the prompt
        (``- node text`` lines, including any timestamp stamps), not an
        arbitrary number of nodes. Nodes are considered in activation order
        and retrieval stops before the first node that would make the rendered
        body exceed the budget. A non-positive budget returns no items.
        `timestamp_style` must match the one the caller will render with so
        the measurement and the final body agree.

        When `state_changing` is True (the default) surfaced nodes update
        exposure telemetry and receive the bounded familiarity benefit; the
        Hebbian step also strengthens co-occurrence edges up to their cap.
        Read-only previews pass `state_changing=False`.
        """
        if not self.graph.nodes:
            return []
        budget = max(0, int(token_budget))
        if budget == 0:
            return []
        act = self.test_activation(query, user_id=user_id)
        ranked = sorted(
            ((a, nid) for nid, a in act.items() if a >= self.config.min_activation),
            reverse=True,
        )
        items: list[MemoryItem] = []
        surfaced_ids: list[str] = []
        for a, nid in ranked:
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            item = self._node_to_item(node, a)
            if _count_tokens(
                self.format_items([*items, item], timestamp_style=timestamp_style)
            ) > budget:
                break
            items.append(item)
            surfaced_ids.append(nid)
        if not items:
            return []
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
                edge.weight = min(
                    float(edge.creation_weight or 0.0) + 0.10,
                    float(edge.weight) + lr,
                )

    # --------------------------------------------------------------- rendering
    @staticmethod
    def format_items(items: list[MemoryItem], timestamp_style: str = "both") -> str:
        """Render recalled nodes exactly as they appear in the KG section."""
        order = {"person": 0, "fact": 1, "episode": 2, "entity": 3, "self": 4}
        ranked = sorted(
            items,
            key=lambda item: (
                order.get(item.metadata.get("node_kind"), 9),
                -item.score,
            ),
        )
        return "\n".join(item_bullet(item, timestamp_style) for item in ranked)

    def _node_to_item(self, node: Node, activation: float) -> MemoryItem:
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
        metadata: dict[str, Any] = {
            "node_id": node.id,
            "node_kind": kind,
            "activation": float(activation),
        }
        # Surfaced so prompt rendering can stamp the node's age (see
        # item_bullet); None on nodes never assigned a creation time.
        created_at = getattr(node, "created_at", None)
        if created_at is not None:
            metadata["created_at"] = float(created_at)
        shift = getattr(node, "emotional_shift", None)
        if isinstance(shift, dict):
            self_node = self.graph.nodes.get(self.graph.SELF_ID)
            current_mood = getattr(self_node, "current_mood", {}) or {}
            impact = emotional_impact(shift)
            similarity = emotion_similarity(shift, current_mood)
            metadata.update(
                {
                    "emotional_shift": dict(shift),
                    "raw_emotional_impact": impact,
                    "emotion_similarity": similarity,
                    "impact": impact,
                    "similarity": similarity,
                }
            )
        return MemoryItem(
            text=text,
            score=float(activation),
            kind="knowledge_graph",
            metadata=metadata,
        )

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> "KnowledgeGraphRetriever":
        if self.hybrid is None or self.store is None:
            return self
        self.graph = save_graph(
            self.graph,
            self.store,
            self.hybrid,
            path,
            replace=self._replace_on_next_save,
        )
        self._replace_on_next_save = False
        self._known_users = self._graph_user_ids()
        return self

    def load_persisted(self, path: str) -> "KnowledgeGraphRetriever":
        if self.hybrid is None or self.store is None:
            return self
        if not has_persisted(self.store, path):
            return self
        self.graph = load_graph(self.store)
        self._replace_on_next_save = False
        try:
            self.hybrid.load(path)
        except Exception:
            # The SQLite tables are the source of truth for nodes/edges; a
            # stale/missing FAISS index is rebuilt on the next save().
            pass
        self._known_users = self._graph_user_ids()
        return self

    def has_persisted(self, path: str) -> bool:
        if self.store is None:
            return False
        return has_persisted(self.store, path)

    def reset(self) -> "KnowledgeGraphRetriever":
        """Wipe the in-memory graph (counters included)."""
        self.graph = KnowledgeGraph()
        self._known_users = []
        self._replace_on_next_save = True
        return self

    # ----------------------------------------------------------------- helpers
    def _graph_user_ids(self) -> list[str]:
        """All primary/linked person identifiers, in stable graph order."""
        ids: list[str] = []
        for node in self.graph.nodes_of_kind("person"):
            ids.extend(getattr(node, "user_ids", []) or [])
            ids.append(getattr(node, "user_id", "") or "")
        return list(dict.fromkeys(user_id for user_id in ids if user_id))

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

    def _sync_index(self) -> None:
        """Incrementally align node-text search with the in-memory graph."""
        if self.hybrid is None:
            return
        try:
            sync_hybrid_index(self.hybrid, graph_index_chunks(self.graph))
        except Exception:
            # Embeddings unreachable: graph rows remain durable and a later
            # persistence/load repair can restore the index.
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
