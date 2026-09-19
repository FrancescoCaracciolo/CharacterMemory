"""A character bundles an identity with the memory systems it can recall from and write to."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence
from ._timing import time_memory
from character_memory.llm.base import LLMClient
from character_memory.prompts import INTERMEDIATE_PROMPT_PREFIX, PromptConfig
from .memory.extract import Extractor, ExtractionContext, build_extraction
from .memory.base import Memory, MemoryItem, RecallResult
from .memory.structured import StructuredMemory
from .rag.base import Query
from .temporal import TemporalResolution
from .reranking import (
    Budget, UNSET, MemoryCandidate, MemoryReranker, ScoreMemoryReranker,
    TokenCounter, count_tokens, validate_budget,
)


@dataclass
class MemoryRecallSnapshot:
    """One memory's exact contribution to a built context."""

    name: str
    title: str
    scope: str
    body: str
    section: str
    items: list[MemoryItem] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """A context plus the recall details used to assemble it."""

    sections: dict[str, str]
    recalls: dict[str, MemoryRecallSnapshot]
    query: Query
    user_id: str
    participants: list[str]
    temporal_resolution: Optional[TemporalResolution] = None
    memory_token_count: int = 0
    memory_budget: Optional[int] = None
    memory_types: Optional[list[str]] = None


class Character:

    def __init__(
        self, character_name: str, base_instruction: str = "",
        memories: Optional[list[Memory]] = None,
        llm: Optional[LLMClient] = None, prompts: Optional[PromptConfig] = None,
        *, budget: Optional[int] = None,
        reranker: Optional[MemoryReranker] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        validate_budget(budget)
        self.budget = budget
        self.reranker = reranker
        self.token_counter = token_counter if token_counter is not None else count_tokens
        self.character_name = character_name
        self.base_instruction = base_instruction
        self.memories = memories if memories is not None else []
        self.llm = llm
        self.prompts = prompts
        self._by_name = {m.name: m for m in self.memories}

    def recall(
        self, query: Query, user_id: str = "default", *,
        limits: Optional[dict[str, int]] = None,
        participants: Optional[list[str]] = None,
        budget: Budget = UNSET, reranker: Optional[MemoryReranker] = None,
        temporal_resolution: Optional[TemporalResolution] = None,
        temporal_weight: float = 1.0,
        memory_types: Optional[Sequence[str]] = None,
    ) -> ContextSnapshot:
        """Select memories and return their items, sections and token usage.

        Omitted budget inherits the character default; None is unlimited.
        Unlike the older build_context API, omitted limits use MemoryConfig
        defaults so direct Character callers get a useful candidate pool.
        """
        if limits is None:
            from .config import MemoryConfig

            defaults = MemoryConfig()
            limits = {m.name: defaults.retrieval_limit_for(m.name) for m in self.memories}
        return self.build_context_snapshot(
            query, user_id, limits=limits, participants=participants,
            temporal_resolution=temporal_resolution, temporal_weight=temporal_weight,
            budget=budget, reranker=reranker, memory_types=memory_types,
        )

    def _count_memory_tokens(self, text: str) -> int:
        value = self.token_counter(text)
        if type(value) is not int or value < 0 or (not text and value != 0):
            raise ValueError("token_counter must return nonnegative integers (zero for empty text)")
        return value

    def _select_memories(
        self, query: Query, recalls: dict[str, MemoryRecallSnapshot], *,
        participants: list[str], template: str, budget: Optional[int],
        reranker: MemoryReranker,
    ) -> tuple[dict[str, MemoryRecallSnapshot], int]:
        """Rank once, then fit whole items against exact grouped rendering."""
        candidates: list[MemoryCandidate] = []
        for name, recall in recalls.items():
            mem = self._by_name[name]
            for item in recall.items or [None]:
                body = mem.format_selection([item], participants) if item is not None else recall.body
                text = template.format(title=recall.title, body=body)
                candidates.append(MemoryCandidate(
                    id=len(candidates), memory_name=name, item=item,
                    rendered_text=text, token_count=self._count_memory_tokens(text),
                ))
        ranked = list(reranker.rerank(query, tuple(candidates), budget=budget))
        supplied = {c.id: c for c in candidates}
        seen: set[int] = set()
        for candidate in ranked:
            if (not isinstance(candidate, MemoryCandidate)
                    or type(candidate.id) is not int
                    or supplied.get(candidate.id) is not candidate):
                raise ValueError("reranker returned an unknown or replaced candidate")
            if candidate.id in seen:
                raise ValueError("reranker returned a duplicate candidate")
            seen.add(candidate.id)

        def render(selected: set[int]) -> dict[str, MemoryRecallSnapshot]:
            grouped: dict[str, list[MemoryItem]] = {}
            for candidate in candidates:
                if candidate.id in selected:
                    items = grouped.setdefault(candidate.memory_name, [])
                    if candidate.item is not None:
                        items.append(candidate.item)
            result = {}
            for name, items in grouped.items():
                original = recalls[name]
                body = (self._by_name[name].format_selection(items, participants)
                        if original.items else original.body)
                if body:
                    result[name] = replace(
                        original, items=items, body=body,
                        section=template.format(title=original.title, body=body),
                    )
            return result

        selected: set[int] = set()
        result: dict[str, MemoryRecallSnapshot] = {}
        used = 0
        for candidate in ranked:
            trial = selected | {candidate.id}
            rendered = render(trial)
            tokens = self._count_memory_tokens("\n\n".join(r.section for r in rendered.values()))
            if budget is None or tokens <= budget:
                selected, result, used = trial, rendered, tokens
        return result, used

    def add_memory(self, memory: Memory):
        self.memories.append(memory)
        self._by_name[memory.name] = memory

    def _header_for(self, name: str) -> str:
        if self.prompts is not None:
            header = getattr(self.prompts, f"{name}_header", None)
            if header:
                return header
        return self._by_name[name].title

    def _known_user_facts(
        self, user_id: str, limit: int = 8, participants: Optional[list[str]] = None
    ) -> list[str]:
        """Top known user-fact texts, highest importance first.

        Used to ground the extractor so it does not re-extract what is already
        stored. Returns [] when user_facts is absent or empty. In a group chat
        the facts of every participant are gathered (each capped at `limit`)
        so the model avoids re-extracting for any of them.
        """
        mem = self._by_name.get("user_facts")
        if not isinstance(mem, StructuredMemory):
            return []
        users = participants if participants else [user_id]
        facts: list[str] = []
        for uid in users:
            rows = mem.all_rows(user_id=uid)
            rows.sort(key=lambda r: float(r.get("importance", 0.0)), reverse=True)
            for r in rows[:limit]:
                text = mem.row_text(r).strip()
                if text:
                    facts.append(text)
        return facts

    def _extraction_context(
        self, user_id: str, participants: Optional[list[str]] = None
    ) -> ExtractionContext:
        """Build the identity + grounding context for an extraction call."""
        known = self._known_user_facts(user_id, participants=participants)
        p = self.prompts
        return ExtractionContext(
            character_name=self.character_name,
            user_name=user_id,
            persona=self.base_instruction.strip(),
            known_facts=known,
            participants=list(participants) if participants else [],
            header=p.extraction_header if p is not None else ExtractionContext.header,
            sentence_rule=p.extraction_sentence_rule if p is not None else ExtractionContext.sentence_rule,
            known_facts_intro=p.extraction_known_facts_intro if p is not None else ExtractionContext.known_facts_intro,
            footer=p.extraction_footer if p is not None else ExtractionContext.footer,
            multi_note=p.extraction_multi_note if p is not None else ExtractionContext.multi_note,
        )

    def extract(
        self,
        turns: list[dict[str, Any]],
        user_id: str = "default",
        llm: Optional[LLMClient] = None,
        participants: Optional[list[str]] = None,
        chat_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Run extraction over `turns`.

        ``user_id`` is the chat's default user (owner / current speaker).
        ``participants`` lists the human speakers of a group chat; when it has
        more than one entry extraction runs in multi-user mode (per-speaker
        transcript labelling + a ``user_id`` enum on per-user fields, each item
        attributed to the participant it is about). With one or no participant
        the legacy single-user path runs unchanged.

        ``chat_id`` (optional) identifies the conversation extraction ran over;
        chat-scoped memories (``user_facts``, ``episodic``) stamp it onto their
        rows so the knowledge graph can link facts and episodes of the same
        chat. ``None`` ⇒ legacy / single-user behaviour.
        """
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No LLM specified. Character.extract needs a LLM configured")
        parts = participants if participants else None
        context = self._extraction_context(user_id, participants=parts)
        # Only enabled memories that opt into extraction.
        participating = [
            (mem, spec)
            for mem in self.memories
            if mem.enabled
            for spec in [mem.extraction_spec(context)]
            if spec is not None
        ]
        if not participating:
            return None
        schema, instruction = build_extraction([spec for _, spec in participating], context=context)
        extracted = Extractor(llm).extract(turns, schema=schema, instruction=instruction, context=context)
        added: dict[str, list[MemoryItem]] = {}
        for mem, spec in participating:
            from .concurrency import object_lock
            with object_lock(mem):
                items = mem.apply_extraction(extracted.get(spec.field), user_id, chat_id=chat_id)
            if items:
                added[mem.name] = items
        # Carry the freshly-added items so the caller (e.g. a deduplicator)
        # can act on them without re-querying the memories.
        extracted["__added__"] = added
        return extracted

    def build_context_snapshot(
        self,
        query: Query,
        user_id: str = "default",
        limits: dict[str, int] = {},
        participants: Optional[list[str]] = None,
        temporal_resolution: Optional[TemporalResolution] = None,
        temporal_weight: float = 1.0,
        *,
        budget: Budget = UNSET,
        reranker: Optional[MemoryReranker] = None,
        memory_types: Optional[Sequence[str]] = None,
    ) -> ContextSnapshot:
        """Build the context and retain the exact items recalled by each memory.

        ``query`` may be a plain string or a list of ``(text, weight)`` pairs
        (one per recent chat message, with older ones weighted less); it is
        forwarded to each memory's recall and never interpolated into prompt
        text. ``limits`` contains top-k counts for regular memories and a token
        budget for the knowledge graph. When ``participants`` has more than
        one entry each memory is rendered through its participants-aware path
        (PER_USER memories recall
        + group per speaker; CHARACTER memories recall once). A single
        participant (or none) uses the legacy single-user rendering unchanged.

        ``budget`` caps rendered memory sections, including headers and
        separators, but excludes system instructions and intermediate prompts.
        Omitted budget inherits the constructor default; None is unlimited.
        Rerankers order/filter candidates; only selected items are reinforced.
        ``memory_types`` restricts recall to only the specified memory names,
        bypassing recall for omitted memories to save latency.
        """
        budget = self.budget if budget is UNSET else budget
        validate_budget(budget)
        reranker = self.reranker if reranker is None else reranker
        selective = budget is not None or reranker is not None
        # Fail before lifecycle/retrieval side effects for a broken tokenizer.
        self._count_memory_tokens("")
        order = self.prompts.section_order if self.prompts is not None else [m.name for m in self.memories]
        template = self.prompts.section_template if self.prompts is not None else "## {title}\n{body}"
        intermediate_prompts = (
            self.prompts.intermediate_prompts if self.prompts is not None else {}
        )
        multi = bool(participants and len(participants) > 1)
        sections: dict[str, str] = {}
        recalls: dict[str, MemoryRecallSnapshot] = {}
        allowed_memories = set(memory_types) if memory_types is not None else None
        allowed_memories_lower = (
            {m.lower() for m in allowed_memories if isinstance(m, str)}
            if allowed_memories is not None
            else None
        )
        for name in order:
            if name.startswith(INTERMEDIATE_PROMPT_PREFIX):
                prompt = intermediate_prompts.get(name, "")
                if prompt.strip():
                    sections[name] = prompt
                continue
            mem = self._by_name.get(name)
            if mem is None or not mem.enabled:
                continue
            if allowed_memories is not None:
                if name not in allowed_memories and name.lower() not in allowed_memories_lower:
                    continue
            if selective:
                mem.prepare_recall()
            if budget == 0:
                continue
            recall_kwargs = {"state_changing": False} if selective else {}
            with time_memory(self.character_name, name):
                if multi:
                    result: RecallResult = mem.build_section_participants_result(
                        query,
                        participants,
                        limits.get(name, 0),
                        **recall_kwargs,
                        temporal_resolution=temporal_resolution,
                        temporal_weight=temporal_weight,
                    )
                else:
                    result = mem.build_section_result(
                        query,
                        user_id,
                        limits.get(name, 0),
                        **recall_kwargs,
                        temporal_resolution=temporal_resolution,
                        temporal_weight=temporal_weight,
                    )
            if not result.body:
                continue
            title = self._header_for_multi(name) if multi else self._header_for(name)
            section = template.format(title=title, body=result.body)
            sections[name] = section
            recalls[name] = MemoryRecallSnapshot(
                name=name,
                title=title,
                scope=getattr(mem.scope, "value", str(mem.scope)),
                body=result.body,
                section=section,
                items=list(result.items),
                diagnostics=dict(result.diagnostics or {}),
            )
        # Heartbeat/world remain first-class prompt memories. If their exact
        # source row was also activated through the graph, keep the graph's
        # traversal/diagnostics but suppress duplicate prompt text.
        kg_recall = recalls.get("knowledge_graph")
        kg_memory = self._by_name.get("knowledge_graph")
        if kg_recall is not None and kg_memory is not None:
            direct_sources: set[str] = set()
            heartbeat_recall = recalls.get("heartbeat")
            if heartbeat_recall is not None:
                direct_sources.update(
                    f"heartbeat:{item.metadata['id']}"
                    for item in heartbeat_recall.items
                    if item.metadata.get("id") is not None
                )
            world_recall = recalls.get("world")
            if world_recall is not None:
                direct_sources.update(
                    f"world_records:{item.metadata['id']}"
                    for item in world_recall.items
                    if not item.metadata.get("world_current")
                    and item.metadata.get("id") is not None
                )
            filtered = [
                item
                for item in kg_recall.items
                if item.metadata.get("source") not in direct_sources
            ]
            if len(filtered) != len(kg_recall.items):
                if filtered:
                    body = kg_memory.format(filtered)
                    section = template.format(title=kg_recall.title, body=body)
                    kg_recall.items = filtered
                    kg_recall.body = body
                    kg_recall.section = section
                    sections["knowledge_graph"] = section
                else:
                    sections.pop("knowledge_graph", None)
                    recalls.pop("knowledge_graph", None)
        if selective:
            recalls, memory_tokens = self._select_memories(
                query, recalls, participants=list(participants or [user_id]),
                template=template, budget=budget,
                reranker=reranker if reranker is not None else ScoreMemoryReranker(),
            )
            sections = {
                name: recalls[name].section if name in recalls else text
                for name, text in sections.items()
                if name in recalls or name.startswith(INTERMEDIATE_PROMPT_PREFIX)
            }
            from .concurrency import object_lock

            for name, recall in recalls.items():
                mem = self._by_name[name]
                with object_lock(mem):
                    mem.record_recall(recall.items)
        else:
            memory_tokens = self._count_memory_tokens("\n\n".join(r.section for r in recalls.values()))
        return ContextSnapshot(
            sections=sections,
            recalls=recalls,
            query=query,
            user_id=user_id,
            participants=list(participants or [user_id]),
            temporal_resolution=temporal_resolution,
            memory_token_count=memory_tokens,
            memory_budget=budget,
            memory_types=list(memory_types) if memory_types is not None else None,
        )

    def build_context(
        self,
        query: Query,
        user_id: str = "default",
        limits: dict[str, int] = {},
        participants: Optional[list[str]] = None,
        temporal_resolution: Optional[TemporalResolution] = None,
        temporal_weight: float = 1.0,
        *,
        budget: Budget = UNSET,
        reranker: Optional[MemoryReranker] = None,
        memory_types: Optional[Sequence[str]] = None,
    ) -> dict[str, str]:
        """Return ordered rendered memory sections and intermediate prompts."""
        return self.build_context_snapshot(
            query,
            user_id,
            limits=limits,
            participants=participants,
            temporal_resolution=temporal_resolution,
            temporal_weight=temporal_weight,
            budget=budget,
            reranker=reranker,
            memory_types=memory_types,
        ).sections

    def _header_for_multi(self, name: str) -> str:
        """Section header for the multi-participant rendering (pluralised)."""
        if self.prompts is not None:
            header = getattr(self.prompts, f"{name}_header_multi", None)
            if header:
                return header
        # Fall back to the singular header.
        return self._header_for(name)

    def render_prompt(
        self,
        query: Query,
        user_id: str = "default",
        limits: dict[str, int] = {},
        participants: Optional[list[str]] = None,
        temporal_resolution: Optional[TemporalResolution] = None,
        temporal_weight: float = 1.0,
        *,
        budget: Budget = UNSET,
        reranker: Optional[MemoryReranker] = None,
        memory_types: Optional[Sequence[str]] = None,
    ) -> str:
        """Full system-style context block (system line + all sections)."""
        sections = self.build_context(
            query,
            user_id,
            limits=limits,
            participants=participants,
            temporal_resolution=temporal_resolution,
            temporal_weight=temporal_weight,
            budget=budget,
            reranker=reranker,
            memory_types=memory_types,
        )
        if self.prompts is None:
            parts = []
            if self.base_instruction:
                parts.append(self.base_instruction)
            parts.extend(sections.values())
            return "\n\n".join(parts)
        system = self.prompts.system.format(
            character_name=self.character_name,
            base_instruction=self.base_instruction.strip(),
        )
        return system + "\n\n" + "\n\n".join(sections[n] for n in self.prompts.section_order if n in sections)
