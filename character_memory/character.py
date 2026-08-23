"""A character bundles an identity with the memory systems it can recall from and write to."""

from dataclasses import dataclass, field
from typing import Any, Optional
from character_memory.llm.base import LLMClient
from character_memory.prompts import PromptConfig
from .memory.extract import Extractor, ExtractionContext, build_extraction
from .memory.base import Memory, MemoryItem, RecallResult
from .memory.structured import StructuredMemory
from .rag.base import Query


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


class Character:

    def __init__(self, character_name: str, base_instruction: str = "", memories: Optional[list[Memory]] = None, llm: Optional[LLMClient] = None, prompts: Optional[PromptConfig] = None) -> None:
        self.character_name = character_name
        self.base_instruction = base_instruction
        self.memories = memories if memories is not None else []
        self.llm = llm
        self.prompts = prompts
        self._by_name = {m.name: m for m in self.memories}

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
        """
        order = self.prompts.section_order if self.prompts is not None else [m.name for m in self.memories]
        template = self.prompts.section_template if self.prompts is not None else "## {title}\n{body}"
        multi = bool(participants and len(participants) > 1)
        sections: dict[str, str] = {}
        recalls: dict[str, MemoryRecallSnapshot] = {}
        for name in order:
            mem = self._by_name.get(name)
            if mem is None or not mem.enabled:
                continue
            if multi:
                result: RecallResult = mem.build_section_participants_result(
                    query, participants, limits.get(name, 0)
                )
            else:
                result = mem.build_section_result(query, user_id, limits.get(name, 0))
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
        return ContextSnapshot(
            sections=sections,
            recalls=recalls,
            query=query,
            user_id=user_id,
            participants=list(participants or [user_id]),
        )

    def build_context(
        self,
        query: Query,
        user_id: str = "default",
        limits: dict[str, int] = {},
        participants: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Return `{memory_name: rendered_section}` for enabled, non-empty memories."""
        return self.build_context_snapshot(
            query, user_id, limits=limits, participants=participants
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
    ) -> str:
        """Full system-style context block (system line + all sections)."""
        sections = self.build_context(query, user_id, limits=limits, participants=participants)
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
