"""A character bundles an identity with the memory systems it can recall from and write to."""

from typing import Optional
from character_memory.llm.base import LLMClient
from character_memory.prompts import PromptConfig
from .memory.extract import Extractor, ExtractionContext, build_extraction
from .memory.base import Memory, MemoryItem
from .memory.structured import StructuredMemory


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

    def _known_user_facts(self, user_id: str, limit: int = 8) -> list[str]:
        """Top known user-fact texts for `user_id`, highest importance first.

        Used to ground the extractor so it does not re-extract what is already
        stored. Returns [] when user_facts is absent or empty for this user.
        """
        mem = self._by_name.get("user_facts")
        if not isinstance(mem, StructuredMemory):
            return []
        rows = mem.all_rows(user_id=user_id)
        rows.sort(key=lambda r: float(r.get("importance", 0.0)), reverse=True)
        facts: list[str] = []
        for r in rows[:limit]:
            text = mem.row_text(r).strip()
            if text:
                facts.append(text)
        return facts

    def _extraction_context(self, user_id: str) -> ExtractionContext:
        """Build the identity + grounding context for an extraction call."""
        known = self._known_user_facts(user_id)
        p = self.prompts
        return ExtractionContext(
            character_name=self.character_name,
            user_name=user_id,
            persona=self.base_instruction.strip(),
            known_facts=known,
            header=p.extraction_header if p is not None else ExtractionContext.header,
            sentence_rule=p.extraction_sentence_rule if p is not None else ExtractionContext.sentence_rule,
            known_facts_intro=p.extraction_known_facts_intro if p is not None else ExtractionContext.known_facts_intro,
            footer=p.extraction_footer if p is not None else ExtractionContext.footer,
        )

    def extract(self, turns: list[dict[str, str]], user_id: str = "default", llm: Optional[LLMClient] = None) -> Optional[dict]:
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No LLM specified. Character.extract needs a LLM configured")
        context = self._extraction_context(user_id)
        # Only enabled memories that opt into extraction.
        participants = [
            (mem, spec)
            for mem in self.memories
            if mem.enabled
            for spec in [mem.extraction_spec(context)]
            if spec is not None
        ]
        if not participants:
            return None
        schema, instruction = build_extraction([spec for _, spec in participants], context=context)
        extracted = Extractor(llm).extract(turns, schema=schema, instruction=instruction, context=context)
        added: dict[str, list[MemoryItem]] = {}
        for mem, spec in participants:
            items = mem.apply_extraction(extracted.get(spec.field), user_id)
            if items:
                added[mem.name] = items
        # Carry the freshly-added items so the caller (e.g. a deduplicator)
        # can act on them without re-querying the memories.
        extracted["__added__"] = added
        return extracted

    def build_context(self, query: str, user_id: str = "default", limits: dict[str, int] = {}) -> dict[str, str]:
        """Return `{memory_name: rendered_section}` for enabled, non-empty memories."""
        order = self.prompts.section_order if self.prompts is not None else [m.name for m in self.memories]
        template = self.prompts.section_template if self.prompts is not None else "## {title}\n{body}"
        sections: dict[str, str] = {}
        for name in order:
            mem = self._by_name.get(name)
            if mem is None or not mem.enabled:
                continue
            body = mem.build_section(query, user_id, limits.get(name, 0))
            if not body:
                continue
            sections[name] = template.format(title=self._header_for(name), body=body)
        return sections

    def render_prompt(self, query: str, user_id: str = "default", limits: dict[str, int] = {}) -> str:
        """Full system-style context block (system line + all sections)."""
        sections = self.build_context(query, user_id, limits=limits)
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
