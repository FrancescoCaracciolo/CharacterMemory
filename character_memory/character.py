

from typing import Optional
from character_memory.llm.base import LLMClient
from character_memory.prompts import PromptConfig
from .memory.extract import Extractor, build_extraction
from .memory.base import Memory


class Character:

    def __init__(self, character_name: str, base_instruction: str = "", memories: list[Memory] = [], llm: Optional[LLMClient] = None, prompts: Optional[PromptConfig] = None) -> None:
        self.character_name = character_name
        self.base_instruction = base_instruction
        self.memories = memories
        self.llm = llm
        self.prompts = prompts
        self._by_name = {m.name: m for m in memories}

    def add_memory(self, memory: Memory):
        self.memories.append(memory)
        self._by_name[memory.name] = memory

    def _header_for(self, name: str) -> str:
        if self.prompts is not None:
            header = getattr(self.prompts, f"{name}_header", None)
            if header:
                return header
        return self._by_name[name].title

    def extract(self, turns: list[dict[str, str]], user_id: str = "default", llm: Optional[LLMClient] = None) -> Optional[dict]:
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No LLM specified. Character.extract needs a LLM configured")
        # Only enabled memories that opt into extraction.
        participants = [
            (mem, spec)
            for mem in self.memories
            if mem.enabled
            for spec in [mem.extraction_spec()]
            if spec is not None
        ]
        if not participants:
            return None
        schema, instruction = build_extraction([spec for _, spec in participants])
        extracted = Extractor(llm).extract(turns, schema=schema, instruction=instruction)
        for mem, spec in participants:
            mem.apply_extraction(extracted.get(spec.field), user_id)
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
            return "\n\n".join(sections.values())
        system = self.prompts.system.format(character_name=self.character_name)
        return system + "\n\n" + "\n\n".join(sections[n] for n in self.prompts.section_order if n in sections)
