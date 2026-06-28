"""LLM-based extraction of structured memories from a batch of turns.

Every `extract_interval` turns the agent asks each *enabled* memory for an
:class:`~character_memory.memory.base.ExtractionSpec`, composes a single
combined JSON schema + instruction out of those specs, runs one structured-LLM
call, and hands each memory the value it asked for. Because the schema is built
from the memories themselves, a disabled (or self-skipping) memory contributes
nothing — that type simply isn't extracted.
"""

from typing import Any

from ..llm.base import LLMClient
from .base import ExtractionSpec

_INSTRUCTION_HEADER = (
    "You are a memory extractor for a role-play character. Analyze the recent "
    "conversation and extract durable, reusable information about the USER and "
    "about events.\n"
)
_INSTRUCTION_FOOTER = (
    "Only include genuinely new, non-trivial items. Return [] where nothing "
    "fits, and omit fields entirely if they are not present in the schema.\n\n"
    "Recent conversation:"
)


def build_extraction(specs: list[ExtractionSpec]) -> tuple[dict[str, Any], str]:
    """Compose one JSON schema + instruction from the given memory specs.

    Each spec contributes one property (keyed by ``spec.field``) and one bullet
    line to the instruction. The returned schema/instruction are passed to a
    single structured-LLM call.
    """
    properties = {s.field: s.schema for s in specs}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": [s.field for s in specs],
    }
    instruction = _INSTRUCTION_HEADER + "\n".join(s.instruction for s in specs) + "\n" + _INSTRUCTION_FOOTER
    return schema, instruction


class Extractor:
    """Turns recent exchanges into structured-memory updates."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(
        self,
        turns: list[dict[str, str]],
        schema: dict[str, Any],
        instruction: str,
    ) -> dict[str, Any]:
        """Run one structured extraction pass.

        `turns` is a list of `{"role": "user"|"assistant", "content": ...}`.
        `schema`/`instruction` are built by :func:`build_extraction` from the
        participating memories' specs.
        """
        transcript = "\n\n".join(
            f"{'User' if t['role'] == 'user' else 'Character'}: {t['content']}"
            for t in turns
            if t.get("content")
        )
        fields = list(schema.get("properties", {}).keys())
        empty = {f: ([] if schema["properties"][f].get("type") == "array" else {}) for f in fields}
        if not transcript.strip():
            return empty
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": transcript},
        ]
        try:
            result = self.llm.chat_structured(messages, schema)
        except Exception as exc:  # pragma: no cover - network/model errors
            return {**empty, "error": str(exc)}
        for f, default in empty.items():
            result.setdefault(f, default)
        return result
