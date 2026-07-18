"""LLM-based extraction of structured memories from a batch of turns.

Every `extract_interval` turns the agent asks each *enabled* memory for an
:class:`~character_memory.memory.base.ExtractionSpec`, composes a single
combined JSON schema + instruction out of those specs, runs one structured-LLM
call, and hands each memory the value it asked for. Because the schema is built
from the memories themselves, a disabled (or self-skipping) memory contributes
nothing — that type simply isn't extracted.

The extractor can optionally be made aware of *who* is in the conversation by
passing an :class:`ExtractionContext`. When provided, the prompt names the
character and the user, surfaces the character's short persona, lists the user
facts already stored (so the model does not re-extract them), enforces a
full-sentence rule, and the transcript uses the real speaker names instead of
generic ``User``/``Character`` labels. Omitting the context keeps the legacy
behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.base import LLMClient
from .base import ExtractionSpec

# Default prompt fragments. These are also exposed as overridable fields on
# `PromptConfig` (`extraction_header`, `extraction_sentence_rule`,
# `extraction_known_facts_intro`, `extraction_footer`) so callers can change the
# wording exactly like they already override the dedup prompts.

_DEFAULT_EXTRACTION_HEADER = (
    "You are the memory extractor for {character_name}, a role-play character"
    "{persona_clause}. Analyze the recent conversation and extract durable, "
    "reusable information about {user_name} and about events {character_name} "
    "experienced.\n"
)
_DEFAULT_SENTENCE_RULE = (
    "Each extracted item must be a self-contained full sentence that uses the "
    "real names ({character_name} for the character, {user_name} for the user). "
    "For example, write \"{user_name} has an exam on the 17th of July\" — not "
    "\"exam on 17th\" or \"the user mentioned an exam\"."
)
_DEFAULT_KNOWN_FACTS_INTRO = (
    "What you already know about {user_name} (do not re-extract these; only add "
    "genuinely new information):"
)
_DEFAULT_EXTRACTION_FOOTER = (
    "Only include genuinely new, non-trivial items. Return [] where nothing "
    "fits, and omit fields entirely if they are not present in the schema.\n\n"
    "Recent conversation:"
)

# Legacy header/footer kept for the no-context path (backward compatibility).
_INSTRUCTION_HEADER = (
    "You are a memory extractor for a role-play character. Analyze the recent "
    "conversation and extract durable, reusable information about the USER and "
    "about events.\n"
)
_INSTRUCTION_FOOTER = _DEFAULT_EXTRACTION_FOOTER


@dataclass
class ExtractionContext:
    """Identity + grounding context threaded through a single extraction call.

    All fields have safe defaults, so callers can populate only what they know.
    When this is passed to :func:`build_extraction` / :meth:`Extractor.extract`,
    the prompt is rendered with the real character/user names, the character's
    short persona, the user facts already stored, and a full-sentence rule.
    """

    character_name: str = "Character"
    user_name: str = "User"
    persona: str = ""
    known_facts: list[str] = field(default_factory=list)
    # Prompt templates (override via PromptConfig); interpolated at build time.
    header: str = _DEFAULT_EXTRACTION_HEADER
    sentence_rule: str = _DEFAULT_SENTENCE_RULE
    known_facts_intro: str = _DEFAULT_KNOWN_FACTS_INTRO
    footer: str = _DEFAULT_EXTRACTION_FOOTER


def _format_instruction(specs: list[ExtractionSpec], context: ExtractionContext) -> str:
    """Compose the full instruction string (header + bullets + footer)."""
    persona_clause = f", described as: {context.persona}" if context.persona else ""
    header = context.header.format(
        character_name=context.character_name,
        user_name=context.user_name,
        persona_clause=persona_clause,
    )
    sentence_rule = context.sentence_rule.format(
        character_name=context.character_name,
        user_name=context.user_name,
    )
    parts = [header.rstrip(), sentence_rule.rstrip()]
    if context.known_facts:
        intro = context.known_facts_intro.format(user_name=context.user_name)
        bullets = "\n".join(f"- {f}" for f in context.known_facts)
        parts.append(f"{intro}\n{bullets}")
    parts.append("\n".join(s.instruction for s in specs))
    parts.append(context.footer)
    return "\n\n".join(parts)


def build_extraction(
    specs: list[ExtractionSpec],
    context: Optional[ExtractionContext] = None,
) -> tuple[dict[str, Any], str]:
    """Compose one JSON schema + instruction from the given memory specs.

    Each spec contributes one property (keyed by ``spec.field``) and one bullet
    line to the instruction. The returned schema/instruction are passed to a
    single structured-LLM call.

    When ``context`` is given the instruction is rendered with the
    participants' real names, the character persona, the stored user facts, and
    a full-sentence rule (see :class:`ExtractionContext`). Without it the legacy
    generic header is used.
    """
    properties = {s.field: s.schema for s in specs}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": [s.field for s in specs],
    }
    if context is None:
        instruction = (
            _INSTRUCTION_HEADER
            + "\n".join(s.instruction for s in specs)
            + "\n"
            + _INSTRUCTION_FOOTER
        )
    else:
        instruction = _format_instruction(specs, context)
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
        *,
        context: Optional[ExtractionContext] = None,
    ) -> dict[str, Any]:
        """Run one structured extraction pass.

        `turns` is a list of `{"role": "user"|"assistant", "content": ...}`.
        `schema`/`instruction` are built by :func:`build_extraction` from the
        participating memories' specs.

        When ``context`` is given, the transcript labels speakers with the real
        character/user names instead of the generic ``User``/``Character``.
        """
        if context is not None:
            user_name = context.user_name or "User"
            char_name = context.character_name or "Character"
        else:
            user_name = "User"
            char_name = "Character"
        transcript = "\n\n".join(
            f"{user_name if t['role'] == 'user' else char_name}: {t['content']}"
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
