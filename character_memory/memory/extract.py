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

For **group chats** (multiple participants), pass ``participants`` on the
context. The extractor then labels each transcript turn with its real speaker,
augments every ``per_user`` spec's item schema with a ``user_id`` enum of the
participants, and adds an attribution rule; each per-user memory's
``apply_extraction`` honours the per-item ``user_id``. Single-participant
contexts (the default) are bit-for-bit identical to the legacy single-user path.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
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

# Extra rule appended in multi-user (group chat) extraction: per-user fields
# must stamp each item with the participant it is about. Exposed on
# `PromptConfig.extraction_multi_note` so it is overridable like every prompt.
_DEFAULT_MULTI_NOTE = (
    "This conversation has several participants: {participants}. For each "
    "per-user field, set the item's `user_id` to the participant the item is "
    "about (one of the listed names). Only attribute an item to someone when "
    "the conversation actually establishes it about them."
)

_PROVENANCE_NOTE = (
    "Every extracted fact, directive, and episode must include "
    "`source_message_ids`: the IDs of the transcript messages that directly "
    "support it. Cite only IDs shown in the transcript."
)

_SNAPSHOT_NOTE = (
    "Snapshot fields ({fields}) are complete replacement values, not deltas. "
    "When returning an item for one of these fields, start from the current "
    "stored snapshot supplied in that field's instructions and follow that "
    "field's selection and length rules. Retain supported core information, "
    "incorporate relevant new information, and remove details outside those rules. "
    "A snapshot may need rewriting to meet those rules even without new facts. "
    "Return an empty array when the existing snapshot already meets the rules "
    "and nothing material changed, or when there is insufficient information "
    "to create a new snapshot."
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

    ``participants`` lists the human speakers in a group chat. When it has more
    than one entry the extraction runs in multi-user mode: the transcript is
    labelled with each turn's real speaker, per-user fields are augmented with
    a ``user_id`` enum, and an attribution note is added. Empty or single-entry
    ``participants`` keeps the legacy single-user behaviour.
    """

    character_name: str = "Character"
    user_name: str = "User"
    persona: str = ""
    known_facts: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    # Prompt templates (override via PromptConfig); interpolated at build time.
    header: str = _DEFAULT_EXTRACTION_HEADER
    sentence_rule: str = _DEFAULT_SENTENCE_RULE
    known_facts_intro: str = _DEFAULT_KNOWN_FACTS_INTRO
    footer: str = _DEFAULT_EXTRACTION_FOOTER
    multi_note: str = _DEFAULT_MULTI_NOTE

    @property
    def multi_user(self) -> bool:
        """True when this extraction call should attribute items per speaker."""
        return len(self.participants) > 1


def _augment_per_user_schema(schema: dict[str, Any], participants: list[str]) -> dict[str, Any]:
    """Stamp a per-user ``user_id`` enum onto every array-of-objects item schema.

    Operates on a per-field schema fragment (the value stored under
    ``ExtractionSpec.schema``). For array-of-objects fields it adds a
    ``user_id`` property whose enum is the participants, and lists it as
    required, so the LLM must attribute each extracted item. Non-array or
    non-object-item schemas are returned unchanged (e.g. the multi-user emotion
    spec already carries its own ``user_id``).
    """
    if not participants or schema.get("type") != "array":
        return schema
    items = schema.get("items")
    if not isinstance(items, dict) or items.get("type") != "object":
        return schema
    new_items = {k: (dict(v) if isinstance(v, dict) else v) for k, v in items.items()}
    props = dict(new_items.get("properties") or {})
    props["user_id"] = {"type": "string", "enum": list(participants)}
    new_items["properties"] = props
    required = list(new_items.get("required") or [])
    if "user_id" not in required:
        required.append("user_id")
    new_items["required"] = required
    new_schema = dict(schema)
    new_schema["items"] = new_items
    return new_schema


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
    parts.append(_PROVENANCE_NOTE)
    if context.multi_user:
        parts.append(
            context.multi_note.format(participants=", ".join(context.participants))
        )
    parts.append(context.footer)
    snapshot_specs = [s for s in specs if s.snapshot]
    if snapshot_specs:
        parts.append(
            _SNAPSHOT_NOTE.format(fields=", ".join(s.field for s in snapshot_specs))
        )
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

    In multi-user mode (``context.multi_user``) each ``per_user`` spec's
    array-of-objects schema is augmented with a ``user_id`` enum of the
    participants (see :func:`_augment_per_user_schema`).
    """
    participants = context.participants if (context and context.multi_user) else []
    properties = {
        s.field: (
            _augment_per_user_schema(s.schema, participants)
            if s.per_user and participants
            else s.schema
        )
        for s in specs
    }
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
            + _PROVENANCE_NOTE
            + "\n"
            + _INSTRUCTION_FOOTER
        )
    else:
        instruction = _format_instruction(specs, context)
    snapshot_specs = [s for s in specs if s.snapshot]
    if snapshot_specs and context is None:
        instruction += "\n\n" + _SNAPSHOT_NOTE.format(
            fields=", ".join(s.field for s in snapshot_specs)
        )
    return schema, instruction


class Extractor:
    """Turns recent exchanges into structured-memory updates."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(
        self,
        turns: list[dict[str, Any]],
        schema: dict[str, Any],
        instruction: str,
        *,
        context: Optional[ExtractionContext] = None,
    ) -> dict[str, Any]:
        """Run one structured extraction pass.

        `turns` contains role/content plus an optional speaker, message ID, and
        occurrence timestamp, as produced by
        :meth:`Chat.messages_with_speakers`.
        `schema`/`instruction` are built by :func:`build_extraction` from the
        participating memories' specs.

        When ``context`` is given, the transcript labels speakers with the real
        character/user names instead of the generic ``User``/``Character``. In
        multi-user mode each user turn is labelled with its own speaker.
        """
        if context is not None:
            user_name = context.user_name or "User"
            char_name = context.character_name or "Character"
        else:
            user_name = "User"
            char_name = "Character"
        multi = bool(context and context.multi_user)
        lines: list[str] = []
        for t in turns:
            content = t.get("content")
            if not content:
                continue
            if t.get("role") == "user":
                # In a group chat prefer the turn's real speaker; otherwise the
                # single user name.
                speaker = (t.get("user_id") or user_name) if multi else user_name
                label = f"{speaker}: {content}"
            else:
                label = f"{char_name}: {content}"
            annotations: list[str] = []
            if t.get("message_id") is not None:
                annotations.append(f"message_id={int(t['message_id'])}")
            if t.get("occurred_at") is not None:
                timestamp = datetime.fromtimestamp(float(t["occurred_at"]), tz=UTC)
                annotations.append(f"occurred_at={timestamp.isoformat()}")
            prefix = f"[{' | '.join(annotations)}] " if annotations else ""
            lines.append(prefix + label)
        transcript = "\n\n".join(lines)
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
        valid_message_ids = {
            int(turn["message_id"])
            for turn in turns
            if turn.get("message_id") is not None
        }
        for value in result.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict) or "source_message_ids" not in item:
                    continue
                supplied = item.get("source_message_ids")
                if not isinstance(supplied, list):
                    item["source_message_ids"] = []
                    continue
                item["source_message_ids"] = [
                    int(message_id)
                    for message_id in supplied
                    if isinstance(message_id, int)
                    and not isinstance(message_id, bool)
                    and int(message_id) in valid_message_ids
                ]
        for f, default in empty.items():
            result.setdefault(f, default)
        return result
