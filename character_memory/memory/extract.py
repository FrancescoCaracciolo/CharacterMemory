"""LLM-based extraction of structured memories from a batch of turns.

Every `extract_interval` turns the agent hands the recent exchanges to an
`Extractor`, which asks the LLM (via structured output) to surface new
facts, directives, episodes, and emotion updates. The parsed result is applied
to the relevant memories.
"""

from typing import Any

from ..llm.base import LLMClient

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "content": {"type": "string"},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "content", "importance", "confidence"],
            },
        },
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "importance": {"type": "number"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["content", "importance", "keywords"],
            },
        },
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "importance": {"type": "number"},
                    "emotional_shift": {"type": "number"},
                },
                "required": ["summary", "importance", "emotional_shift"],
            },
        },
        "emotion_deltas": {
            "type": "object",
            "description": "Signed adjustments to per-user emotion dims.",
            "additionalProperties": {"type": "number"},
        },
    },
    "required": ["facts", "directives", "episodes", "emotion_deltas"],
}

DEFAULT_INSTRUCTION = (
    "You are a memory extractor for a role-play character. Analyze the recent "
    "conversation and extract durable, reusable information about the USER and "
    "about events.\n"
    "- facts: stable facts about the user (occupation, preferences, relationships, "
    "goals) or general facts the user stated. importance 0-1 (how much it should "
    "shape the character's behaviour), confidence 0-1.\n"
    "- directives: standing instructions the user gave (e.g. 'always answer "
    "formally'). importance 0-1. keywords: terms that should trigger retrieval.\n"
    "- episodes: notable things that happened. emotional_shift -1..1 (negative to "
    "positive) capturing how the event shifted the character's feelings.\n"
    "- emotion_deltas: small signed adjustments to the character's feelings toward "
    "this user (affection, valence, trust, ...).\n"
    "Only include genuinely new, non-trivial items. Return [] where nothing fits.\n\n"
    "Recent conversation:"
)


class Extractor:
    """Turns recent exchanges into structured-memory updates."""

    def __init__(self, llm: LLMClient, instruction: str = DEFAULT_INSTRUCTION) -> None:
        self.llm = llm
        self.instruction = instruction

    def extract(self, turns: list[dict[str, str]]) -> dict[str, Any]:
        """`turns` is a list of `{"role": "user"|"assistant", "content": ...}`."""
        transcript = "\n\n".join(
            f"{'User' if t['role'] == 'user' else 'Character'}: {t['content']}"
            for t in turns
            if t.get("content")
        )
        if not transcript.strip():
            return {"facts": [], "directives": [], "episodes": [], "emotion_deltas": {}}
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": transcript},
        ]
        try:
            result = self.llm.chat_structured(messages, SCHEMA)
        except Exception as exc:  # pragma: no cover - network/model errors
            return {"facts": [], "directives": [], "episodes": [], "emotion_deltas": {},
                    "error": str(exc)}
        result.setdefault("facts", [])
        result.setdefault("directives", [])
        result.setdefault("episodes", [])
        result.setdefault("emotion_deltas", {})
        return result
