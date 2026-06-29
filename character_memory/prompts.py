"""Prompt templates.

Every string the agent injects into the LLM prompt lives here and is
overridable via `PromptConfig`. Each placeholder ``{name}`` is filled by
the agent at render time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """All prompt templates, each individually overridable."""

    system: str = (
        "You are {character_name}, role-playing as this character. Stay in "
        "character at all times and reflect the personality, knowledge and "
        "speech style shown in the provided material. Use the memories below "
        "to inform your responses, but do not break character to mention them."
    )

    emotion_note: str = (
        "Current emotional state — baseline: {baseline}; "
        "toward this user: {user_state}. Let this colour your tone."
    )

    section_template: str = "## {title}\n{body}"

    character_info_header: str = "Character Information"
    dialogue_style_header: str = "Example Exchanges (style reference)"
    user_facts_header: str = "What you remember about this user"
    user_directives_header: str = "Standing instructions from this user"
    episodic_header: str = "Episodes you've shared with this user"
    heartbeat_header: str = "Recent discoveries / actions of yours"
    emotion_header: str = "Emotional state"

    # Order in which memory sections appear in the prompt.
    section_order: list[str] = field(
        default_factory=lambda: [
            "character_info",
            "emotion",
            "user_directives",
            "user_facts",
            "episodic",
            "heartbeat",
            "dialogue_style",
        ]
    )
