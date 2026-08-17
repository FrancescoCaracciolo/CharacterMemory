"""Prompt templates.

Every string the agent injects into the LLM prompt lives here and is
overridable via `PromptConfig`. Each placeholder ``{name}`` is filled by
the agent at render time.
"""


from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """All prompt templates, each individually overridable."""

    system: str = (
        "{base_instruction}\n\n"
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
    conversation_events_header: str = "Relevant source conversations"
    heartbeat_header: str = "Recent discoveries / actions of yours"
    user_summary_header: str = "Summary of this user"
    user_summary_header_multi: str = "Summaries of these users"
    emotion_header: str = "Emotional state"
    world_header: str = "Current World"
    calendar_header: str = "Calendar"
    calendar_header_multi: str = "Calendar (character + participants)"
    knowledge_graph_header: str = "Activated knowledge (graph)"
    knowledge_graph_header_multi: str = "Activated knowledge (graph)"

    # Plural headers used when a section is rendered for a multi-participant
    # (group) chat. When a `*_header_multi` is set, the renderer picks it over
    # the singular header; otherwise it falls back to the singular one. Only
    # the per-user sections need a distinct plural form.
    user_facts_header_multi: str = "What you remember about these users"
    user_directives_header_multi: str = "Standing instructions from these users"
    episodic_header_multi: str = "Episodes you've shared with these users"
    conversation_events_header_multi: str = "Relevant source conversations"
    emotion_header_multi: str = "Emotional state (baseline + toward each user)"

    # Extraction LLM prompts (see character_memory.memory.extract). These render
    # the instruction built from the participating memories' ExtractionSpecs and
    # interpolate {character_name}, {user_name}, {persona_clause} as appropriate.
    extraction_header: str = (
        "You are the memory extractor for {character_name}, a role-play character"
        "{persona_clause}. Analyze the recent conversation and extract durable, "
        "reusable information about {user_name} and about events {character_name} "
        "experienced.\n"
    )
    extraction_sentence_rule: str = (
        "Each extracted item must be a self-contained full sentence that uses the "
        "real names ({character_name} for the character, {user_name} for the user). "
        "For example, write \"{user_name} has an exam on the 17th of July\" — not "
        "\"exam on 17th\" or \"the user mentioned an exam\"."
    )
    extraction_known_facts_intro: str = (
        "What you already know about {user_name} (do not re-extract these; only add "
        "genuinely new information):"
    )
    extraction_footer: str = (
        "Only include genuinely new, non-trivial items. Return [] where nothing "
        "fits, and omit fields entirely if they are not present in the schema.\n\n"
        "Recent conversation:"
    )
    # Appended in multi-user (group chat) extraction. Interpolates
    # {participants}; instructs the model to stamp each per-user item with the
    # participant it is about.
    extraction_multi_note: str = (
        "This conversation has several participants: {participants}. For each "
        "per-user field, set the item's `user_id` to the participant the item is "
        "about (one of the listed names). Only attribute an item to someone when "
        "the conversation actually establishes it about them."
    )

    # Deduplication LLM prompts (see character_memory.memory.dedup).
    dedup_judge: str = (
        "You are a duplicate detector for a character memory store. You will be "
        "given two memory entries. Decide whether they convey the same piece of "
        "information (one is a restatement, paraphrase, or subset of the other) or "
        "whether they are genuinely distinct. Minor wording or formatting "
        "differences do NOT make them distinct. "
        'Respond ONLY with a single JSON object: {"same": true} or {"same": false}.'
    )

    dedup_consolidate: str = (
        "You are consolidating two near-duplicate memory entries into one. Merge "
        "them into a single concise entry that preserves every distinct fact from "
        "both, removes redundancy, and keeps the same tone and tense. Do not invent "
        "new information. "
        'Respond ONLY with a single JSON object: {"text": "<the merged entry>."}'
    )

    dedup_contradict: str = (
        "You are checking whether two memory entries about the same subject "
        "CONTRADICT each other — they cannot both be true at the same time. "
        "Each entry is shown with the time it was recorded. USE THE TIMESTAMPS: "
        "two entries that describe different points in time (e.g. the user's "
        "mood on Monday vs Tuesday) are NOT contradictions, they are a change "
        "over time. A contradiction is when both assert incompatible facts about "
        "the same point in time (e.g. 'X is a doctor' vs 'X is an engineer'). "
        'Respond ONLY with: {"contradicts": true} or {"contradicts": false}.'
    )

    # Order in which memory sections appear in the prompt.
    section_order: list[str] = field(
        default_factory=lambda: [
            "character_info",
            "emotion",
            "world",
            "calendar",
            "user_directives",
            "user_facts",
            "episodic",
            "conversation_events",
            "heartbeat",
            "user_summary",
            "knowledge_graph",
            "dialogue_style",
        ]
    )
