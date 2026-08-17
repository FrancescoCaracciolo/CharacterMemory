"""Persisted per-character manifest.

.. deprecated::
    Superseded by :mod:`character_memory.character_config` and the comprehensive
    ``config.yaml`` format. The agent no longer reads or writes
    ``character.json``; the new loader reads a legacy manifest for one-way
    migration only. This module is kept importable so older scripts that
    construct ``CharacterManifest`` directly keep working.

Historically a "character" was *only* a directory (``assets/<Name>/`` with
``Information/`` + ``Dialogues/`` subfolders). Its identity (persona), the
prompt templates, and every memory toggle lived exclusively in Python code —
nothing was persisted per character, so configuring one from the GUI was
impossible without writing code.

``CharacterManifest`` is the JSON-serialisable source of truth introduced to
fix that. It is written to ``<character_dir>/character.json`` and maps onto the
existing :class:`MemoryConfig` / :class:`PromptConfig` dataclasses plus the
agent's ``persona`` argument, so the in-process ``CharacterAgent`` is built
from the same values the GUI showed. Characters that pre-date the manifest
(Kurisu, for instance) simply have no file and fall back to the dataclass
defaults — preserving today's behaviour.

Scope is deliberately limited to what the GUI exposes: persona + system prompt
+ section order, the per-memory ``enabled_*`` toggles + ``*_k`` retrieval
sizes, ``emotion_baseline`` and ``extract_interval``. Extraction / dedup /
knowledge-graph tuning prompts stay in code; per-character LLM credentials
stay global (``OPENAI_*``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .config import MemoryConfig
from .prompts import PromptConfig

# All memory names the agent can build, in prompt-section order. Kept in sync
# with MemoryConfig's enabled_* / *_k fields and PromptConfig.section_order.
MEMORY_NAMES: tuple[str, ...] = (
    "character_info",
    "dialogue_style",
    "user_facts",
    "user_directives",
    "episodic",
    "conversation_events",
    "emotion",
    "world",
    "calendar",
    "heartbeat",
    "user_summary",
    "knowledge_graph",
)

MANIFEST_FILENAME = "character.json"


def _default_section_order() -> list[str]:
    # Mirror PromptConfig.section_order's default so a fresh manifest matches
    # what the library would have produced on its own.
    return list(PromptConfig().section_order)


def _default_emotion_baseline() -> dict[str, float]:
    return dict(MemoryConfig().emotion_baseline)


@dataclass
class CharacterManifest:
    """Persisted per-character configuration (``character.json``)."""

    name: str
    persona: str = ""
    # Top-level system prompt template; {character_name} / {base_instruction}
    # are interpolated at render time. Empty -> PromptConfig default.
    system_prompt: str = ""
    # Order memory sections appear in the prompt (subset of MEMORY_NAMES).
    section_order: list[str] = field(default_factory=_default_section_order)
    # Section header overrides keyed by memory name; empty -> PromptConfig default.
    section_headers: dict[str, str] = field(default_factory=dict)
    # Per-memory toggles. Anything not present defaults to the library default.
    enabled: dict[str, bool] = field(default_factory=dict)
    # Per-memory retrieval sizes (top-k).
    k_sizes: dict[str, int] = field(default_factory=dict)
    # Emotion baseline (dims -> 0..1) and how often extraction runs (turns).
    emotion_baseline: dict[str, float] = field(default_factory=_default_emotion_baseline)
    extract_interval: Optional[int] = None

    # ------------------------------------------------------------------ disk
    @classmethod
    def path_for(cls, character_dir: str) -> str:
        return os.path.join(character_dir, MANIFEST_FILENAME)

    @classmethod
    def load(cls, character_dir: str, *, name: Optional[str] = None) -> "CharacterManifest":
        """Read the manifest under ``character_dir``.

        Falls back to a synthesised default (using ``name`` or the folder name)
        when no file exists, so callers always get a writable manifest back.
        """
        fallback_name = name or os.path.basename(os.path.normpath(character_dir))
        path = cls.path_for(character_dir)
        if not os.path.isfile(path):
            return cls(name=fallback_name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            return cls(name=fallback_name)
        data.setdefault("name", fallback_name)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterManifest":
        """Build a manifest, ignoring unknown keys (forward-compatible)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    def save(self, character_dir: str) -> str:
        """Write the manifest under ``character_dir`` and return its path."""
        os.makedirs(character_dir, exist_ok=True)
        path = self.path_for(character_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------- map onto configs
    def to_memory_config(self, base: Optional[MemoryConfig] = None) -> MemoryConfig:
        """Build a MemoryConfig applying this manifest's overrides on ``base``."""
        m = base or MemoryConfig()
        for name, on in self.enabled.items():
            attr = f"enabled_{name}"
            if hasattr(m, attr):
                setattr(m, attr, bool(on))
        for name, k in self.k_sizes.items():
            if hasattr(m, f"{name}_k"):
                setattr(m, f"{name}_k", int(k))
        if self.emotion_baseline:
            m.emotion_baseline = dict(self.emotion_baseline)
        if self.extract_interval is not None:
            m.extract_interval = max(1, int(self.extract_interval))
        return m

    def to_prompt_config(self, base: Optional[PromptConfig] = None) -> PromptConfig:
        """Build a PromptConfig applying this manifest's overrides on ``base``."""
        p = base or PromptConfig()
        if self.system_prompt.strip():
            p.system = self.system_prompt
        if self.section_order:
            p.section_order = list(self.section_order)
        for name, header in self.section_headers.items():
            attr = f"{name}_header"
            if hasattr(p, attr) and header:
                setattr(p, attr, header)
        return p
