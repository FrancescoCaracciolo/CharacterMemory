"""Per-character YAML configuration: the single source of truth for prompts
and every sub-config.

A character is, at its core, a directory on disk (``assets/<Name>/`` with
``Information/`` + ``Dialogues/`` subfolders). Every tunable that used to live
only in Python — identity (persona), the prompt templates, the LLM/embedding
endpoints, chunking, the per-memory toggles and retrieval sizes, decay,
deduplication, and the knowledge-graph parameters — now also has a persisted,
hand-editable home: ``<character_dir>/config.yaml``.

Load it in one line::

    agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
    agent.load_from_config("assets/Kurisu/config.yaml")
    agent.build()

Or, because the file lives next to the character dir, omit the path and the
agent will discover it during ``load_from_config()`` — see
:func:`load_from_character_dir`.

Schema (every section is optional; missing keys fall back to the dataclass
default, unknown keys are ignored so the format is forward-compatible)::

    name: Kurisu
    persona: "A neuroscience researcher..."
    llm:        { base_url, api_key, model, temperature, max_tokens, timeout }
    embedding:  { base_url, api_key, model, dim, batch_size, timeout }
    chunking:   { info_chunker, dialogue_chunker, header_max_tokens,
                  header_min_tokens, dialogue_turns_per_chunk,
                  dialogue_context_width }
    memory:
      enabled_character_info: true     # and every other enabled_* toggle
      character_info_k: 4              # and every other *_k size
      sticky_threshold, extract_interval, decay_half_life
      emotion_baseline: { ... }
      emotion_user_dims: { ... }
      enabled_knowledge_graph: false
      dedup:          { enabled, exact, similarity_threshold, llm_judge,
                        consolidate, per_user, candidate_pool }
      knowledge_graph: { decay, decay_half_life, gain, hops, hop_decay,
                         base_weight, spread_weight, min_activation,
                         hebbian_threshold, hebbian_lr, self_seed,
                         match_base, match_gain, fact_batch_size,
                         episode_batch_size, wiki_batch_size,
                         extraction_token_limit }
    prompts:
      system, emotion_note, section_template, *_header, *_header_multi,
      extraction_*, dedup_*, section_order

Backward compatibility: when ``config.yaml`` is absent but a legacy
``character.json`` manifest exists, :func:`load_config` reads the manifest's
``persona`` + ``enabled`` + ``k_sizes`` + ``section_order`` + ``system_prompt``
into the new structures, so characters created by an older webUI keep working.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union

import yaml

from .config import (
    CharacterMemoryConfig,
    ChunkingConfig,
    DedupConfig,
    EmbeddingConfig,
    KnowledgeGraphConfig,
    LLMConfig,
    MemoryConfig,
)
from .prompts import PromptConfig

CONFIG_FILENAME = "config.yaml"
# Legacy manifest the webUI used to write. Read for back-compat, never written.
LEGACY_MANIFEST_FILENAME = "character.json"

# Patterns a persona may use to mention the character's other names. Used only
# to *supplement* hand-authored `aliases:` — never as the sole source. Matches
# quoted nicknames ("Christina"), parenthetical aliases (Makise Kurisu), and
# explicit "also known as / aka / AKA" clauses.
_PERSONA_ALIAS_QUOTED = re.compile(r'[“"]([^”"]{2,40})[”"]')
_PERSONA_ALIAS_PAREN = re.compile(r"\(([^)]{2,40})\)")
# The AKA name class deliberately excludes '.' (a period ends the alias) so
# "Also known as Kuri." captures just "Kuri", not the rest of the sentence.
_PERSONA_ALIAS_AKA = re.compile(
    r"(?:also known as|aka|a\.k\.a\.?)\s+([A-Za-z][\w '\-]{1,40})",
    re.IGNORECASE,
)


def _extract_aliases_from_persona(persona: str, name: str = "") -> list[str]:
    """Best-effort scan of `persona` for the character's other names.

    Supplements a hand-authored ``aliases:`` list so a character whose
    persona mentions nicknames (e.g. *Christina*, *Makise Kurisu*) gets the
    self-dedup benefit without the author having to list every variant. The
    canonical `name` is excluded, as are values that are obviously not a
    name (contain a sentence-ending period, or are the whole persona).
    Returns a de-duplicated list, order-preserving.
    """
    if not persona:
        return []
    name_lc = (name or "").strip().lower()
    found: list[str] = []
    for pat in (_PERSONA_ALIAS_QUOTED, _PERSONA_ALIAS_PAREN, _PERSONA_ALIAS_AKA):
        for m in pat.findall(persona):
            cand = (m or "").strip().strip("\u201c\u201d\"'.,")
            if not cand or len(cand) > 40:
                continue
            # Skip full sentences / clauses, not names. A trailing period
            # (e.g. "Also known as Kuri.") was already stripped above; this
            # guards against multi-sentence parentheticals.
            if cand.count(" ") > 4:
                continue
            if cand.lower() == name_lc:
                continue
            found.append(cand)
    # De-dup case-insensitively, keep first-seen surface form.
    seen: set[str] = set()
    out: list[str] = []
    for c in found:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Path helpers.
# --------------------------------------------------------------------------- #
def path_for(character_dir: str) -> str:
    """Return the canonical config path inside ``character_dir``."""
    return os.path.join(character_dir, CONFIG_FILENAME)


def _resolve_path(path_or_dir: Union[str, os.PathLike]) -> str:
    """Resolve ``path_or_dir`` to a concrete file path.

    Accepts either a path that already points to a ``config.yaml`` file, or a
    directory that should contain one.
    """
    p = os.fspath(path_or_dir)
    if os.path.isdir(p):
        return path_for(p)
    return p


# --------------------------------------------------------------------------- #
# Generic dataclass <-> dict bridging.
# --------------------------------------------------------------------------- #
# Fields whose dataclass default is a nested config dataclass. We use these to
# recurse when building from a dict.
_SUBCONFIG_TYPES = {
    LLMConfig: "llm",
    EmbeddingConfig: "embedding",
    ChunkingConfig: "chunking",
    MemoryConfig: "memory",
    DedupConfig: "dedup",
    KnowledgeGraphConfig: "knowledge_graph",
}


def _coerce_scalar(typ: Any, value: Any) -> Any:
    """Best-effort coerce a YAML scalar to the dataclass field's annotation."""
    if value is None:
        return None
    # PEP 604 unions (e.g. ``int | None``) — accept the value as-is when it is
    # already the right shape, otherwise try the first non-None type.
    typename = getattr(typ, "__name__", str(typ))
    try:
        if typename in ("int",) and not isinstance(value, bool):
            return int(value)
        if typename in ("float",):
            return float(value)
        if typename in ("str",):
            return str(value)
        if typename in ("bool",):
            return bool(value)
    except (TypeError, ValueError):
        return value
    return value


def _build_subconfig(cls: type, data: Optional[dict[str, Any]]) -> Any:
    """Construct a config dataclass from a dict, ignoring unknown keys.

    Nested dataclass-valued fields (``MemoryConfig.dedup`` /
    ``MemoryConfig.knowledge_graph``) are recursed into. Missing keys fall back
    to the dataclass default.
    """
    if data is None:
        return cls()
    if not isinstance(data, dict):
        return cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        # Recurse into nested config dataclasses.
        if f.type in _SUBCONFIG_TYPES and isinstance(raw, dict):
            kwargs[f.name] = _build_subconfig(f.type, raw)
        elif is_dataclass(getattr(f.type, "__origin__", f.type)):
            # Unresolved string annotation; skip — we don't expect any beyond
            # the subconfig types handled above.
            kwargs[f.name] = raw
        else:
            kwargs[f.name] = _coerce_scalar(f.type, raw)
    return cls(**kwargs)


def _build_memory_config(data: Optional[dict[str, Any]]) -> MemoryConfig:
    """Build a MemoryConfig, recursing into its dedup/knowledge_graph sub-configs."""
    if data is None:
        return MemoryConfig()
    if not isinstance(data, dict):
        return MemoryConfig()
    kwargs: dict[str, Any] = {}
    for f in fields(MemoryConfig):
        if f.name not in data:
            continue
        raw = data[f.name]
        if f.name == "dedup":
            kwargs["dedup"] = _build_subconfig(DedupConfig, raw if isinstance(raw, dict) else None)
        elif f.name == "knowledge_graph":
            kwargs["knowledge_graph"] = _build_subconfig(
                KnowledgeGraphConfig, raw if isinstance(raw, dict) else None
            )
        elif isinstance(raw, dict) and f.type in (dict, "dict"):
            kwargs[f.name] = dict(raw)
        elif isinstance(raw, list) and f.type in (list, "list"):
            kwargs[f.name] = list(raw)
        else:
            kwargs[f.name] = _coerce_scalar(f.type, raw)
    return MemoryConfig(**kwargs)


def _build_prompt_config(data: Optional[dict[str, Any]]) -> PromptConfig:
    """Build a PromptConfig from a dict, ignoring unknown keys."""
    if data is None or not isinstance(data, dict):
        return PromptConfig()
    known = {f.name for f in fields(PromptConfig)}
    clean: dict[str, Any] = {}
    for k, v in data.items():
        if k not in known:
            continue
        if k == "section_order":
            clean[k] = list(v) if isinstance(v, list) else v
        else:
            clean[k] = v if v is not None else ""
    return PromptConfig(**clean)


def build_full_config(data: dict[str, Any]) -> tuple[CharacterMemoryConfig, PromptConfig]:
    """Build the full config pair (``CharacterMemoryConfig``, ``PromptConfig``)."""
    full = CharacterMemoryConfig(
        llm=_build_subconfig(LLMConfig, data.get("llm")),
        embedding=_build_subconfig(EmbeddingConfig, data.get("embedding")),
        chunking=_build_subconfig(ChunkingConfig, data.get("chunking")),
        memory=_build_memory_config(data.get("memory")),
    )
    prompts = _build_prompt_config(data.get("prompts"))
    return full, prompts


# --------------------------------------------------------------------------- #
# Legacy ``character.json`` back-compat.
# --------------------------------------------------------------------------- #
def _migrate_legacy_manifest(character_dir: str) -> Optional[dict[str, Any]]:
    """Read a legacy ``character.json`` manifest into the new config.yaml schema.

    Returns ``None`` when there is no manifest. The mapping is intentionally
    one-way and lossy: only persona + memory toggles + retrieval sizes +
    section order + system prompt survive — the manifest never persisted more.
    """
    import json

    path = os.path.join(character_dir, LEGACY_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return None

    memory: dict[str, Any] = {}
    for name, on in (m.get("enabled") or {}).items():
        memory[f"enabled_{name}"] = bool(on)
    for name, k in (m.get("k_sizes") or {}).items():
        memory[f"{name}_k"] = int(k)
    if m.get("extract_interval") is not None:
        memory["extract_interval"] = int(m["extract_interval"])
    if m.get("emotion_baseline"):
        memory["emotion_baseline"] = dict(m["emotion_baseline"])

    prompts: dict[str, Any] = {}
    if m.get("system_prompt"):
        prompts["system"] = m["system_prompt"]
    if m.get("section_order"):
        prompts["section_order"] = list(m["section_order"])
    for name, header in (m.get("section_headers") or {}).items():
        if header:
            prompts[f"{name}_header"] = header

    out: dict[str, Any] = {}
    if m.get("name"):
        out["name"] = m["name"]
    if m.get("persona"):
        out["persona"] = m["persona"]
    if memory:
        out["memory"] = memory
    if prompts:
        out["prompts"] = prompts
    return out or None


# --------------------------------------------------------------------------- #
# Public load / save.
# --------------------------------------------------------------------------- #
@dataclass
class LoadedCharacterConfig:
    """Result of loading a ``config.yaml``.

    Bundles the four pieces the agent needs to wire itself up. ``persona`` and
    ``name`` default to empty so callers can detect "not specified".
    ``aliases`` is the hand-authored list from the YAML's top-level
    ``aliases:`` key, supplemented by a best-effort scan of ``persona``;
    always includes ``name``. Used by the knowledge-graph self-dedup so the
    character is one node under every name they go by.
    """

    config: CharacterMemoryConfig
    prompts: PromptConfig
    persona: str = ""
    name: str = ""
    aliases: list[str] = field(default_factory=list)


def load_config(path_or_dir: Union[str, os.PathLike]) -> LoadedCharacterConfig:
    """Read a ``config.yaml`` (or, as back-compat, a legacy manifest).

    Accepts either a path to the YAML file or the character directory. Returns
    a :class:`LoadedCharacterConfig`. When the YAML is absent but a legacy
    ``character.json`` exists, it is read instead (one-way migration).
    """
    path = _resolve_path(path_or_dir)
    data: dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, yaml.YAMLError):
            data = {}
    elif os.path.isdir(os.path.dirname(path)) and not os.path.isfile(path):
        # No config.yaml — try legacy manifest migration.
        char_dir = os.path.dirname(path) or "."
        migrated = _migrate_legacy_manifest(char_dir)
        if migrated is not None:
            data = migrated

    full, prompts = build_full_config(data)
    persona = str(data.get("persona") or "")
    name = str(data.get("name") or "")
    # Hand-authored aliases (YAML top-level `aliases:`) ∪ name ∪ aliases
    # inferred from the persona. Order: name first, then declared, then
    # scanned; de-duplicated case-insensitively.
    raw_aliases = data.get("aliases") or []
    if isinstance(raw_aliases, str):
        raw_aliases = [a.strip() for a in raw_aliases.split(",") if a.strip()]
    declared = [str(a) for a in raw_aliases if str(a or "").strip()]
    scanned = _extract_aliases_from_persona(persona, name)
    merged: list[str] = []
    seen: set[str] = set()
    for a in [name, *declared, *scanned]:
        if not a:
            continue
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(a)
    return LoadedCharacterConfig(
        config=full, prompts=prompts, persona=persona, name=name, aliases=merged,
    )


def load_from_character_dir(character_dir: str) -> LoadedCharacterConfig:
    """Convenience: load ``<character_dir>/config.yaml`` (or legacy manifest)."""
    return load_config(character_dir)


def _config_to_dict(config: CharacterMemoryConfig) -> dict[str, Any]:
    """Serialise a full config to the YAML schema dict shape."""
    from dataclasses import asdict

    return {
        "llm": asdict(config.llm),
        "embedding": asdict(config.embedding),
        "chunking": asdict(config.chunking),
        "memory": asdict(config.memory),
    }


def _prompts_to_dict(prompts: PromptConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(prompts)


def save_config(
    path_or_dir: Union[str, os.PathLike],
    *,
    config: CharacterMemoryConfig,
    prompts: PromptConfig,
    persona: str = "",
    name: str = "",
    aliases: Optional[list[str]] = None,
) -> str:
    """Write a ``config.yaml`` and return its path.

    Accepts a file path or a directory (in which case ``config.yaml`` is used).
    ``aliases`` is written to the top-level ``aliases:`` key when provided so
    the round-trip preserves the character's alternate names.
    """
    path = _resolve_path(path_or_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    doc: dict[str, Any] = {}
    if name:
        doc["name"] = name
    if aliases:
        doc["aliases"] = list(aliases)
    if persona:
        doc["persona"] = persona
    doc.update(_config_to_dict(config))
    doc["prompts"] = _prompts_to_dict(prompts)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)
    return path


def default_config_yaml(name: str = "", persona: str = "", aliases: Optional[list[str]] = None) -> str:
    """Return a fresh, fully-populated ``config.yaml`` text for a new character.

    Every section is filled with dataclass defaults and inline comments mark
    the optional knobs, so the file is a readable starting point.
    """
    full = CharacterMemoryConfig()
    prompts = PromptConfig()
    # Serialise defaults to the YAML schema, then prefix a friendly header.
    doc: dict[str, Any] = {}
    if name:
        doc["name"] = name
    if aliases:
        doc["aliases"] = list(aliases)
    if persona:
        doc["persona"] = persona
    doc.update(_config_to_dict(full))
    doc["prompts"] = _prompts_to_dict(prompts)

    header = (
        "# Character configuration. Every section is optional — delete a key\n"
        "# to fall back to its library default. Edit freely; the agent reloads\n"
        "# this file on `CharacterAgent.load_from_config(path)`.\n"
        "#\n"
        "# `aliases` lists every other name the character goes by (nicknames,\n"
        "# alternate spellings, full name). It is used by the knowledge-graph\n"
        "# self-dedup so the character is one node under every name. The\n"
        "# persona is also scanned for nicknames, so this list is optional.\n"
        "#\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)
    return header + body
