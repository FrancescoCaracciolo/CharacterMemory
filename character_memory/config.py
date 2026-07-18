
"""Configuration objects for the whole character-memory library."""

from dataclasses import dataclass, field
from typing import Optional
import os


def _load_dotenv() -> None:
    """Minimal, dependency-free `.env` loader.

    Reads `KEY=VALUE` lines from a `.env` file at the project root (the parent
    of this package) and exports them via :func:`os.environ.setdefault`, so
    values already present in the real environment win. Runs at import time,
    before the config dataclasses evaluate their `os.getenv` defaults.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.getcwd(), os.path.dirname(here)):
        path = os.path.join(root, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        break


_load_dotenv()

@dataclass
class LLMConfig:
    """Settings for a chat-completions client."""

    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 120.0


@dataclass
class EmbeddingConfig:
    """Settings for an embedding client."""

    base_url: str = os.getenv("OPENAI_EMBEDDINGS_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002")
    dim: Optional[int] = None  # inferred from the first request if None
    batch_size: int = 64
    timeout: float = 120.0


@dataclass
class ChunkingConfig:
    """Chunker selection + parameters used at index time."""

    info_chunker: str = "header"          # name in the chunker registry
    dialogue_chunker: str = "dialogue"
    header_max_tokens: int = 512
    header_min_tokens: int = 64  # fragments below this are merged, not emitted
    dialogue_turns_per_chunk: int = 6
    dialogue_context_width: int = 3       # preceding turns carried as context


@dataclass
class DedupConfig:
    """Deduplication behaviour for structured memories.

    A memory entry is considered a duplicate when it passes every *enabled*
    gate, evaluated in escalating order: exact match → similarity → LLM judge.
    If `consolidate` is on, a confirmed duplicate is merged into one entry
    instead of being dropped.

    - `enabled`: master switch. When False, memories fall back to the legacy
      exact-only guard.
    - `exact`: case-insensitive, stripped string equality (cheapest gate).
    - `similarity_threshold`: cosine similarity above which two entries are
      considered candidate duplicates. `None` disables the similarity gate.
    - `llm_judge`: when True, an LLM confirms that a similarity candidate is
      genuinely the same information. Needs an `LLMClient`.
    - `consolidate`: when True, confirmed duplicates are rewritten into a
      single merged entry instead of the newer one being dropped. Needs an
      `LLMClient`.
    - `per_user`: during a sweep, only compare entries sharing a `user_id`.
    - `candidate_pool`: per-item guard — how many RAG hits to re-rank.
    """

    enabled: bool = False
    exact: bool = True
    similarity_threshold: Optional[float] = 0.92
    llm_judge: bool = False
    consolidate: bool = False
    per_user: bool = True
    candidate_pool: int = 10


@dataclass
class ContradictionPolicy:
    """Per-memory policy for resolving contradicting facts.

    Returned by :meth:`StructuredMemory.contradiction_policy`. When enabled,
    the Deduplicator runs an extra gate — after dedup finds no duplicate —
    that surfaces semantically close rows which *clash* and overwrites the
    older row's text with the newer one's. Disabled by default; stable-fact
    memories override it.

    - `enabled`: master switch for this memory.
    - `similarity_threshold`: cosine bar below the dedup threshold; catches
      pairs that clash but are not restatements (e.g. "doctor" vs "engineer").
    - `candidate_pool`: wider net than dedup's own pool, since contradictions
      sit at lower similarity.
    - `show_timestamps`: pass each row's `created_at` to the contradiction
      judge so it can tell a genuine clash from a change over time.
    """

    enabled: bool = False
    similarity_threshold: float = 0.70
    candidate_pool: int = 20
    show_timestamps: bool = True


@dataclass
class MemoryConfig:
    """Per-memory toggles and retrieval knobs.

    Every memory can be enabled/disabled independently via `enabled_*`.
    """

    # Toggles
    enabled_character_info: bool = True
    enabled_dialogue_style: bool = True
    enabled_user_facts: bool = True
    enabled_user_directives: bool = True
    enabled_episodic: bool = True
    enabled_emotion: bool = True
    enabled_heartbeat: bool = True
    enabled_user_summary: bool = True

    # Retrivial Sizes
    character_info_k: int = 4
    dialogue_style_k: int = 4
    user_facts_k: int = 5
    user_directives_k: int = 4
    episodic_k: int = 4
    heartbeat_k: int = 4
    user_summary_k: int = 2

    # Structured Memory behavior
    # Facts/directives whose effective importance is at/above this value are
    # always injected into the prompt ("sticky"), regardless of the query.
    sticky_threshold: float = 0.95
    # Extraction of facts/directives/episodes runs every N turns.
    extract_interval: int = 5
    # Decay half-life (seconds). Used by facts/episodic/heartbeat.
    decay_half_life: float = 60 * 60 * 24 * 3 # Three days

    # EMOTIONS
    # Baseline (user-independent) emotion vector.
    emotion_baseline: dict = field(
        default_factory=lambda: {
            "neutral": 0.5, "joy": 0.2, "sadness": 0.1, "anxiety": 0,
            "anger": 0.1, "surprise": 0.1,
        }
    )
    # Per-user dimensions maintained alongside the baseline.
    emotion_user_dims: dict = field(
        default_factory=lambda: {"affection": 0.0, "valence": 0.0, "trust": 0.0}
    )

    # DEDUPLICATION
    dedup: DedupConfig = field(default_factory=DedupConfig)

    def is_enabled(self, name: str) -> bool:
        return bool(getattr(self, f"enabled_{name}", False))

    def k_for(self, name: str) -> int:
        return int(getattr(self, f"{name}_k", 4))


@dataclass
class CharacterMemoryConfig:
    """Top-level config: everything needed to build the whole system."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Directory used for the SQLite store + persisted RAG indexes.
    data_dir: str = ".cm_data"

    @classmethod
    def default(cls) -> "CharacterMemoryConfig":
        """A config pointing at a local OpenAI-compatible server."""
        return cls()
