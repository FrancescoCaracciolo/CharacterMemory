
"""Configuration objects for the whole character-memory library."""

from dataclasses import dataclass, field
from typing import Optional
import os

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
    dialogue_turns_per_chunk: int = 6
    dialogue_context_width: int = 3       # preceding turns carried as context

