"""character_memory: a modular memory system for role-play characters."""

from .config import (
    ChunkingConfig,
    LLMConfig,
)
from .llm import (
    LLMClient,
)

__version__ = "0.1.0"

__all__ = [
    "LLMConfig",
    "ChunkingConfig",
    "LLMClient",
    "Chunker",
    "Chunk",
    "get_chunker",
    "register_chunker",
]

from .chunking import Chunk, Chunker, get_chunker, register_chunker 
