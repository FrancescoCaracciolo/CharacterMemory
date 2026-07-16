"""character_memory, a modular memory system for role-play characters.

Public surface:

* `CharacterMemoryConfig` / sub-configs — configure models, toggles, decay.
* `CharacterAgent` - the orchestrator (recall -> prompt -> LLM -> learn).
* Memory classes, the RAG system, chunkers, and the LLM/embedding abstractions
  are all re-exported for direct use or subclassing.
"""

from .agent import CharacterAgent
from .character import Character
from .chat import Chat
from .chunking import (
    Chunk,
    Chunker,
    get_chunker,
    register_chunker,
)
from .config import (
    CharacterMemoryConfig,
    ChunkingConfig,
    DedupConfig,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
)
from .llm import (
    EmbeddingProvider,
    LLMClient,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
)
from .memory import (
    CharacterInfoMemory,
    DialogueStyleMemory,
    DedupReport,
    Deduplicator,
    EmotionStatus,
    EpisodicMemory,
    Extractor,
    HeartbeatJournal,
    Memory,
    MemoryItem,
    SQLiteStore,
    StructuredMemory,
    UserDirectiveMemory,
    UserFactMemory,
)
from .prompts import PromptConfig
from .rag import Hit, HybridSearch, RAGSystem

__version__ = "0.1.0"

__all__ = [
    "CharacterAgent",
    "Character",
    "Chat",
    "CharacterMemoryConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "MemoryConfig",
    "DedupConfig",
    "PromptConfig",
    "LLMClient",
    "EmbeddingProvider",
    "OpenAICompatibleLLM",
    "OpenAICompatibleEmbeddings",
    "RAGSystem",
    "HybridSearch",
    "Hit",
    "Chunker",
    "Chunk",
    "get_chunker",
    "register_chunker",
    "Memory",
    "MemoryItem",
    "StructuredMemory",
    "CharacterInfoMemory",
    "DialogueStyleMemory",
    "UserFactMemory",
    "UserDirectiveMemory",
    "EpisodicMemory",
    "EmotionStatus",
    "HeartbeatJournal",
    "SQLiteStore",
    "Extractor",
    "Deduplicator",
    "DedupReport",
]

