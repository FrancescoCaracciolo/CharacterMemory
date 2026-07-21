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
    ContradictionPolicy,
    DedupConfig,
    EmbeddingConfig,
    KnowledgeGraphConfig,
    LLMConfig,
    MemoryConfig,
)
from .llm import (
    EmbeddingProvider,
    LLMClient,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
)
from .llm.base import LLMResponse
from .memory import (
    CharacterInfoMemory,
    DialogueStyleMemory,
    DedupReport,
    Deduplicator,
    EmotionStatus,
    EpisodicMemory,
    ExtractionContext,
    ExtractionSpec,
    Extractor,
    HeartbeatJournal,
    KnowledgeGraphMemory,
    Memory,
    MemoryItem,
    MemoryScope,
    SQLiteStore,
    StructuredMemory,
    UserDirectiveMemory,
    UserFactMemory,
)
from .knowledge_graph import (
    KnowledgeGraphRetriever,
    KnowledgeGraphRetrivier,
)
from .manifest import MEMORY_NAMES, CharacterManifest
from .prompts import PromptConfig
from .rag import Hit, HybridSearch, RAGSystem
from .tools import (
    TextChunk,
    Tool,
    ToolCall,
    ToolCallEvent,
    ToolRegistry,
    ToolResult,
    ToolResultEvent,
    get_tool,
    global_registry,
    memory_tools,
    register_tool,
    tool,
)

__version__ = "0.1.0"

__all__ = [
    "CharacterAgent",
    "Character",
    "Chat",
    "CharacterMemoryConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "ContradictionPolicy",
    "KnowledgeGraphConfig",
    "MemoryConfig",
    "DedupConfig",
    "PromptConfig",
    "CharacterManifest",
    "MEMORY_NAMES",
    "LLMClient",
    "LLMResponse",
    "EmbeddingProvider",
    "OpenAICompatibleLLM",
    "OpenAICompatibleEmbeddings",
    # Tool calling
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "register_tool",
    "get_tool",
    "global_registry",
    "memory_tools",
    "TextChunk",
    "ToolCallEvent",
    "ToolResultEvent",
    "RAGSystem",
    "HybridSearch",
    "Hit",
    "Chunker",
    "Chunk",
    "get_chunker",
    "register_chunker",
    "Memory",
    "MemoryItem",
    "MemoryScope",
    "ExtractionSpec",
    "StructuredMemory",
    "CharacterInfoMemory",
    "DialogueStyleMemory",
    "UserFactMemory",
    "UserDirectiveMemory",
    "EpisodicMemory",
    "EmotionStatus",
    "HeartbeatJournal",
    "UserSummaryMemory",
    "KnowledgeGraphMemory",
    "KnowledgeGraphRetriever",
    "KnowledgeGraphRetrivier",
    "SQLiteStore",
    "Extractor",
    "ExtractionContext",
    "Deduplicator",
    "DedupReport",
]

