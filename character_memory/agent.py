"""The `CharacterAgent` ties everything together.

Lifecycle (lazy, explicit):

1. `CharacterAgent(directory=..., name=..., save_directory=...)` - cheap
   constructor; stores paths and templates, opens nothing.
2. `load_from_config(...)` or `load(llm, embedder, memories)` - wires the
   backends, opens the shared SQLite store, builds the standard memories (or
   accepts a caller-supplied list), and prepares the chat tables.
3. `build()` / `rebuild()` - load-or-build / force-rebuild the RAG indexes
   over the character's `Information/` and `Dialogues/` material.

After that:

* `create_chat(user)` / `load_chat(id)` manage persistent conversations
  (many chats per user; one row per turn in `messages`).
* `generate_answer(chat_or_messages, stream=, save=)` produces a reply -
  a string, or an iterator of chunks when `stream=True`. When given a
  `Chat` with `save=True` the assistant turn is persisted and, on the
  configured `extract_interval`, learning runs automatically.
* `extract(chat=None)` runs memory extraction over any messages not yet
  processed - idempotent and resumable thanks to the `extracted` flag.

Prompt assembly and per-memory extraction are delegated to a
`Character` class.
"""

import os
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from .chat import Chat, _ChatBackend
from .chunking.registry import get_chunker
from .character import Character, ContextSnapshot
from .config import (
    CharacterMemoryConfig,
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
)
from .llm.base import LLMClient
from .llm.embedding_base import EmbeddingProvider
from .memory.base import Memory
from .memory.character_base import CharacterInfoMemory, DialogueStyleMemory
from .memory.conversation_events import ConversationEventMemory
from .memory.dedup import Deduplicator, DedupReport
from .memory.emotion import EmotionStatus
from .memory.episodic import EpisodicMemory
from .memory.heartbeat import HeartbeatJournal
from .memory.knowledge_graph_memory import KnowledgeGraphMemory
from .memory.store import SQLiteStore
from .memory.structured import StructuredMemory
from .memory.user_directives import UserDirectiveMemory
from .memory.user_facts import UserFactMemory
from .memory.user_summary import UserSummaryMemory
from .knowledge_graph.nodes import Node
from .character_config import load_from_character_dir
from .prompts import PromptConfig
from .rag.base import Query
from .rag.hybrid import HybridSearch
from .tools.base import (
    TextChunk,
    Tool,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from .tools.memory_tools import memory_tools as _build_memory_tools
from .tools.registry import ToolRegistry

_INFO_GLOB = "Information"
_DIALOGUE_GLOB = "Dialogues"

# Names of the two RAG memories populated from the character directory and of
# the structured memories whose hybrid index is rebuilt from SQLite rows.
_STRUCTURED_MEMORIES = (
    "user_facts",
    "user_directives",
    "episodic",
    "conversation_events",
    "heartbeat",
    "user_summary",
)
_DEDUP_MEMORIES = (
    "user_facts", "user_directives", "episodic", "heartbeat", "user_summary"
)
# The knowledge-graph memory has its own persistence layout (kg_index/) and a
# load-or-ingest lifecycle driven by the other memories.
_KG_MEMORY = "knowledge_graph"

# Anything generate_answer / build_context / render_prompt accepts as a
# conversation target.
Target = Union[Chat, str, list[dict[str, str]]]

# What `generate_answer(tools=...)` accepts: a registry, a single Tool, or a
# list of tools/callables (callables are wrapped via the @tool decorator).
ToolsArg = Union[ToolRegistry, Tool, Iterable]


def _as_registry(tools: ToolsArg) -> ToolRegistry:
    """Normalise the ``tools`` argument into a :class:`ToolRegistry`.

    A registry passes through; a single Tool / callable wraps in a fresh
    registry; an iterable builds one. Callables are auto-decorated, so you can
    hand ``generate_answer`` a list of plain functions.
    """
    if isinstance(tools, ToolRegistry):
        return tools
    if isinstance(tools, Tool) or callable(tools):
        return ToolRegistry([tools])
    return ToolRegistry(tools)


def _json_dumps(obj: Any) -> str:
    """``json.dumps`` with ``ensure_ascii=False`` (used for tool-call args)."""
    import json

    return json.dumps(obj, ensure_ascii=False)


class CharacterAgent:
    """Orchestrates memories, prompts, the LLM, persistent chats and extraction."""

    def __init__(
        self,
        directory: str,
        *,
        name: Optional[str] = None,
        save_directory: Optional[str] = None,
        prompt_config: Optional[PromptConfig] = None,
        persona: str = "",
    ) -> None:
        self.character_dir = directory
        self.character_name = name or os.path.basename(os.path.normpath(directory))
        self.save_directory = save_directory or os.path.join(directory, ".cm_data")
        self.prompts = prompt_config or PromptConfig()
        # Whether `prompt_config` was explicitly passed. Used by
        # `load_from_config(path)` to decide whether the file's prompts should
        # win (constructor arg wins when explicitly provided).
        self._prompts_explicit = prompt_config is not None
        # Short character blurb surfaced to the extractor (and the answer prompt)
        # so the model knows who the character is. Auto-summarizing the
        # Information/*.md into a blurb is intentionally out of scope; callers
        # pass this in if they want the persona clause populated. May also be
        # loaded from `config.yaml` via `load_from_config(path)`.
        self.persona = persona
        # Alternate names the character goes by (nicknames, full name, …).
        # Used by the knowledge-graph self-dedup so the character is a single
        # node under every name. Populated from `config.yaml` (top-level
        # `aliases:` ∪ persona-scanned) in `load_from_config`.
        self.character_aliases: list[str] = []
        # Path of the config.yaml last loaded, when `load_from_config(path)`
        # was used. None until then.
        self.config_path: Optional[str] = None

        # Not wired until load_* is called.
        self.config: Optional[CharacterMemoryConfig] = None
        self.llm: Optional[LLMClient] = None
        self.embedder: Optional[EmbeddingProvider] = None
        self.store: Optional[SQLiteStore] = None
        self.memories: dict[str, Memory] = {}
        self.character: Optional[Character] = None
        self.deduplicator: Optional[Deduplicator] = None
        # Per-memory recall bounds: top-k counts, except KG prompt tokens.
        self._limits: dict[str, int] = {}
        self._chats: Optional[_ChatBackend] = None
        self._built = False

    # Require loaded state before executing some actions
    def _require_loaded(self) -> None:
        if self.character is None or self.llm is None or self.store is None:
            raise RuntimeError(
                "CharacterAgent is not loaded. Call load_from_config(...) or "
                "load(llm, embedder, memories) before using it."
            )

    # Loading
    def load_from_config(
        self,
        config_or_path: Union[str, os.PathLike, "CharacterMemoryConfig", None] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingProvider] = None,
        config: Optional[CharacterMemoryConfig] = None,
    ) -> "CharacterAgent":
        """Wire backends from config objects or a ``config.yaml`` path.

        Three call styles, all valid:

        * ``load_from_config("path/to/config.yaml")`` — load a persisted
          per-character YAML (persona, prompts, every sub-config). The
          file is the single source of truth for the character.
        * ``load_from_config(CharacterMemoryConfig(...))`` — pass a full
          in-memory config object as the first positional arg.
        * ``load_from_config(llm_config, embedding_config, memory_config,
          chunking_config)`` — the original positional-subconfig style.

        ``llm`` / ``embedder`` let you override the auto-built OpenAI clients.
        Constructor persona/prompt args always win; the YAML only fills what
        the caller did not set explicitly.
        """
        # Case 1: a path to config.yaml. Load persona + prompts + full config.
        full: CharacterMemoryConfig
        if isinstance(config_or_path, (str, os.PathLike)):
            loaded = load_from_character_dir(os.fspath(config_or_path))
            self.config_path = os.fspath(config_or_path)
            full = loaded.config
            # Constructor args win over the file. `self.persona` was set in
            # __init__ from the persona=... arg; only adopt the file's when the
            # caller did not pass one. `self.prompts` likewise — if the caller
            # passed prompt_config, __init__ already stored it; otherwise adopt
            # the file's.
            if not self.persona and loaded.persona:
                self.persona = loaded.persona
            # Adopt the file's aliases unless the caller passed some via the
            # constructor. The file's list already includes name + persona
            # scans, so it is the richest source.
            if not self.character_aliases and loaded.aliases:
                self.character_aliases = list(loaded.aliases)
            if self._prompts_explicit is not True:
                self.prompts = loaded.prompts
        elif isinstance(config_or_path, CharacterMemoryConfig):
            full = config_or_path
        elif config is not None:
            full = config
        else:
            # config_or_path is the legacy llm_config positional.
            llm_config = config_or_path if isinstance(config_or_path, LLMConfig) else None
            full = CharacterMemoryConfig(
                llm=llm_config or LLMConfig(),
                embedding=embedding_config or EmbeddingConfig(),
                memory=memory_config or MemoryConfig(),
                chunking=chunking_config or ChunkingConfig(),
            )
        self.config = full

        # Backends (pluggable; fall back to the OpenAI-compatible references).
        from .llm.openai_client import OpenAICompatibleLLM
        from .llm.openai_embeddings import OpenAICompatibleEmbeddings

        self.llm = llm or OpenAICompatibleLLM(full.llm)
        self.embedder = embedder or OpenAICompatibleEmbeddings(full.embedding)

        self._open_store()
        self._build_memories(full.memory)
        self._wire_character()
        self._chats = _ChatBackend(self.store)
        return self

    def load(
        self,
        llm: LLMClient,
        embedder: EmbeddingProvider,
        memories: list[Memory],
    ) -> "CharacterAgent":
        """DIY path: supply ready backends + a list of memories.

        The agent still opens its own store at `<save_directory>/memory.db`
        for chat persistence; the caller's memories keep whatever stores they
        already own. No config is recorded, so `build()` / `rebuild()`
        fall back to `ChunkingConfig` defaults.
        """
        self.config = None
        self.llm = llm
        self.embedder = embedder
        self._open_store()
        self.memories = {m.name: m for m in memories}
        self._wire_character()
        self._chats = _ChatBackend(self.store)
        return self

    def _open_store(self) -> None:
        os.makedirs(self.save_directory, exist_ok=True)
        self.store = SQLiteStore(os.path.join(self.save_directory, "memory.db"))

    def _build_memories(self, m: MemoryConfig) -> None:
        """Construct the seven standard memories (config-driven path)."""
        half = m.decay_half_life
        sticky = m.sticky_threshold

        def hybrid() -> HybridSearch:
            return HybridSearch(self.embedder)  # type: ignore[arg-type]

        # A shared deduplicator, built when dedup is enabled. It is *not*
        # injected into memories — it runs as a post-extraction step on the
        # items each memory reports having added. Its LLM prompts come from the
        # agent's PromptConfig so they are overridable like every other prompt.
        self.deduplicator: Optional[Deduplicator] = (
            Deduplicator(self.embedder, self.llm, m.dedup, prompts=self.prompts)
            if m.dedup.enabled
            else None
        )

        self.memories["character_info"] = CharacterInfoMemory(
            hybrid(), enabled=m.is_enabled("character_info")
        )
        self.memories["dialogue_style"] = DialogueStyleMemory(
            hybrid(), enabled=m.is_enabled("dialogue_style")
        )
        self.memories["user_facts"] = UserFactMemory(
            self.store, hybrid(), enabled=m.is_enabled("user_facts"),
            half_life=half, sticky_threshold=sticky,
        )
        self.memories["user_directives"] = UserDirectiveMemory(
            self.store, hybrid(), enabled=m.is_enabled("user_directives"),
            half_life=half, sticky_threshold=sticky,
        )
        emotion = EmotionStatus(
            self.store, enabled=m.is_enabled("emotion"),
            baseline=m.emotion_baseline, user_dims=m.emotion_user_dims,
        )
        self.memories["episodic"] = EpisodicMemory(
            self.store, hybrid(), enabled=m.is_enabled("episodic"),
            half_life=half, sticky_threshold=sticky,
            emotion_baseline=m.emotion_baseline,
            current_mood=(emotion.get_current_mood if emotion.enabled else lambda: {}),
        )
        self.memories["conversation_events"] = ConversationEventMemory(
            self.store,
            hybrid(),
            enabled=m.is_enabled("conversation_events"),
            half_life=half,
            sticky_threshold=sticky,
        )
        self.memories["heartbeat"] = HeartbeatJournal(
            self.store, hybrid(), enabled=m.is_enabled("heartbeat"),
            half_life=half, sticky_threshold=sticky,
        )
        self.memories["emotion"] = emotion
        self.memories["user_summary"] = UserSummaryMemory(
            self.store, hybrid(), enabled=m.is_enabled("user_summary"),
            half_life=half, sticky_threshold=sticky,
        )

        # Knowledge graph — optional, additive retriever built last so it can
        # see every other memory as a source. Wired (LLM/embedder/sources)
        # after construction in :meth:`_wire_knowledge_graph`.
        if m.is_enabled("knowledge_graph"):
            self.memories[_KG_MEMORY] = KnowledgeGraphMemory(
                self.store, hybrid(),
                enabled=m.is_enabled("knowledge_graph"),
                config=m.knowledge_graph,
                token_budget=m.knowledge_graph_token_budget,
            )

    def _wire_character(self) -> None:
        self._limits = {
            name: (
                self.config.memory.retrieval_limit_for(name)
                if self.config
                else (1_000 if name == _KG_MEMORY else 4)
            )
            for name in self.memories
        }
        self.character = Character(
            character_name=self.character_name,
            base_instruction=self.persona,
            memories=list(self.memories.values()),
            llm=self.llm,
            prompts=self.prompts,
        )
        # Wire the knowledge graph's backends + source-memory back-reference
        # so it can ingest/update/apply-deduplication against the others.
        kg = self.memories.get(_KG_MEMORY)
        if isinstance(kg, KnowledgeGraphMemory):
            kg.attach_backends(self.llm, self.embedder)
            kg.wire_sources(self.memories)
            # Pass the character identity (name + persona + aliases) to the
            # retriever so entity/people extraction is context-aware,
            # relevance-filtered, and self-dedup knows every name the
            # character goes by. Always include the canonical name in the
            # alias set so the self-collapse is robust even with no aliases.
            aliases = list(self.character_aliases or [])
            if self.character_name and self.character_name not in aliases:
                aliases = [self.character_name, *aliases]
            kg.retriever.character = {
                "name": self.character_name,
                "persona": (self.persona or "").strip(),
                "aliases": aliases,
            }

    # Indexing
    def _chunking_config(self) -> ChunkingConfig:
        return self.config.chunking if self.config is not None else ChunkingConfig()

    def _index_character(self) -> tuple[int, int]:
        """Chunk + build + persist the wiki & dialogue indexes."""
        assert self.store is not None
        c = self._chunking_config()
        info_dir = os.path.join(self.character_dir, _INFO_GLOB)
        info_chunks = (
            get_chunker(
                c.info_chunker,
                max_tokens=c.header_max_tokens,
                min_tokens=c.header_min_tokens,
            ).chunk_directory(info_dir)
            if os.path.isdir(info_dir) else []
        )
        dlg_dir = os.path.join(self.character_dir, _DIALOGUE_GLOB)
        dlg_chunks = (
            get_chunker(
                c.dialogue_chunker,
                turns_per_chunk=c.dialogue_turns_per_chunk,
                context_width=c.dialogue_context_width,
            ).chunk_directory(dlg_dir)
            if os.path.isdir(dlg_dir) else []
        )
        info_mem = self.memories.get("character_info")
        if isinstance(info_mem, CharacterInfoMemory) and info_chunks:
            info_mem.build(info_chunks)
            info_mem.persist(os.path.join(self.save_directory, "info_index"))
        dlg_mem = self.memories.get("dialogue_style")
        if isinstance(dlg_mem, DialogueStyleMemory) and dlg_chunks:
            dlg_mem.build(dlg_chunks)
            dlg_mem.persist(os.path.join(self.save_directory, "dialogue_index"))
        return len(info_chunks), len(dlg_chunks)

    def _has_index(self, subdir: str) -> bool:
        return os.path.exists(os.path.join(self.save_directory, subdir, "nodes.json"))

    def _wiki_sections(self) -> list[dict[str, Any]]:
        """Header-chunk the character's `Information/*.md` into wiki sections.

        Each chunk becomes one dict ``{text, header, source}`` fed to the KG's
        structural wiki ingest (no LLM). Returns ``[]`` when there is no
        Information directory.
        """
        c = self._chunking_config()
        info_dir = os.path.join(self.character_dir, _INFO_GLOB)
        if not os.path.isdir(info_dir):
            return []
        chunks = get_chunker(
            c.info_chunker,
            max_tokens=c.header_max_tokens,
            min_tokens=c.header_min_tokens,
        ).chunk_directory(info_dir)
        out: list[dict[str, Any]] = []
        for ch in chunks:
            out.append(
                {
                    "text": ch.text,
                    "header": (ch.metadata or {}).get("header", ""),
                    "source": ch.source or "wiki",
                }
            )
        return out

    def _ingest_knowledge_graph(
        self,
        kg: KnowledgeGraphMemory,
        *,
        reset: bool,
        on_node_added: Optional[Callable[[Node], None]],
        on_llm_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Build and persist the KG through the shared monitored ingest path."""
        if reset:
            kg.retriever.reset()
        kg.retriever._ingest_for_build(
            list(self.memories.values()),
            self._wiki_sections(),
            on_node_added=on_node_added,
            on_llm_progress=on_llm_progress,
        )
        kg.persist(os.path.join(self.save_directory, "kg_index"))

    def build(
        self,
        *,
        on_node_added: Optional[Callable[[Node], None]] = None,
        on_llm_progress: Optional[Callable[[int, int], None]] = None,
    ) -> "CharacterAgent":
        """Load persisted indexes if present, otherwise build + persist them.

        During a fresh knowledge-graph build, ``on_node_added(node)`` is called
        after each real node insertion and ``on_llm_progress(completed, total)``
        is called initially and after every structured LLM request. Loading an
        already-persisted graph reports ``(0, 0)`` and adds no nodes.
        """
        self._require_loaded()
        assert self.store is not None
        # RAG memories: load if persisted, else index from the character dir.
        info_ok = self._has_index("info_index")
        dlg_ok = self._has_index("dialogue_index")
        if info_ok:
            mem = self.memories.get("character_info")
            if isinstance(mem, CharacterInfoMemory):
                mem.load(os.path.join(self.save_directory, "info_index"))
        if dlg_ok:
            mem = self.memories.get("dialogue_style")
            if isinstance(mem, DialogueStyleMemory):
                mem.load(os.path.join(self.save_directory, "dialogue_index"))
        if not (info_ok and dlg_ok):
            self._index_character()

        # Structured-memory indexes: load persisted, else rebuild from SQLite.
        for name in _STRUCTURED_MEMORIES:
            mem = self.memories.get(name)
            if mem is None:
                continue
            path = os.path.join(self.save_directory, f"{name}_index")
            if self._has_index(name + "_index"):
                mem.load(path)
            else:
                mem.rebuild_index()
                mem.persist(path)

        # Knowledge graph: load if persisted, otherwise build + persist. Runs
        # after the structured indexes so a fresh build reads the already-loaded
        # rows. The graph never writes back to its sources (modules stay
        # independent). A persisted graph is loaded verbatim here -- to rebuild
        # an existing one use rebuild_knowledge_graph() (kg-only) or rebuild()
        # (everything); neither runs from build().
        kg = self.memories.get(_KG_MEMORY)
        if isinstance(kg, KnowledgeGraphMemory):
            kg_path = os.path.join(self.save_directory, "kg_index")
            if kg.retriever.has_persisted(kg_path):
                kg.load(kg_path)
                if on_llm_progress is not None:
                    on_llm_progress(0, 0)
            else:
                # Wiki is folded into the graph as typed nodes (characters /
                # entities / episodes) via an LLM pass. Expensive, so it only
                # runs on a fresh build; re-running needs an explicit rebuild.
                self._ingest_knowledge_graph(
                    kg,
                    reset=False,
                    on_node_added=on_node_added,
                    on_llm_progress=on_llm_progress,
                )
        self._built = True
        return self

    def rebuild(
        self,
        *,
        on_node_added: Optional[Callable[[Node], None]] = None,
        on_llm_progress: Optional[Callable[[int, int], None]] = None,
    ) -> "CharacterAgent":
        """Force a full re-chunk + re-index, overwriting persisted indexes.

        The optional callbacks monitor the knowledge-graph phase only; they use
        the same contract as :meth:`rebuild_knowledge_graph`.
        """
        self._require_loaded()
        assert self.store is not None
        self._index_character()
        for name in _STRUCTURED_MEMORIES:
            mem = self.memories.get(name)
            if mem is None:
                continue
            mem.rebuild_index()
            mem.persist(os.path.join(self.save_directory, f"{name}_index"))
        # Knowledge graph: rebuild from scratch and re-ingest.
        kg = self.memories.get(_KG_MEMORY)
        if isinstance(kg, KnowledgeGraphMemory):
            self._ingest_knowledge_graph(
                kg,
                reset=True,
                on_node_added=on_node_added,
                on_llm_progress=on_llm_progress,
            )
        self._built = True
        return self

    def rebuild_knowledge_graph(
        self,
        *,
        on_node_added: Optional[Callable[[Node], None]] = None,
        on_llm_progress: Optional[Callable[[int, int], None]] = None,
    ) -> "CharacterAgent":
        """Rebuild only the knowledge graph, leaving every other index alone.

        Cheaper and more targeted than :meth:`rebuild` (which re-chunks the
        character corpus and rebuilds the structured memories too). The graph
        is rebuilt from the existing source-memory rows in SQLite and the
        character's wiki sections, then persisted. Use this for a one-off KG
        refresh -- e.g. ``charactermemory-server --rebuild-kg kurisu``.

        ``on_node_added(node)`` runs synchronously after each new node is
        inserted. ``on_llm_progress(completed, total)`` runs once before the
        first structured LLM request and after each attempt; the number still
        remaining is ``total - completed``.
        """
        self._require_loaded()
        assert self.store is not None
        kg = self.memories.get(_KG_MEMORY)
        if not isinstance(kg, KnowledgeGraphMemory):
            raise RuntimeError(
                f"Character {self.character_name!r} does not have the "
                f"knowledge_graph memory enabled; nothing to rebuild."
            )
        self._ingest_knowledge_graph(
            kg,
            reset=True,
            on_node_added=on_node_added,
            on_llm_progress=on_llm_progress,
        )
        return self

    def persist_structured(self) -> None:
        """Persist the structured-memory indexes (call after learning)."""
        for name in _STRUCTURED_MEMORIES:
            mem = self.memories.get(name)
            if mem is None:
                continue
            mem.persist(os.path.join(self.save_directory, f"{name}_index"))
        # The knowledge graph changes whenever its source memories change, so
        # it is persisted alongside them after learning.
        kg = self.memories.get(_KG_MEMORY)
        if isinstance(kg, KnowledgeGraphMemory):
            kg.persist(os.path.join(self.save_directory, "kg_index"))

    # Target resolution
    def _as_chat(self, target: Union[Chat, str]) -> Optional[Chat]:
        """Resolve a `Chat` or chat id to a `Chat` (or `None` if missing)."""
        assert self._chats is not None
        if isinstance(target, Chat):
            return target
        return self._chats.load_chat(target)

    def _resolve_target(
        self, target: Target, user_id: str
    ) -> tuple[str, str, list[dict[str, str]], list[str]]:
        """Return (query, user_id, prior_messages, participants) for a target.

        - `Chat`                 -> (last user msg or "", chat.user_id, history, chat.participants())
        - chat id `str`          -> same, after load_chat
        - raw query `str`        -> (target, user_id, [], [user_id])
        - `list[dict]` messages  -> (last user content or "", user_id, target, [user_id])

        Participants come from a persisted `Chat` (every human speaker in it).
        Raw query / message-list targets have no stored speakers, so they are
        treated as single-user with ``[user_id]``.
        """
        if isinstance(target, Chat):
            chat = target
            return (
                chat.last_user_message() or "",
                chat.user_id,
                chat.messages(),
                chat.participants(),
            )
        if isinstance(target, str):
            chat = self._as_chat(target)
            if chat is not None:
                return (
                    chat.last_user_message() or "",
                    chat.user_id,
                    chat.messages(),
                    chat.participants(),
                )
            # Bare query string, not a known chat id.
            return (target, user_id, [], [user_id])
        # Message list.
        msgs = list(target)
        last_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user" and m.get("content"):
                last_user = m["content"]
                break
        return (last_user, user_id, msgs, [user_id])

    def _weighted_query(
        self, last_user_msg: str, prior: list[dict[str, str]]
    ) -> Query:
        """Build a history-aware retrieval query from recent user messages.

        Takes the last ``retrieval_history_window`` user-role messages
        (most-recent first) and returns a list of ``(text, weight)`` pairs
        where weight = ``retrieval_recency_decay ** i`` (the current message
        at i=0 has weight 1.0, older ones fade). Returns a bare string when
        history-aware retrieval is disabled (window <= 1) or no prior messages
        exist, so the common path stays a single query.
        """
        window = 1
        decay = 1.0
        if self.config is not None:
            window = max(1, int(getattr(self.config.memory, "retrieval_history_window", 1)))
            decay = float(getattr(self.config.memory, "retrieval_recency_decay", 1.0))
        if window <= 1:
            return last_user_msg
        # Collect user-role contents from `prior`, newest first.
        recent: list[str] = []
        for m in reversed(prior):
            if m.get("role") == "user" and m.get("content"):
                recent.append(m["content"])
            if len(recent) >= window:
                break
        # The current message leads at weight 1.0; drop it from the history
        # tail to avoid double-counting when `prior` already contains it.
        cur = (last_user_msg or "").strip()
        tail = [t for t in recent if t.strip() != cur]
        if not tail:
            # No usable older message: stay on the single-query legacy path.
            return last_user_msg
        if not cur:
            # No explicit current message; lead with the newest from history.
            return [(tail[i], decay ** i) for i in range(min(window, len(tail)))]
        queries: list[tuple[str, float]] = [(last_user_msg, 1.0)]
        for i, text in enumerate(tail[: window - 1]):
            queries.append((text, decay ** (i + 1)))
        return queries

    # Prompt
    def build_context(
        self, target: Target, *, user_id: str = "default"
    ) -> dict[str, str]:
        """Return `{memory_name: rendered_section}` for the target."""
        return self.build_context_snapshot(target, user_id=user_id).sections

    def build_context_snapshot(
        self, target: Target, *, user_id: str = "default"
    ) -> ContextSnapshot:
        """Build context once and expose the exact recalls used to build it."""
        self._require_loaded()
        assert self.character is not None
        last_user_msg, uid, prior, participants = self._resolve_target(target, user_id)
        query = self._weighted_query(last_user_msg, prior)
        return self.character.build_context_snapshot(
            query, uid, limits=self._limits, participants=participants
        )

    def render_prompt(self, target: Target, *, user_id: str = "default") -> str:
        """Full system-style context block (system line + all sections)."""
        self._require_loaded()
        assert self.character is not None
        last_user_msg, uid, prior, participants = self._resolve_target(target, user_id)
        query = self._weighted_query(last_user_msg, prior)
        return self.character.render_prompt(
            query, uid, limits=self._limits, participants=participants
        )

    # Chat management
    def create_chat(self, user: str, *, title: str = "") -> Chat:
        """Create and persist a new chat for `user`."""
        self._require_loaded()
        assert self._chats is not None
        return self._chats.create_chat(user, title=title)

    def load_chat(self, chat_id: str) -> Optional[Chat]:
        """Load any chat by id (across users)."""
        self._require_loaded()
        assert self._chats is not None
        return self._chats.load_chat(chat_id)

    def list_chats(self, user: Optional[str] = None) -> list[Chat]:
        """List chats, optionally filtered by user."""
        self._require_loaded()
        assert self._chats is not None
        return self._chats.list_chats(user)

    # Generation
    def _build_messages(
        self,
        query: Query,
        user_id: str,
        prior: list[dict[str, str]],
        participants: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        assert self.character is not None
        system = self.character.render_prompt(
            query, user_id, limits=self._limits, participants=participants
        )
        return [{"role": "system", "content": system}, *prior]

    def _maybe_auto_extract(self, chat: Chat) -> None:
        """If the chat hit the extract interval, run extraction on it."""
        interval = max(
            1,
            self.config.memory.extract_interval if self.config is not None else 5,
        )
        # Count user turns already persisted.
        user_turns = self.store.select(  # type: ignore[union-attr]
            "messages",
            where={"chat_id": chat.id, "role": "user"},
        )
        if len(user_turns) % interval == 0:
            self._extract_chat(chat)

    def generate_answer(
        self,
        target: Target,
        *,
        stream: bool = False,
        save: bool = True,
        user_id: str = "default",
        tools: Optional[ToolsArg] = None,
        max_tool_iterations: int = 8,
        tool_choice: Optional[Any] = None,
        auto_extract: bool = True,
    ) -> Union[str, Iterator[Any]]:
        """Generate an assistant reply for `target`.

        `target` is a `Chat`, a chat id, or a raw list of openai-style
        messages. When `target` is a `Chat` (or chat id) and `save` is
        true, the latest user message is expected to already be persisted in
        the chat, and the assistant reply is persisted too; auto-extraction
        fires on the configured interval. `stream=True` returns an iterator
        of text chunks (the full reply is still persisted once the stream
        completes when `save` is true).

        Tool calling (opt-in):

        * `tools` — a :class:`~character_memory.tools.ToolRegistry`, a single
          :class:`~character_memory.tools.Tool`, or a list of either. Pass
          :meth:`memory_tools` for the built-in memory self-tools. When
          provided, the agent runs a model → tool → model loop and returns the
          final text reply.
        * `max_tool_iterations` caps how many tool rounds the loop will run
          before forcing a plain-text reply (default 8).
        * `tool_choice` follows the OpenAI convention (``"auto"``, ``"none"``,
          ``"required"``, or a specific tool). ``None`` lets the backend pick.

        Streaming + tools: with ``stream=True`` and ``tools`` set, the return
        value is an iterator of :class:`~character_memory.tools.TextChunk` /
        :class:`~character_memory.tools.ToolCallEvent` /
        :class:`~character_memory.tools.ToolResultEvent` events. Without
        tools, ``stream=True`` keeps the legacy plain-text-chunk behaviour.

        Persistence: only the final assistant **text** reply is written to the
        chat. Intermediate tool-call / tool-result messages stay in memory for
        the loop and are never persisted, so chat history and extraction stay
        clean.

        ``auto_extract=False`` persists the assistant reply but skips the
        automatic extraction pass — useful when the caller wants the reply back
        as soon as it's generated and will run :meth:`extract` / extraction on
        its own (e.g. on a background thread). The reply row is still written,
        so conversation history stays intact.
        """
        self._require_loaded()
        assert self.llm is not None
        last_user_msg, uid, prior, participants = self._resolve_target(target, user_id)
        query = self._weighted_query(last_user_msg, prior)
        messages = self._build_messages(query, uid, prior, participants=participants)

        chat: Optional[Chat] = None
        if isinstance(target, Chat):
            chat = target
        elif isinstance(target, str):
            chat = self._as_chat(target)

        # No tools → legacy single-shot path, bit-for-bit unchanged.
        if tools is None:
            if stream:
                return self._stream_answer(messages, chat, save, uid, auto_extract)
            reply = self.llm.chat(messages)
            self._after_generate(chat, save, reply, auto_extract)
            return reply

        registry = _as_registry(tools)
        if len(registry) == 0:
            # Empty tool set is equivalent to no tools; don't insist on a
            # tool-calling client for it.
            if stream:
                return self._stream_answer(messages, chat, save, uid, auto_extract)
            reply = self.llm.chat(messages)
            self._after_generate(chat, save, reply, auto_extract)
            return reply

        if stream:
            return self._stream_answer_with_tools(
                messages, registry, chat, save, uid,
                max_tool_iterations=max_tool_iterations, tool_choice=tool_choice,
                auto_extract=auto_extract,
            )
        reply = self._generate_with_tools(
            messages, registry,
            max_tool_iterations=max_tool_iterations, tool_choice=tool_choice,
        )
        self._after_generate(chat, save, reply, auto_extract)
        return reply

    # ------------------------------------------------------------------ #
    # Tool-calling loop
    # ------------------------------------------------------------------ #
    def _generate_with_tools(
        self,
        messages: list[dict],
        registry: ToolRegistry,
        *,
        max_tool_iterations: int,
        tool_choice: Optional[Any],
    ) -> str:
        """Non-streaming model↔tool loop; returns the final assistant text.

        Each round appends the assistant tool-call message + one ``tool``-role
        message per call to the in-memory ``messages`` list. Those intermediate
        messages are never persisted — only the returned final text is saved by
        the caller. The loop stops when the model returns plain text (no tool
        calls) or when ``max_tool_iterations`` is reached.
        """
        assert self.llm is not None
        schemas = registry.schemas()
        for _ in range(max(1, max_tool_iterations)):
            resp = self.llm.chat_with_tools(
                messages, schemas, tool_choice=tool_choice,
            )
            if not resp.has_tool_calls:
                return resp.content
            # Append the assistant turn carrying the tool calls (the API needs
            # the exact tool_calls payload echoed back), then the results.
            messages.append(self._assistant_tool_message(resp.content, resp.tool_calls))
            for tc in resp.tool_calls:
                result = registry.execute(tc.name, tc.arguments)
                # Stamp the correlation id so the backend can pair request/result.
                result.call.id = tc.id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result.text,
                })
        # Out of iterations: ask once more for a plain-text answer, no tools.
        resp = self.llm.chat_with_tools(messages, schemas, tool_choice="none")
        return resp.content

    def _stream_answer_with_tools(
        self,
        messages: list[dict],
        registry: ToolRegistry,
        chat: Optional[Chat],
        save: bool,
        user_id: str,
        *,
        max_tool_iterations: int,
        tool_choice: Optional[Any],
        auto_extract: bool = True,
    ) -> Iterator[Any]:
        """Streaming model↔tool loop.

        Yields the event union (:class:`TextChunk` / :class:`ToolCallEvent` /
        :class:`ToolResultEvent`). Each tool round streams its text/ToolCall
        events; the :class:`ToolCallEvent` already carries the *fully
        assembled* calls (the client accumulates OpenAI's argument fragments),
        so the loop uses those directly — no second model call per round. The
        final assistant text streams as text chunks; the concatenated text is
        persisted once the generator closes (when ``save`` and a chat is
        bound), matching the plain-stream save semantics.
        """
        assert self.llm is not None
        schemas = registry.schemas()
        collected: list[str] = []
        for _ in range(max(1, max_tool_iterations)):
            round_calls: list[ToolCall] = []
            round_content: list[str] = []
            has_calls = False
            for ev in self.llm.chat_with_tools_stream(
                messages, schemas, tool_choice=tool_choice,
            ):
                if isinstance(ev, TextChunk):
                    collected.append(ev.text)
                    round_content.append(ev.text)
                    yield ev
                elif isinstance(ev, ToolCallEvent):
                    has_calls = True
                    round_calls = ev.calls
                    yield ev
            if not has_calls:
                # Plain-text reply (streamed above); done.
                if save and chat is not None:
                    self._after_generate(chat, True, "".join(collected), auto_extract)
                return
            # The ToolCallEvent carried the completed calls; use them to extend
            # the transcript and execute the tools (one ToolResultEvent each).
            content = "".join(round_content)
            messages.append(self._assistant_tool_message(content, round_calls))
            for tc in round_calls:
                result = registry.execute(tc.name, tc.arguments)
                result.call.id = tc.id
                yield ToolResultEvent(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result.text,
                })
        # Out of iterations: one more round with tools disabled, streamed.
        for ev in self.llm.chat_with_tools_stream(messages, schemas, tool_choice="none"):
            if isinstance(ev, TextChunk):
                collected.append(ev.text)
                yield ev
        if save and chat is not None:
            self._after_generate(chat, True, "".join(collected), auto_extract)

    @staticmethod
    def _assistant_tool_message(content: str, calls: list[ToolCall]) -> dict:
        """Render an assistant message carrying tool calls in OpenAI's shape."""
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json_dumps(tc.arguments),
                    },
                }
                for tc in calls
            ],
        }

    def memory_tools(self) -> list[Tool]:
        """The built-in read-only memory self-tools for this agent.

        Returns a fresh list each call. Pass it to ``generate_answer``::

            agent.generate_answer(chat, tools=agent.memory_tools())

        Let the character look up its own memories mid-conversation. Always
        read-only — writes stay on the extractor/MCP path.
        """
        return _build_memory_tools(self)

    def _stream_answer(
        self,
        messages: list[dict[str, str]],
        chat: Optional[Chat],
        save: bool,
        user_id: str,
        auto_extract: bool = True,
    ) -> Iterator[str]:
        assert self.llm is not None
        collected: list[str] = []
        for chunk in self.llm.chat_stream(messages):
            collected.append(chunk)
            yield chunk
        if save and chat is not None:
            self._after_generate(chat, True, "".join(collected), auto_extract)

    def _after_generate(
        self, chat: Optional[Chat], save: bool, reply: str, auto_extract: bool = True
    ) -> None:
        if chat is None or not save:
            return
        chat.add_message("assistant", reply)
        # Auto-extraction (a second LLM call on the configured interval) can be
        # deferred by the caller when it wants the reply back the instant the
        # answer is generated — e.g. the configurator's mini-chat, which runs
        # extraction on a background thread so the user isn't blocked on it.
        if auto_extract:
            self._maybe_auto_extract(chat)

    # Extraction
    def _extract_messages(
        self,
        rows: list[dict[str, Any]],
        user_id: str,
        participants: Optional[list[str]] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        """Run extraction over a batch of message rows.

        ``user_id`` is the chat's default user. ``participants`` (when given,
        more than one) drives multi-user extraction: each turn is labelled with
        its real speaker and per-user items are attributed per participant.
        Speaker is read from each row's ``user_id`` (NULL ⇒ the chat owner).

        ``chat_id`` (optional) identifies the conversation; chat-scoped memories
        (``user_facts``, ``episodic``) stamp it onto their rows so the knowledge
        graph can link facts and episodes of the same chat.
        """
        assert self.character is not None
        if not rows:
            return
        event_memory = self.memories.get("conversation_events")
        events_changed = False
        if (
            isinstance(event_memory, ConversationEventMemory)
            and event_memory.enabled
            and chat_id is not None
        ):
            event_memory.ingest_messages(
                rows,
                default_user_id=user_id,
                chat_id=chat_id,
                character_name=self.character_name,
            )
            events_changed = True
        turns = [
            {
                "role": r["role"],
                "content": r["content"],
                "user_id": r.get("user_id"),
                "message_id": r["id"],
                "occurred_at": r.get("occurred_at"),
            }
            for r in rows
        ]
        ids = [int(r["id"]) for r in rows]
        result = self.character.extract(
            turns, user_id=user_id, participants=participants, chat_id=chat_id
        )
        if result is not None:
            added = result.pop("__added__", {})
            if isinstance(event_memory, ConversationEventMemory) and event_memory.enabled:
                event_memory.link_extracted(added, self.memories)
            self.persist_structured()
            # Knowledge graph: ingest the freshly-added items so its nodes
            # exist before dedup possibly mutates their source rows.
            self._update_knowledge_graph(added)
            # Post-extraction dedup: compact the freshly-added items against
            # each memory's existing rows, then mirror the mutations into
            # the knowledge graph.
            self._dedup_added(added)
        elif events_changed:
            self.persist_structured()
        # Mark extracted whether or not the extractor returned data: a None
        # return means no participating memories, so there is nothing to learn.
        assert self.store is not None
        qs = ", ".join("?" for _ in ids)
        self.store.execute(
            f"UPDATE messages SET extracted=1 WHERE id IN ({qs})", ids
        )

    def _extract_chat(self, chat: Chat) -> None:
        interval = max(
            1,
            self.config.memory.extract_interval if self.config is not None else 5,
        )
        window = max(interval * 2, 4)
        rows = chat.unextracted()
        if not rows:
            return
        self._extract_messages(
            rows[-window:], chat.user_id, participants=chat.participants(), chat_id=chat.id
        )

    def _update_knowledge_graph(self, added: dict[str, list]) -> None:
        """Feed freshly-extracted items to the knowledge graph (if enabled).

        The graph reads its source memories' rows to ingest; the `added` map
        tells it which rows are new so the update is incremental rather than
        a full re-ingest. No-op when the KG memory is not configured.
        """
        kg = self.memories.get(_KG_MEMORY)
        if not isinstance(kg, KnowledgeGraphMemory):
            return
        # Only the source memories the graph cares about carry weight here;
        # `KnowledgeGraphRetriever.update` ignores the rest.
        kg.retriever.update(added)
        kg.persist(os.path.join(self.save_directory, "kg_index"))

    def _dedup_added(self, added: dict[str, list]) -> None:
        """Post-extraction step: compact freshly-added items per memory.

        ``added`` maps memory name -> the items that memory's
        ``apply_extraction`` reported as newly added. Only runs when a
        `Deduplicator` is configured. Any mutations are mirrored into the
        knowledge graph via `apply_deduplication`.
        """
        if not self.deduplicator:
            return
        reports: dict[str, DedupReport] = {}
        for name, items in added.items():
            mem = self.memories.get(name)
            if isinstance(mem, StructuredMemory) and items:
                reports[name] = self.deduplicator.dedup_items(mem, items)
        # Mirror the dedup mutations into the knowledge graph so stale nodes
        # are removed/refreshed in lockstep with their source rows.
        kg = self.memories.get(_KG_MEMORY)
        if isinstance(kg, KnowledgeGraphMemory) and reports:
            kg.retriever.apply_deduplication(reports)
            kg.persist(os.path.join(self.save_directory, "kg_index"))

    def dedup(
        self,
        memory_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, DedupReport]:
        """Sweep one or all structured memories for duplicates and compact them.

        - `memory_name`: sweep just that memory; `None` sweeps every structured
          memory (``user_facts``, ``user_directives``, ``episodic``, ``heartbeat``,
          ``user_summary``).
        - `user_id`: sweep a single user's rows only.

        Returns a ``{memory_name: DedupReport}`` mapping. Uses the agent's
        `Deduplicator` if configured; otherwise uses the `Deduplicator` when dedup is
        configured; otherwise a fresh default-config one is built on the fly so
        a one-off sweep is always possible.
        """
        self._require_loaded()
        assert self.embedder is not None
        dedup = self.deduplicator or Deduplicator(
            self.embedder, self.llm, prompts=self.prompts
        )
        names = (memory_name,) if memory_name else _DEDUP_MEMORIES
        reports: dict[str, DedupReport] = {}
        for name in names:
            mem = self.memories.get(name)
            if isinstance(mem, StructuredMemory):
                reports[name] = dedup.sweep(mem, user_id=user_id)
        return reports

    def extract(self, target: Optional[Union[Chat, str]] = None) -> None:
        """Run memory extraction over messages that have not been processed.

        - `target` a `Chat` / chat id: extract that chat only.
        - `target` None: extract every un-extracted message across all chats,
          one batch per chat so a group chat extracts with its real
          participants (per-speaker attribution).
        Idempotent: processed messages are flagged `extracted=1`.
        """
        self._require_loaded()
        assert self.store is not None and self._chats is not None
        if target is not None:
            chat = target if isinstance(target, Chat) else self._as_chat(target)
            if chat is not None:
                self._extract_chat(chat)
            return

        # All chats: batch per chat (not per user) so a group chat keeps its
        # participant set for multi-user extraction. A 1:1 chat behaves exactly
        # as before — its participants list is the single owner.
        rows = self._chats.all_unextracted()
        if not rows:
            return
        interval = max(
            1,
            self.config.memory.extract_interval if self.config is not None else 5,
        )
        window = max(interval * 2, 4)
        by_chat: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_chat.setdefault(r["chat_id"], []).append(r)
        for chat_id, chat_rows in by_chat.items():
            chat = self._chats.load_chat(chat_id)
            if chat is None:
                continue
            self._extract_messages(
                chat_rows[-window:],
                chat.user_id,
                participants=chat.participants(),
                chat_id=chat_id,
            )

    def close(self) -> None:
        if self.store is not None:
            self.persist_structured()
            self.store.close()
