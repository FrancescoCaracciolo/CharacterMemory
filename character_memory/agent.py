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
from typing import Any, Iterator, Optional, Union

from .chat import Chat, _ChatBackend
from .chunking.registry import get_chunker
from .character import Character
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
from .memory.dedup import Deduplicator, DedupReport
from .memory.emotion import EmotionStatus
from .memory.episodic import EpisodicMemory
from .memory.heartbeat import HeartbeatJournal
from .memory.store import SQLiteStore
from .memory.structured import StructuredMemory
from .memory.user_directives import UserDirectiveMemory
from .memory.user_facts import UserFactMemory
from .prompts import PromptConfig
from .rag.hybrid import HybridSearch

_INFO_GLOB = "Information"
_DIALOGUE_GLOB = "Dialogues"

# Names of the two RAG memories populated from the character directory and of
# the structured memories whose hybrid index is rebuilt from SQLite rows.
_STRUCTURED_MEMORIES = ("user_facts", "user_directives", "episodic", "heartbeat")

# Anything generate_answer / build_context / render_prompt accepts as a
# conversation target.
Target = Union[Chat, str, list[dict[str, str]]]


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
        # Short character blurb surfaced to the extractor (and the answer prompt)
        # so the model knows who the character is. Auto-summarizing the
        # Information/*.md into a blurb is intentionally out of scope; callers
        # pass this in if they want the persona clause populated.
        self.persona = persona

        # Not wired until load_* is called.
        self.config: Optional[CharacterMemoryConfig] = None
        self.llm: Optional[LLMClient] = None
        self.embedder: Optional[EmbeddingProvider] = None
        self.store: Optional[SQLiteStore] = None
        self.memories: dict[str, Memory] = {}
        self.character: Optional[Character] = None
        self.deduplicator: Optional[Deduplicator] = None
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
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingProvider] = None,
        config: Optional[CharacterMemoryConfig] = None,
    ) -> "CharacterAgent":
        """Wire backends from config objects.

        Either pass a top-level `config` (a full
        `CharacterMemoryConfig`) or any combination of the individual
        sub-configs; omitted ones default to their dataclass defaults.
        `llm` / `embedder` let you override the auto-built OpenAI clients.
        """
        if config is not None:
            full = config
        else:
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
        self.memories["episodic"] = EpisodicMemory(
            self.store, hybrid(), enabled=m.is_enabled("episodic"),
            half_life=half, sticky_threshold=sticky,
        )
        self.memories["heartbeat"] = HeartbeatJournal(
            self.store, hybrid(), enabled=m.is_enabled("heartbeat"),
            half_life=half, sticky_threshold=sticky,
        )
        self.memories["emotion"] = EmotionStatus(
            self.store, enabled=m.is_enabled("emotion"),
            baseline=m.emotion_baseline, user_dims=m.emotion_user_dims,
        )

    def _wire_character(self) -> None:
        self._limits = {
            name: (self.config.memory.k_for(name) if self.config else 4)
            for name in self.memories
        }
        self.character = Character(
            character_name=self.character_name,
            base_instruction=self.persona,
            memories=list(self.memories.values()),
            llm=self.llm,
            prompts=self.prompts,
        )

    # Indexing
    def _chunking_config(self) -> ChunkingConfig:
        return self.config.chunking if self.config is not None else ChunkingConfig()

    def _index_character(self) -> tuple[int, int]:
        """Chunk + build + persist the wiki & dialogue indexes."""
        assert self.store is not None
        c = self._chunking_config()
        info_dir = os.path.join(self.character_dir, _INFO_GLOB)
        info_chunks = (
            get_chunker(c.info_chunker, max_tokens=c.header_max_tokens).chunk_directory(info_dir)
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

    def build(self) -> "CharacterAgent":
        """Load persisted indexes if present, otherwise build + persist them."""
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
        self._built = True
        return self

    def rebuild(self) -> "CharacterAgent":
        """Force a full re-chunk + re-index, overwriting persisted indexes."""
        self._require_loaded()
        assert self.store is not None
        self._index_character()
        for name in _STRUCTURED_MEMORIES:
            mem = self.memories.get(name)
            if mem is None:
                continue
            mem.rebuild_index()
            mem.persist(os.path.join(self.save_directory, f"{name}_index"))
        self._built = True
        return self

    def persist_structured(self) -> None:
        """Persist the structured-memory indexes (call after learning)."""
        for name in _STRUCTURED_MEMORIES:
            mem = self.memories.get(name)
            if mem is None:
                continue
            mem.persist(os.path.join(self.save_directory, f"{name}_index"))

    # Target resolution
    def _as_chat(self, target: Union[Chat, str]) -> Optional[Chat]:
        """Resolve a `Chat` or chat id to a `Chat` (or `None` if missing)."""
        assert self._chats is not None
        if isinstance(target, Chat):
            return target
        return self._chats.load_chat(target)

    def _resolve_target(
        self, target: Target, user_id: str
    ) -> tuple[str, str, list[dict[str, str]]]:
        """Return (query, user_id, prior_messages) for a target.

        - `Chat`                 -> (last user msg or "", chat.user_id, history)
        - chat id `str`          -> same, after load_chat
        - raw query `str`        -> (target, user_id, [])
        - `list[dict]` messages  -> (last user content or "", user_id, target)
        """
        if isinstance(target, Chat):
            chat = target
            return (chat.last_user_message() or "", chat.user_id, chat.messages())
        if isinstance(target, str):
            chat = self._as_chat(target)
            if chat is not None:
                return (chat.last_user_message() or "", chat.user_id, chat.messages())
            # Bare query string, not a known chat id.
            return (target, user_id, [])
        # Message list.
        msgs = list(target)
        last_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user" and m.get("content"):
                last_user = m["content"]
                break
        return (last_user, user_id, msgs)

    # Prompt
    def build_context(
        self, target: Target, *, user_id: str = "default"
    ) -> dict[str, str]:
        """Return `{memory_name: rendered_section}` for the target."""
        self._require_loaded()
        assert self.character is not None
        query, uid, _ = self._resolve_target(target, user_id)
        return self.character.build_context(query, uid, limits=self._limits)

    def render_prompt(self, target: Target, *, user_id: str = "default") -> str:
        """Full system-style context block (system line + all sections)."""
        self._require_loaded()
        assert self.character is not None
        query, uid, _ = self._resolve_target(target, user_id)
        return self.character.render_prompt(query, uid, limits=self._limits)

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
        self, query: str, user_id: str, prior: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        assert self.character is not None
        system = self.character.render_prompt(query, user_id, limits=self._limits)
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
    ) -> Union[str, Iterator[str]]:
        """Generate an assistant reply for `target`.

        `target` is a `Chat`, a chat id, or a raw list of openai-style
        messages. When `target` is a `Chat` (or chat id) and `save` is
        true, the latest user message is expected to already be persisted in
        the chat, and the assistant reply is persisted too; auto-extraction
        fires on the configured interval. `stream=True` returns an iterator
        of text chunks (the full reply is still persisted once the stream
        completes when `save` is true).
        """
        self._require_loaded()
        assert self.llm is not None
        query, uid, prior = self._resolve_target(target, user_id)
        messages = self._build_messages(query, uid, prior)

        chat: Optional[Chat] = None
        if isinstance(target, Chat):
            chat = target
        elif isinstance(target, str):
            chat = self._as_chat(target)

        if stream:
            return self._stream_answer(messages, chat, save, uid)
        reply = self.llm.chat(messages)
        self._after_generate(chat, save, reply)
        return reply

    def _stream_answer(
        self,
        messages: list[dict[str, str]],
        chat: Optional[Chat],
        save: bool,
        user_id: str,
    ) -> Iterator[str]:
        assert self.llm is not None
        collected: list[str] = []
        for chunk in self.llm.chat_stream(messages):
            collected.append(chunk)
            yield chunk
        if save and chat is not None:
            self._after_generate(chat, True, "".join(collected))

    def _after_generate(self, chat: Optional[Chat], save: bool, reply: str) -> None:
        if chat is None or not save:
            return
        chat.add_message("assistant", reply)
        self._maybe_auto_extract(chat)

    # Extraction
    def _extract_messages(
        self, rows: list[dict[str, Any]], user_id: str
    ) -> None:
        """Run extraction over a batch of message rows for one user."""
        assert self.character is not None
        if not rows:
            return
        turns = [{"role": r["role"], "content": r["content"]} for r in rows]
        ids = [int(r["id"]) for r in rows]
        result = self.character.extract(turns, user_id=user_id)
        if result is not None:
            self.persist_structured()
            # Post-extraction dedup: compact the freshly-added items against
            # each memory's existing rows.
            self._dedup_added(result.pop("__added__", {}))
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
        self._extract_messages(rows[-window:], chat.user_id)

    def _dedup_added(self, added: dict[str, list]) -> None:
        """Post-extraction step: compact freshly-added items per memory.

        ``added`` maps memory name -> the items that memory's
        ``apply_extraction`` reported as newly added. Only runs when a
        `Deduplicator` is configured.
        """
        if not self.deduplicator:
            return
        for name, items in added.items():
            mem = self.memories.get(name)
            if isinstance(mem, StructuredMemory) and items:
                self.deduplicator.dedup_items(mem, items)

    def dedup(
        self,
        memory_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, DedupReport]:
        """Sweep one or all structured memories for duplicates and compact them.

        - `memory_name`: sweep just that memory; `None` sweeps every structured
          memory (``user_facts``, ``user_directives``, ``episodic``, ``heartbeat``).
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
        names = (memory_name,) if memory_name else _STRUCTURED_MEMORIES
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
          grouped by user.
        Idempotent: processed messages are flagged `extracted=1`.
        """
        self._require_loaded()
        assert self.store is not None and self._chats is not None
        if target is not None:
            chat = target if isinstance(target, Chat) else self._as_chat(target)
            if chat is not None:
                self._extract_chat(chat)
            return

        # All chats: group un-extracted rows by user_id so each batch feeds the
        # right per-user memories.
        rows = self._chats.all_unextracted()
        if not rows:
            return
        interval = max(
            1,
            self.config.memory.extract_interval if self.config is not None else 5,
        )
        window = max(interval * 2, 4)
        by_user: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            uid = self._chats.chat_user(r["chat_id"]) or "default"
            by_user.setdefault(uid, []).append(r)
        for uid, user_rows in by_user.items():
            self._extract_messages(user_rows[-window:], uid)

    def close(self) -> None:
        if self.store is not None:
            self.persist_structured()
            self.store.close()
