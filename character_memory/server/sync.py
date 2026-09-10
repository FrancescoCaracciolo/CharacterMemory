"""Background cache synchronizer: keeps the server's in-RAM memory indexes
and knowledge graph in sync with writes from other processes that share the
same per-character ``save_directory``.

Why this exists
---------------
Each :class:`CharacterAgent` caches memory state in RAM at ``build()`` time:

* ``StructuredMemory.hybrid`` (BM25 + FAISS over SQLite rows) — used by
  ``user_facts`` / ``user_directives`` / ``episodic`` / ``heartbeat`` /
  ``user_summary``.
* ``KnowledgeGraphMemory.retriever.graph`` (in-memory node/edge dict) plus
  its node-text hybrid index.

These caches are only refreshed by writes in *this* process. SQLite-level
concurrency is already cross-process safe (WAL + ``busy_timeout`` + ``flock``
in :meth:`HybridSearch.persist`), so the on-disk data is always consistent —
but the server's in-RAM caches grow stale when another process (the Discord
bot, the CLI, a second worker) learns and persists. Without this module the
next ``search_memory`` / ``search_knowledge_graph`` / ``/api/graph`` /
``/context`` would rank against the old text and miss the freshly-learned
rows and graph nodes.

What it watches
---------------
Each memory's persisted hybrid-index file (the atomic thing its write path
touches last):

* ``<save>/user_facts_index/nodes.json``     — ``user_facts``
* ``<save>/user_directives_index/nodes.json`` — ``user_directives``
* ``<save>/episodic_index/nodes.json``       — ``episodic``
* ``<save>/heartbeat_index/nodes.json``      — ``heartbeat``
* ``<save>/user_summary_index/nodes.json``   — ``user_summary``
* ``<save>/character_info_index/nodes.json`` — ``character_info``
* ``<save>/dialogue_index/nodes.json``       — ``dialogue_style``
* ``<save>/kg_index/nodes.json``             — ``knowledge_graph``

Every learning pass ends with :meth:`CharacterAgent.persist_structured` (or
the write-tool's ``_persist_after_write``), which rewrites these files via
``HybridSearch.persist``'s temp-write + ``os.replace``. ``nodes.json`` is
published last, after the matching FAISS file, and readers take the same
index lock while loading both. Its mtime is therefore a commit signal that
fires exactly once per completed write — in this process *or* another.

We deliberately do **not** watch ``memory.db`` directly: SQLite runs in WAL
mode (``PRAGMA journal_mode=WAL``) for cross-process concurrency, and under
WAL a row write lands in ``memory.db-wal`` without advancing the main file's
mtime until a checkpoint. The hybrid index, by contrast, is rewritten
end-to-end on every persist, so its mtime is the trustworthy signal. The KG
case is slightly special: its graph lives in ``kg_nodes`` / ``kg_edges`` (so
``memory.db-wal`` carries it), but ``persist_structured`` always rewrites
``kg_index/nodes.json`` right after, so watching that file catches KG
mutations too.

Emotion and browse/pagination are already live (they read SQLite on every
call), so nothing is reloaded for them.

Concurrency notes
-----------------
Reloads are gated by an mtime check: a reload only fires when the on-disk
file is newer than what we last loaded. Because the write paths
(``_persist_after_write`` / ``persist_structured``) advance the mtime as
their *final* atomic step (temp-write + ``os.replace``), the poller either
sees the old mtime (no reload — correct, nothing new committed yet) or the
new mtime (reload — correct, loads the freshly-persisted state). The shared
index lock also prevents a manual/direct load from combining files from two
different publications.

A per-agent :class:`threading.RLock` serializes the poller against a manual
:meth:`check_now` (from the ``/refresh`` endpoint) so two concurrent reloads
don't stomp each other; it intentionally does *not* wrap normal reads, so a
search in flight while a reload swaps state may, in the rare case it straddles
the GIL-yielding embed call, return one slightly-off ranking. The next read
sees consistent state. This matches the trade-off accepted for read-side
cache coherence.
"""

from __future__ import annotations

import os
import threading
import warnings
from typing import Any, Optional

from ..agent import CharacterAgent
from ..memory.character_base import RAGMemory
from ..memory.knowledge_graph_memory import KnowledgeGraphMemory
from ..memory.structured import StructuredMemory
from ..memory.world import WorldMemory


def _mtime(path: str) -> float:
    """Return the mtime of ``path`` (0.0 when it does not exist)."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


class MemorySync:
    """Per-character mtime watcher that reloads in-RAM caches when disk is newer.

    Construct one per agent, :meth:`start` the daemon thread, and call
    :meth:`stop` on shutdown. :meth:`check_now` forces an immediate reload
    (used by the ``POST /api/admin/characters/{name}/refresh`` endpoint and
    the ``refresh_memory`` MCP tool) so a caller can refresh right after a
    known external write rather than waiting up to ``interval`` seconds.
    """

    def __init__(self, agent: CharacterAgent, *, interval: float = 3.0) -> None:
        self.agent = agent
        self.interval = max(0.0, float(interval))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # Set during in-process bulk writes (e.g. ``CharacterAgent.rebuild()``
        # from the admin ``/rebuild`` job) so the poller doesn't race the
        # writer and call ``mem.load()`` on an agent mid-rebuild.
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Last-seen mtimes for every persisted hybrid index, seeded from disk
        # so the first tick only reloads things that change *after* the monitor
        # starts. Keyed by memory name; the value is the mtime of
        # ``<save>/<name>_index/nodes.json`` (or ``kg_index/nodes.json`` for
        # the knowledge graph). Memories without a built index map to 0.0 and
        # are skipped until they appear on disk.
        self._index_mtimes: dict[str, float] = self._snapshot_mtimes()

    def _snapshot_mtimes(self) -> dict[str, float]:
        """Current mtime of every memory's persisted index (0.0 if absent)."""
        out: dict[str, float] = {}
        for name in self.agent.memories:
            out[name] = _mtime(self._index_nodes_json(name))
        return out

    # --------------------------------------------------------------- paths
    def _index_dir(self, name: str) -> str:
        # The knowledge graph persists under ``kg_index``; every other memory
        # under ``<name>_index``. Both follow the same ``nodes.json`` layout
        # (see CharacterAgent.persist_structured / build for the path choice).
        subdir = "kg_index" if name == "knowledge_graph" else f"{name}_index"
        return os.path.join(self.agent.save_directory, subdir)

    def _index_nodes_json(self, name: str) -> str:
        return os.path.join(self._index_dir(name), "nodes.json")

    # ----------------------------------------------------------- lifecycle
    def start(self) -> "MemorySync":
        """Start the daemon poller. No-op when ``interval <= 0`` (disabled)."""
        if self.interval <= 0 or self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"mem-sync-{self.agent.character_name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the poller to stop and briefly join. Safe to call twice."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def pause(self) -> None:
        """Temporarily suspend reloads. ``_tick`` becomes a no-op while set.

        Used around in-process bulk writes (e.g. ``CharacterAgent.rebuild()``
        from the admin ``/rebuild`` job) so the poller doesn't race the writer
        and call ``mem.load()`` on an agent mid-rebuild.
        """
        self._paused.set()

    def resume(self) -> None:
        """Re-enable reloads and re-seed mtimes from current disk state.

        Call this after :meth:`pause` + the bulk write so the poller treats the
        freshly-written state as the new baseline rather than immediately
        reloading it (which would be harmless but wasteful).
        """
        with self._lock:
            self._index_mtimes = self._snapshot_mtimes()
        self._paused.clear()

    # ---------------------------------------------------------------- core
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - never let the poller die
                warnings.warn(
                    f"MemorySync tick for {self.agent.character_name!r} raised "
                    f"{e!r}; will retry next interval.",
                    stacklevel=2,
                )
            # Event.wait returns True when set, so we get a prompt shutdown.
            self._stop.wait(self.interval)

    def check_now(self) -> bool:
        """Force an immediate reload check. Returns True if anything reloaded.

        Public entry point for the ``/refresh`` endpoint and ``refresh_memory``
        MCP tool — call this when you *know* another process just wrote and you
        don't want to wait for the next poller tick. Honours :meth:`pause`: a
        paused monitor reports ``False`` (a bulk write is in progress and the
        caller should not be served a reload right now).
        """
        if self._paused.is_set():
            return False
        with self._lock:
            return self._tick()

    def _tick(self) -> bool:
        """Return True if any cache was reloaded. Idempotent when nothing moved.

        Returns False without touching anything when :meth:`pause` is set, so
        the background loop is a cheap no-op during an in-process bulk write.

        For each memory on the agent, compares the current mtime of its
        persisted index file against the last-seen value. When newer, reloads
        just that memory's hybrid (and, for the knowledge graph, its graph
        from ``kg_nodes`` / ``kg_edges``). Per-memory granularity means a
        ``user_facts`` write doesn't needlessly reload ``episodic``.
        """
        if self._paused.is_set():
            return False
        reloaded = False
        for name, mem in self.agent.memories.items():
            hybrid = getattr(mem, "hybrid", None)
            if isinstance(mem, WorldMemory):
                hybrid = mem.records.hybrid
            if isinstance(mem, KnowledgeGraphMemory):
                hybrid = mem.retriever.hybrid
            if getattr(hybrid, "remote", False):
                reloaded = hybrid.refresh() or reloaded
                continue
            p = self._index_nodes_json(name)
            m = _mtime(p)
            if m <= self._index_mtimes.get(name, 0.0) or m <= 0.0:
                continue
            if isinstance(mem, KnowledgeGraphMemory):
                if self._reload_kg():
                    reloaded = True
                    self._index_mtimes[name] = m
            elif isinstance(mem, (StructuredMemory, RAGMemory, WorldMemory)):
                if self._safe_load(mem, name, self._index_dir(name)):
                    reloaded = True
                    self._index_mtimes[name] = m
        return reloaded

    # ---------------------------------------------------------- reloaders
    def _reload_kg(self) -> bool:
        """Reload the knowledge-graph memory's graph + hybrid from disk.

        ``KnowledgeGraphMemory.load`` reads the ``kg_nodes`` / ``kg_edges``
        SQLite tables (the source of truth for graph structure) and the
        ``kg_index`` hybrid directory (node-text retrieval). A failure is
        warned and reported as ``False`` so one bad KG can't poison the
        poller or block reloads of the other memories.
        """
        kg = self.agent.memories.get("knowledge_graph")
        if not isinstance(kg, KnowledgeGraphMemory):
            return False
        kg_path = os.path.join(self.agent.save_directory, "kg_index")
        try:
            kg.load(kg_path)
            return True
        except Exception as e:  # noqa: BLE001 - keep the poller alive
            warnings.warn(
                f"MemorySync: KG reload for {self.agent.character_name!r} "
                f"failed: {e!r}.",
                stacklevel=2,
            )
            return False

    def _safe_load(self, mem: Any, name: str, index_dir: str) -> bool:
        """Load a memory's hybrid from ``index_dir``; warn+skip on failure."""
        if not os.path.isfile(os.path.join(index_dir, "nodes.json")):
            return False  # never built; nothing to reload
        try:
            mem.load(index_dir)
            return True
        except Exception as e:  # noqa: BLE001 - keep the poller alive
            warnings.warn(
                f"MemorySync: reload of {name!r} for "
                f"{self.agent.character_name!r} failed: {e!r}.",
                stacklevel=2,
            )
            return False


__all__ = ["MemorySync"]
