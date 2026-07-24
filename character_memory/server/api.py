"""FastAPI app exposing a two-step chat flow over `CharacterAgent`.

This is the optional server subpackage: install it with
``pip install charactermemory[server]`` (which pulls ``fastapi`` and
``uvicorn``), then run either with the console script::

    charactermemory-server

or with uvicorn directly::

    uvicorn character_memory.server:app --reload

The flow it implements is the typical "thin client" pattern:

1. ``POST /context`` — given a character name, a user id, a user message and
   an optional ``chat_id``, load (or create) the chat, persist the user turn
   (attributed to that speaker), and return the assembled memory context plus
   the (possibly new) chat id. The client uses that context to drive its own
   LLM call.

2. ``POST /save``  — the client comes back with the assistant answer it
   generated; this endpoint persists the assistant turn and runs memory
   extraction so the character learns from the exchange.

The ``user`` field is the **current speaker**. A chat can host several
speakers (a group chat): any caller holding the chat id may post as any
``user``, each turn is attributed to its speaker, and memories are extracted
per participant. A 1:1 chat behaves exactly as before — the ``user`` is the
single participant.

Characters are discovered from the ``assets/`` directory at startup: every
subdirectory of ``assets/`` (e.g. ``assets/Kurisu``) becomes an available
character keyed by its folder name. Override the assets root with the
``CM_ASSETS_DIR`` environment variable.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from character_memory import (
    CharacterAgent,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
)

# Read-side memory browser: normalises each memory backend into paged,
# searchable records and powers the GUI served at /gui.
from .adapters import overview as memory_overview
from .adapters import read_graph as read_graph_view
from .adapters import read_memory as read_memory_page
from .editor import create_record, delete_record, edit_schema, update_record

# MCP (Model Context Protocol) endpoint: JSON-RPC 2.0 over the Streamable
# HTTP transport, JSON-only. Mounted at /mcp with ``?character=<name>``
# binding every call to one CharacterAgent from the AGENTS dict below.
from .mcp import build_router as build_mcp_router
# Admin router: write-side endpoints (create / configure / delete / rebuild /
# chat) that back the Configure tab of the GUI. Reads live in `adapters.py`.
from .admin import build_admin_router as build_admin_router_impl
from .admin import build_jobs_router as build_jobs_router_impl
# Background cache synchronizer: reloads in-RAM hybrid indexes + the KG graph
# when another process (the Discord bot, the CLI, a second worker) writes to
# the shared per-character save_directory.
from .sync import MemorySync

# Default to the current working directory: once installed the package has no
# notion of a "repo root", so the server operates relative to the cwd it is
# launched from. Both are overridable via the environment variables below.
ASSETS_DIR = os.environ.get("CM_ASSETS_DIR", os.path.join(os.getcwd(), "assets"))
SAVE_ROOT = os.environ.get("CM_SAVE_DIR", os.path.join(os.getcwd(), ".cm_servers"))

# Background sync poll interval (seconds). The monitor reloads a character's
# in-RAM caches when its on-disk state is touched by another process; 0
# disables the background poller (caches only refresh on manual ``/refresh``
# / ``refresh_memory`` calls). Useful for tests and single-process runs.
SYNC_INTERVAL = float(os.environ.get("CM_SYNC_INTERVAL", "3.0"))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# --------------------------------------------------------------------------- #
# Character registry: build one CharacterAgent per subfolder of `assets/`.
# --------------------------------------------------------------------------- #
def _parse_rebuild_kg_env() -> set[str]:
    """Parse `CM_REBUILD_KG`.

    Empty/unset -> no rebuilds. ``all`` or ``*`` -> the sentinel ``{"*"}``,
    meaning "every KG-enabled character". Otherwise the comma-separated
    character names to rebuild (lowercased for case-insensitive matching).
    """
    raw = os.environ.get("CM_REBUILD_KG", "").strip().lower()
    if not raw:
        return set()
    if raw in {"all", "*"}:
        return {"*"}
    return {c.strip() for c in raw.split(",") if c.strip()}


def _discover_characters(assets_dir: str) -> dict[str, CharacterAgent]:
    """Build a `CharacterAgent` for each character directory under `assets_dir`.

    A character opts into the knowledge-graph retriever either by shipping a
    `.knowledge_graph` marker file in its directory or by being explicitly
    listed in `CM_KG_CHARACTERS` (a comma-separated env var). Kurisu ships
    with the marker so the KG is on by default for her.

    The KG itself is loaded from disk if a persisted `kg_index` is present --
    it is never silently rebuilt on a plain start. To force a (re)build, list
    the character (or ``all``) in `CM_REBUILD_KG`; that triggers
    `CharacterAgent.rebuild_knowledge_graph()` after the normal load (via
    :func:`_apply_rebuild_kg`). This runs at module import time, so it only
    sees `CM_REBUILD_KG` when the module is imported by a uvicorn worker
    *after* `main()` set it -- i.e. the ``--reload`` subprocess path. The
    in-process (default, no-reload) path is handled directly in :func:`main`.
    """
    agents: dict[str, CharacterAgent] = {}
    if not os.path.isdir(assets_dir):
        return agents
    kg_chars = {
        c.strip() for c in os.environ.get("CM_KG_CHARACTERS", "").split(",") if c.strip()
    }
    rebuild_kg = _parse_rebuild_kg_env()
    rebuild_all = "*" in rebuild_kg
    for name in sorted(os.listdir(assets_dir)):
        char_dir = os.path.join(assets_dir, name)
        if not os.path.isdir(char_dir):
            continue
        marker = os.path.join(char_dir, ".knowledge_graph")
        config_path = os.path.join(char_dir, "config.yaml")
        agent = CharacterAgent(
            directory=char_dir,
            name=name,
            save_directory=os.path.join(SAVE_ROOT, name),
        )
        # Prefer the per-character config.yaml when present (it carries every
        # override: prompts, memory toggles, persona, sub-configs). Otherwise
        # fall back to dataclass defaults + a plain MemoryConfig. The KG opt-in
        # (env override or `.knowledge_graph` marker) is layered on top either
        # way so it stays the single source of truth for "is the graph on".
        if os.path.isfile(config_path):
            agent.load_from_config(config_path)
            if (name in kg_chars or os.path.isfile(marker)) and agent.config is not None:
                agent.config.memory.enabled_knowledge_graph = True
                # Rebuild memories with the KG toggle flipped: load_from_config
                # already wired the standard memories without KG, so reload.
                agent.load_from_config(agent.config)
        else:
            memory_config = MemoryConfig()
            if name in kg_chars or os.path.isfile(marker):
                memory_config.enabled_knowledge_graph = True
            agent.load_from_config(LLMConfig(), EmbeddingConfig(), memory_config)
        agent.build()  # load-or-build (idempotent); KG is loaded, not rebuilt
        agents[name] = agent
    # Apply a requested KG rebuild once all agents are built, using the shared
    # helper so behaviour matches the in-process path in main().
    if rebuild_kg:
        _apply_rebuild_kg(agents, rebuild_all=rebuild_all, targets=rebuild_kg)
    return agents


AGENTS: dict[str, CharacterAgent] = _discover_characters(ASSETS_DIR)


# One cache synchronizer per character. The poller reloads in-RAM hybrid
# indexes + the KG graph when another process (Discord bot, CLI, …) writes to
# the shared save_directory, so the server never serves stale search results
# or graph views. ``CM_SYNC_INTERVAL=0`` leaves the map empty (poller off);
# callers can still force a reload via POST /api/admin/characters/{name}/refresh
# or the ``refresh_memory`` MCP tool. Built at import time so a uvicorn reload
# subprocess re-creates the monitors alongside AGENTS.
SYNC_MONITORS: dict[str, MemorySync] = {
    name: MemorySync(agent, interval=SYNC_INTERVAL).start()
    for name, agent in AGENTS.items()
    if SYNC_INTERVAL > 0
}


def _get_agent(character: str) -> CharacterAgent:
    if character not in AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown character {character!r}. Available: {sorted(AGENTS)}.",
        )
    return AGENTS[character]


# --------------------------------------------------------------------------- #
# Request / response schemas.
# --------------------------------------------------------------------------- #
class ContextRequest(BaseModel):
    character: str = Field(..., description="Character name (a subfolder of assets/).")
    user: str = Field(..., description="User id of the current speaker for this turn.")
    message: str = Field(..., description="The user's latest message.")
    chat_id: Optional[str] = Field(
        default=None,
        description="Existing chat id. If absent or unknown, a new chat is created.",
    )


class ContextResponse(BaseModel):
    chat_id: str
    context: dict[str, str]


class SaveRequest(BaseModel):
    chat_id: str = Field(..., description="The chat id returned by /context.")
    answer: str = Field(..., description="The assistant answer to persist.")


class SaveResponse(BaseModel):
    ok: bool = True
    chat_id: str
    extracted: bool = Field(
        ..., description="Whether memory extraction fired for this turn."
    )


class MemoryMutationRequest(BaseModel):
    """Fields submitted by the browser's schema-driven memory editor."""

    values: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# App + handlers.
# --------------------------------------------------------------------------- #
app = FastAPI(title="CharacterMemory server")

# Mount the MCP (Model Context Protocol) JSON-RPC 2.0 endpoint. Reads reuse
# ``adapters.read_memory`` (semantic search + lexical fallback + pagination)
# so quality matches the GUI; writes commit to SQLite, rebuild the affected
# memory's hybrid index, and ``persist_structured()`` so the change survives
# a server restart. ``SYNC_MONITORS`` is wired in so the ``refresh_memory``
# tool can force an immediate cache reload. ``build_mcp_router`` is in :file:`.mcp`.
app.include_router(build_mcp_router(AGENTS, SYNC_MONITORS))

# Mount the admin router (write side: create/configure/delete/rebuild/chat).
# It receives the live AGENTS dict so mutations are reflected immediately.
# SYNC_MONITORS is passed in so the manual ``/refresh`` endpoint can force an
# immediate cache reload on demand.
app.include_router(build_admin_router_impl(AGENTS, ASSETS_DIR, SAVE_ROOT, SYNC_MONITORS))
app.include_router(build_jobs_router_impl())


@app.on_event("shutdown")
def _shutdown() -> None:
    """Stop the sync pollers, then flush structured-memory indexes to disk."""
    for monitor in SYNC_MONITORS.values():
        try:
            monitor.stop()
        except Exception:  # pragma: no cover - best effort
            pass
    for agent in AGENTS.values():
        try:
            agent.close()
        except Exception:  # pragma: no cover - best effort
            pass


@app.get("/")
def list_characters() -> dict[str, list[str]]:
    """List the characters available behind this server."""
    return {"characters": sorted(AGENTS)}


@app.post("/context", response_model=ContextResponse)
def context(req: ContextRequest) -> ContextResponse:
    """Resolve the chat (creating it if needed), store the user turn as spoken
    by `req.user`, and return the assembled memory context for the character +
    conversation participants.

    A chat id is an unguessable room key: any caller holding it may post as
    any speaker, which is what enables group chats. The chat owner is whoever
    created it; subsequent speakers are recorded via the per-turn `user_id`.
    """
    agent = _get_agent(req.character)

    chat = None
    if req.chat_id:
        chat = agent.load_chat(req.chat_id)
        if chat is None:
            raise HTTPException(
                status_code=404,
                detail=f"chat_id {req.chat_id!r} does not exist for {req.character!r}.",
            )
    if chat is None:
        chat = agent.create_chat(req.user, title=req.message[:60])

    # Persist the user turn attributed to the current speaker. For a group
    # chat this is what makes each participant's messages attributable.
    chat.add_message("user", req.message, user_id=req.user)

    sections = agent.build_context(chat)
    return ContextResponse(chat_id=chat.id, context=sections)


@app.post("/save", response_model=SaveResponse)
def save(req: SaveRequest) -> SaveResponse:
    """Persist the assistant answer and run memory extraction over the chat.

    Returns ``extracted`` so the client knows whether learning fired this turn
    (extraction is throttled by the agent's ``extract_interval``)."""
    # Locate the chat across every character; a chat id is globally unique.
    chat = None
    owner: Optional[CharacterAgent] = None
    for agent in AGENTS.values():
        chat = agent.load_chat(req.chat_id)
        if chat is not None:
            owner = agent
            break
    if chat is None or owner is None:
        raise HTTPException(status_code=404, detail=f"Unknown chat_id {req.chat_id!r}.")

    before = len(chat.unextracted())
    chat.add_message("assistant", req.answer)
    # Force extraction over the chat: anything new gets learned + flagged.
    owner.extract(chat)
    after = len(chat.unextracted())
    owner.persist_structured()

    return SaveResponse(chat_id=chat.id, extracted=before != after)


# --------------------------------------------------------------------------- #
# Memory browser GUI + JSON endpoints.
# --------------------------------------------------------------------------- #
# Static assets (index.html / styles.css / app.js) ship inside the package.
if os.path.isdir(STATIC_DIR):
    app.mount("/gui/static", StaticFiles(directory=STATIC_DIR), name="gui-static")


@app.get("/gui", response_class=HTMLResponse)
def gui() -> HTMLResponse:
    """Serve the single-page memory browser.

    The page talks to the `/api/memories/...` endpoints below. Characters are
    listed via `GET /`.
    """
    index = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index):
        raise HTTPException(status_code=404, detail="GUI assets not built.")
    with open(index, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/memories/{character}")
def list_memories(character: str) -> dict:
    """Sidebar overview: every memory with its record count + known users."""
    agent = _get_agent(character)
    rows = memory_overview(agent)
    for row in rows:
        row["editable"] = bool(edit_schema(agent.memories[row["name"]])["editable"])
    return {"character": character, "memories": rows}


@app.get("/api/memories/{character}/{memory}")
def read_memories(
    character: str,
    memory: str,
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=100),
    user: Optional[str] = Query(None, description="Filter to one user id."),
    q: Optional[str] = Query(None, description="Search query (semantic, lexical fallback)."),
) -> dict:
    """One page of records for a memory, with optional search + user filter."""
    agent = _get_agent(character)
    try:
        payload = read_memory_page(agent, memory, page=page, size=size, user=user, q=q)
        payload["editor"] = edit_schema(agent.memories[memory])
        return payload
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown memory {memory!r} for {character!r}. "
                f"Available: {sorted(agent.memories)}."
            ),
        )


def _memory_for_edit(character: str, memory: str):
    agent = _get_agent(character)
    mem = agent.memories.get(memory)
    if mem is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown memory {memory!r} for {character!r}.",
        )
    if not edit_schema(mem)["editable"]:
        raise HTTPException(status_code=405, detail=edit_schema(mem)["description"])
    return agent, mem


@app.post("/api/memories/{character}/{memory}", status_code=201)
def add_memory_record(character: str, memory: str, req: MemoryMutationRequest) -> dict:
    """Add one editable memory record and refresh its retrieval index."""
    agent, mem = _memory_for_edit(character, memory)
    try:
        record_id = create_record(agent, mem, req.values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "id": record_id, "memory": memory}


@app.put("/api/memories/{character}/{memory}/{record_id}")
def edit_memory_record(
    character: str, memory: str, record_id: str, req: MemoryMutationRequest
) -> dict:
    """Edit one memory record without disturbing its decay bookkeeping."""
    agent, mem = _memory_for_edit(character, memory)
    try:
        updated_id = update_record(agent, mem, record_id, req.values)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Memory record {record_id!r} does not exist.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "id": updated_id, "memory": memory}


@app.delete("/api/memories/{character}/{memory}/{record_id}")
def remove_memory_record(character: str, memory: str, record_id: str) -> dict:
    """Delete one editable memory record and refresh its retrieval index."""
    agent, mem = _memory_for_edit(character, memory)
    try:
        deleted_id = delete_record(agent, mem, record_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Memory record {record_id!r} does not exist.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "id": deleted_id, "memory": memory}


@app.get("/api/graph/{character}")
def read_graph(
    character: str,
    q: Optional[str] = Query(None, description="Query (drives activation; empty = whole graph)."),
    user: Optional[str] = Query(None, description="Bias the user's own PersonNode."),
    limit: int = Query(50, ge=1, le=6000, description="Max nodes to return by activation."),
    hops: int = Query(1, ge=0, le=2, description="Subgraph expansion hops around top nodes."),
    max_edges: int = Query(400, ge=0, le=12000, description="Max edges to return."),
    include_co_occurrence: bool = Query(False, description="Include co_occurrence edges (default on in full mode)."),
    full: bool = Query(False, description="Return the whole graph (retrieved nodes flagged)."),
    retrieve_k: int = Query(30, ge=1, le=500, description="Top-k nodes flagged 'retrieved' when full."),
) -> dict:
    """Return the activation-weighted knowledge-graph for the viz.

    `full=False` (default) returns the activation-weighted subgraph; `full=True`
    returns the entire graph with a `retrieved` flag per node so the GUI can
    light up the retrieved nodes and gray out the rest. Node radius / opacity /
    color encode `activation`; edges carry their `weight`. Requires the
    `knowledge_graph` memory to be enabled for this character.
    """
    agent = _get_agent(character)
    if "knowledge_graph" not in agent.memories:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Character {character!r} does not have the knowledge_graph "
                f"memory enabled."
            ),
        )
    try:
        return read_graph_view(
            agent, q=q, user=user, limit=limit, hops_subgraph=hops, max_edges=max_edges,
            include_co_occurrence=include_co_occurrence, full=full, retrieve_k=retrieve_k,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="knowledge_graph memory not built.")


def main() -> None:  # pragma: no cover - manual run helper / console script
    """Console-script entry point: run the server with uvicorn.

    Examples::

        charactermemory-server
        charactermemory-server --rebuild-kg kurisu
        charactermemory-server --rebuild-kg        # all KG-enabled characters
        charactermemory-server --rebuild-kg Kurisu Mayuri --port 9000
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="charactermemory-server",
        description="Run the CharacterMemory FastAPI server.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000).",
    )
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument(
        "--reload", dest="reload", action="store_true", default=None,
        help="Enable uvicorn auto-reload (default unless --rebuild-kg is given).",
    )
    reload_group.add_argument(
        "--no-reload", dest="reload", action="store_false",
        help="Disable uvicorn auto-reload.",
    )
    parser.add_argument(
        "--rebuild-kg", nargs="*", default=None, metavar="CHARACTER",
        help=(
            "Rebuild one or more characters' knowledge graph at startup, then "
            "serve. With no names, rebuilds every KG-enabled character. "
            "Default: load existing graphs (no rebuild)."
        ),
    )
    args = parser.parse_args()

    # A KG rebuild runs an expensive LLM extraction pass; default to no reload
    # when --rebuild-kg is given so it isn't re-triggered on every file change.
    reload_flag = args.reload
    if reload_flag is None:
        reload_flag = args.rebuild_kg is None
        if args.rebuild_kg is not None and not reload_flag:
            print(
                "[charactermemory] --rebuild-kg given: auto-reload disabled by "
                "default (pass --reload to force)."
            )

    requested = args.rebuild_kg
    if requested is not None:
        names = [n.strip() for n in requested if n and n.strip()]
        rebuild_all = not names
        targets = {n.lower() for n in names}
        # Bridge to a uvicorn worker subprocess (the --reload re-import path)
        # which reads CM_REBUILD_KG inside _discover_characters.
        os.environ["CM_REBUILD_KG"] = ",".join(names) if names else "all"
        # In-process path (reload off, the default). `AGENTS` was built at
        # module import time -- before main() ran -- so it has NOT seen
        # CM_REBUILD_KG; rebuild the already-loaded agents directly. When
        # reload is on the fresh subprocess will do it, so skip here to avoid a
        # double rebuild.
        if not reload_flag:
            applied = _apply_rebuild_kg(AGENTS, rebuild_all=rebuild_all, targets=targets)
            if not applied:
                print(
                    "[charactermemory] --rebuild-kg: no matching characters "
                    f"({sorted(names) or 'all'}). Available: {sorted(AGENTS)}."
                )

    import uvicorn

    uvicorn.run(
        "character_memory.server:app",
        host=args.host,
        port=args.port,
        reload=reload_flag,
    )


def _apply_rebuild_kg(
    agents: "dict[str, CharacterAgent]",
    *,
    rebuild_all: bool,
    targets: "set[str]",
) -> bool:
    """Rebuild the knowledge graph for the matching agents in `agents`.

    Returns True if at least one rebuild ran. A character is rebuilt when it
    has the knowledge_graph memory enabled AND (rebuild_all is set, `targets`
    contains the ``"*"`` sentinel, OR its name matches `targets`
    case-insensitively). Names that don't exist or don't have KG enabled are
    reported and skipped.
    """
    applied = False
    rebuild_all = rebuild_all or "*" in targets
    for name, agent in sorted(agents.items()):
        wants = rebuild_all or name.lower() in targets
        if not wants:
            continue
        kg_enabled = "knowledge_graph" in agent.memories
        if not kg_enabled:
            print(
                f"[charactermemory] --rebuild-kg: {name} has the knowledge_graph "
                f"memory disabled; skipping."
            )
            continue
        print(f"[charactermemory] rebuilding knowledge graph: {name}")
        agent.rebuild_knowledge_graph()
        applied = True
    return applied


if __name__ == "__main__":  # pragma: no cover - manual run helper
    main()
