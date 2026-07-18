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
    PromptConfig,
)

# Read-side memory browser: normalises each memory backend into paged,
# searchable records and powers the GUI served at /gui.
from .adapters import overview as memory_overview
from .adapters import read_memory as read_memory_page

# Default to the current working directory: once installed the package has no
# notion of a "repo root", so the server operates relative to the cwd it is
# launched from. Both are overridable via the environment variables below.
ASSETS_DIR = os.environ.get("CM_ASSETS_DIR", os.path.join(os.getcwd(), "assets"))
SAVE_ROOT = os.environ.get("CM_SAVE_DIR", os.path.join(os.getcwd(), ".cm_servers"))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# --------------------------------------------------------------------------- #
# Character registry: build one CharacterAgent per subfolder of `assets/`.
# --------------------------------------------------------------------------- #
def _discover_characters(assets_dir: str) -> dict[str, CharacterAgent]:
    """Build a `CharacterAgent` for each character directory under `assets_dir`."""
    agents: dict[str, CharacterAgent] = {}
    if not os.path.isdir(assets_dir):
        return agents
    for name in sorted(os.listdir(assets_dir)):
        char_dir = os.path.join(assets_dir, name)
        if not os.path.isdir(char_dir):
            continue
        agent = CharacterAgent(
            directory=char_dir,
            name=name,
            save_directory=os.path.join(SAVE_ROOT, name),
            prompt_config=PromptConfig(),
        )
        agent.load_from_config(LLMConfig(), EmbeddingConfig(), MemoryConfig())
        agent.build()  # load-or-build (idempotent)
        agents[name] = agent
    return agents


AGENTS: dict[str, CharacterAgent] = _discover_characters(ASSETS_DIR)


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


# --------------------------------------------------------------------------- #
# App + handlers.
# --------------------------------------------------------------------------- #
app = FastAPI(title="CharacterMemory server")


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush structured-memory indexes to disk on exit."""
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
    return {"character": character, "memories": memory_overview(agent)}


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
        return read_memory_page(agent, memory, page=page, size=size, user=user, q=q)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown memory {memory!r} for {character!r}. "
                f"Available: {sorted(agent.memories)}."
            ),
        )


def main() -> None:  # pragma: no cover - manual run helper / console script
    """Console-script entry point: run the server with uvicorn."""
    import uvicorn

    uvicorn.run("character_memory.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    main()
