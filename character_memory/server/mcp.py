"""Minimal MCP (Model Context Protocol) HTTP server for memory edit/search.

A lightweight MCP-conformant JSON-RPC 2.0 server that lets MCP clients
(Claude Desktop, MCP Inspector, the Python ``mcp`` client, …) edit and
search a character's memories. The endpoint is ``POST /mcp`` with a
required ``?character=<name>`` query parameter and an optional category
selection such as ``?character=Kurisu&tools=heartbeat``; one
``CharacterAgent`` per character is reused from the global registry that
:mod:`.api` builds at startup.

We don't pull in the official ``mcp`` Python package: the protocol surface
we need (initialize, tools/list, tools/call, ping, the
``notifications/initialized`` ack) is small enough to implement directly
on top of FastAPI, and a hand-rolled handler routes every request through
the ``?character=`` query param without the package's session indirection.
This is the JSON-Mode subset of MCP's Streamable HTTP transport: every
response is a regular JSON envelope. Logic-only tools (search/list) reuse
:func:`character_memory.server.adapters.read_memory` so search quality
and pagination are exactly what the GUI sees.

Write tools commit to the character's SQLite store, incrementally update the
affected memory's hybrid index, and call ``agent.persist_structured()`` so RAG
retrieval stays in sync — without
that step the next ``search_memory`` call would rank against the old text
(see the gotcha note in :file:`character_memory/agent.py`).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from datetime import date as calendar_date
from datetime import datetime, time, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from ..agent import CharacterAgent
from ..chunking import Chunk
from ..emotion_vectors import emotion_vector, encode_emotion_vector
from .adapters import overview as memory_overview, read_graph, read_memory
from ..memory.character_base import RAGMemory
from ..memory.emotion import EmotionStatus
from ..memory.knowledge_graph_memory import KnowledgeGraphMemory
from ..memory.structured import StructuredMemory
from ..memory.world import WorldCommand, WorldMemory
from ..tools import (
    CalculateTimeDifference,
    GetConversationEvents,
    GetEventNeighbors,
    GetKnowledgeGraphNeighbors,
    GetKnowledgeGraphNodes,
    ResolveTimeRange,
    SearchConversationEvents,
    Tool,
)
from .sync import MemorySync


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SERVER_NAME = "character-memory-mcp"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"  # MCP protocol version this server speaks.
TOOL_CATEGORIES = frozenset({"core", "memory", "events", "kg", "heartbeat", "world"})

# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _parse_iso(ts: Optional[str]) -> Optional[float]:
    """Parse an ISO 8601 string to a Unix epoch float; reject bad input.

    ``"Z"`` shorthand (UTC) is normalised to ``+00:00`` so
    :meth:`datetime.fromisoformat` accepts it on Python 3.10.
    """
    if ts is None or ts == "":
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid ISO 8601 timestamp {ts!r}: {e}") from e


def _calendar_day_bounds(
    value: Any, timezone_name: Any = "UTC"
) -> tuple[float, float, str, str]:
    """Return the half-open epoch range for one calendar day.

    ``timezone_name`` is an IANA zone so dates remain correct across daylight
    saving transitions; the returned ISO strings make the interpreted range
    explicit to MCP callers.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("date is required and must use YYYY-MM-DD format.")
    try:
        day = calendar_date.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(f"Invalid calendar date {value!r}; expected YYYY-MM-DD.") from e

    zone_name = str(timezone_name or "UTC").strip() or "UTC"
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as e:
        raise ValueError(
            f"Unknown IANA timezone {zone_name!r}; use a name such as 'UTC' or 'Europe/Rome'."
        ) from e

    start = datetime.combine(day, time.min, tzinfo=zone)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone)
    return start.timestamp(), end.timestamp(), start.isoformat(), end.isoformat()


def _json_result(payload: Any) -> dict[str, Any]:
    """Wrap a Python ``payload`` into MCP's tools/call content shape."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
        "isError": False,
    }


def _json_error(text: str) -> dict[str, Any]:
    """Wrap an error message into MCP's tools/call content shape (``isError=true``)."""
    return {
        "content": [{"type": "text", "text": text}],
        "isError": True,
    }


def _as_tool_result(payload: Any) -> dict[str, Any]:
    """Preserve an MCP error result returned directly by a handler."""
    if (
        isinstance(payload, dict)
        and payload.get("isError") is True
        and isinstance(payload.get("content"), list)
    ):
        return payload
    return _json_result(payload)


def _require_character(
    request: Request, agent_registry: dict[str, CharacterAgent]
) -> tuple[Optional[CharacterAgent], Optional[JSONResponse]]:
    """Resolve the ``?character=`` query param to an agent; on failure
    return a JSON-RPC error envelope (NOT a FastAPI HTTPException) so MCP
    clients see ``{"jsonrpc":..., "error":...}`` instead of FastAPI's
    default ``{"detail": …}`` shape, which they cannot parse.

    Returns ``(agent, error_response)``. Exactly one of the two is
    ``None`` — the caller branches on which is set.
    """
    name = request.query_params.get("character")
    if not name:
        return None, JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Missing ?character=<name> query parameter.",
                },
            },
            status_code=400,
        )
    if name not in agent_registry:
        return None, JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": f"Unknown character {name!r}. Available: {sorted(agent_registry)}.",
                },
            },
            status_code=404,
        )
    return agent_registry[name], None


def _args(args: dict, key: str, default=None):
    """``args[key]`` with a default; ``None`` sent by callers falls back."""
    return args[key] if args.get(key) is not None else default


def _parse_tool_categories(raw: Optional[str]) -> Optional[frozenset[str]]:
    """Parse the optional comma-separated ``?tools=`` category selection.

    ``None`` and the explicit value ``all`` mean the complete registry. An
    empty or unknown selection is rejected so a misspelled category cannot
    silently expose a different set of tools than the caller intended.
    """
    if raw is None:
        return None
    values = frozenset(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )
    if not values:
        raise ValueError(
            "?tools= must contain one or more categories: "
            + ", ".join(sorted(TOOL_CATEGORIES))
            + ", or all."
        )
    if "all" in values:
        if len(values) != 1:
            raise ValueError("?tools=all cannot be combined with other categories.")
        return None
    unknown = values - TOOL_CATEGORIES
    if unknown:
        raise ValueError(
            f"Unknown tool categories: {sorted(unknown)}. Available: "
            f"{sorted(TOOL_CATEGORIES)}."
        )
    return values


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    """Clamp a possibly-bad numeric to ``[lo, hi]``; non-numerics fall back."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _persist_after_write(
    agent: CharacterAgent,
    mem: Any,
    *,
    updated_ids: tuple[int, ...] = (),
    removed_ids: tuple[int, ...] = (),
) -> None:
    """Incrementally refresh an affected index, then persist all memories.

    Adds already update their index through ``StructuredMemory.add``. Updates
    and removals are mirrored here; unsupported RAG backends fall back to one
    full rebuild through ``apply_index_changes``.
    """
    if isinstance(mem, StructuredMemory):
        try:
            mem.apply_index_changes(
                updated_ids=updated_ids,
                removed_ids=removed_ids,
            )
        except Exception as e:
            warnings.warn(
                f"index update for {mem.name!r} failed: {e!r}; the in-RAM "
                f"index is stale but the SQLite row is durable.",
                stacklevel=2,
            )
    try:
        agent.persist_structured()
    except Exception as e:
        warnings.warn(
            f"persist_structured failed after writing to {mem.name!r}: {e!r}; "
            f"the change is in SQLite but not flushed to the on-disk index.",
            stacklevel=2,
        )


def _persist_only(agent: CharacterAgent) -> None:
    """Persist after a write whose in-memory index is already synchronized."""
    try:
        agent.persist_structured()
    except Exception as e:
        warnings.warn(
            f"persist_structured failed in set_user_summary: {e!r}.",
            stacklevel=2,
        )


# --------------------------------------------------------------------------- #
# Read tools
# --------------------------------------------------------------------------- #
def _tool_list_memories(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Sidebar overview: every memory with title, kind, count, known users."""
    return {"character": agent.character_name, "memories": memory_overview(agent)}


def _tool_graph_overview(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """High-level knowledge-graph stats: node/edge counts per kind + users.

    Returns an error envelope when the character does not have the
    `knowledge_graph` memory enabled.
    """
    kg = agent.memories.get("knowledge_graph")
    if not isinstance(kg, KnowledgeGraphMemory):
        return _json_error(
            f"Character {agent.character_name!r} does not have the "
            f"knowledge_graph memory enabled."
        )
    return {"character": agent.character_name, "graph": kg.retriever.overview()}


def _tool_search_knowledge_graph(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Activation search over the knowledge graph.

    Returns the top nodes by activation as records, plus the activation
    subgraph (nodes + edges) so an MCP client can render a visualization.
    `query` drives the spreading-activation search; omit it to get the
    whole graph (capped by `limit`). `user_id` biases the user's PersonNode.
    """
    kg = agent.memories.get("knowledge_graph")
    if not isinstance(kg, KnowledgeGraphMemory):
        return _json_error(
            f"Character {agent.character_name!r} does not have the "
            f"knowledge_graph memory enabled."
        )
    q = str(_args(args, "query", "") or "").strip() or None
    user = _args(args, "user_id", None)
    limit = max(1, min(300, int(_args(args, "limit", 50) or 50)))
    hops = max(0, min(2, int(_args(args, "hops", 1) or 1)))
    try:
        return read_graph(agent, q=q, user=user, limit=limit, hops_subgraph=hops)
    except KeyError:
        return _json_error("knowledge_graph memory not built.")


def _run_provider_tool(
    tool_type: type[Tool], agent: CharacterAgent, args: dict, *, bound: bool
) -> dict:
    """Run a provider-neutral library tool through the MCP adapter."""
    tool = tool_type(agent) if bound else tool_type()
    result = tool.run(**args)
    if not isinstance(result, dict):
        return {"result": result}
    return result


def _tool_deduplicate_knowledge_graph(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """One-time person/alias dedup of an already-built knowledge graph.

    Collapses any ``PersonNode`` that is actually the character (by name or
    any declared alias) into the singular ``self`` node, then folds
    ``PersonNode``s that share a name/alias into one survivor — edges are
    rewired so no relationship is lost. Use this to clean up a graph built
    before the self-dedup existed, or after editing the character's
    ``aliases``. No LLM cost; the result is persisted.
    """
    kg = agent.memories.get("knowledge_graph")
    if not isinstance(kg, KnowledgeGraphMemory):
        return _json_error(
            f"Character {agent.character_name!r} does not have the "
            f"knowledge_graph memory enabled."
        )
    before = kg.retriever.overview()
    report = kg.retriever.deduplicate_persons()
    try:
        kg.persist(os.path.join(agent.save_directory, "kg_index"))
    except Exception as e:  # noqa: BLE001 - surface persistence failure to the caller
        return _json_error(f"dedup ran but persist failed: {e!r}")
    after = kg.retriever.overview()
    return {
        "character": agent.character_name,
        "before": before,
        "after": after,
        "report": report,
    }


def _tool_refresh_memory(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Force an immediate reload of this character's in-RAM memory caches.

    The background sync poller (``CM_SYNC_INTERVAL``, default 3 s) already
    picks up writes from other processes (Discord bot, CLI, …), but this tool
    lets an MCP client force a sync the instant it knows a write landed — e.g.
    right after driving the bot — rather than waiting up to ``interval``
    seconds for the next tick. Returns ``reloaded: true`` when a reload ran,
    ``false`` when the on-disk state was already current. No-op (and still
    ``200`` / ``reloaded: false``) when the poller is disabled
    (``CM_SYNC_INTERVAL=0``) — in that mode the caller relies on this tool as
    the only sync path.
    """
    monitor = _SYNC_MONITORS_REF[0].get(agent.character_name)
    reloaded = bool(monitor.check_now()) if monitor is not None else False
    return {"character": agent.character_name, "reloaded": reloaded}


def _tool_search_memory(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Search one memory by query (or page all rows when ``query`` is omitted).

    * ``user_id`` narrows to one participant (per-user memories only).
    * ``date_from`` / ``date_to`` (ISO 8601) bound on the row's
      ``created_at``; honoured for ``user_facts`` / ``user_directives`` /
      ``episodic`` / ``heartbeat`` / ``user_summary`` and silently ignored
      for ``character_info`` / ``dialogue_style`` / ``emotion`` (which
      don't carry a timestamp).
    * ``limit`` caps the page size (1..200, default 25).
    """
    page_in = int(_args(args, "page", 1) or 1)
    size_in = int(_args(args, "limit", 25) or 25)
    page_in = max(1, page_in)
    size_in = max(1, min(200, size_in))
    q = str(_args(args, "query", "") or "").strip()
    user = _args(args, "user_id", None)

    date_from = _parse_iso(_args(args, "date_from", None))
    date_to = _parse_iso(_args(args, "date_to", None))
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from must not be after date_to.")

    # ``read_memory`` uses a half-open upper bound. Move an explicitly
    # inclusive ``date_to`` to the next representable float so the public
    # range semantics remain unchanged.
    date_before = math.nextafter(date_to, math.inf) if date_to is not None else None

    page = read_memory(
        agent,
        mem.name,
        page=page_in,
        size=size_in,
        user=user,
        q=q or None,
        created_from=date_from if isinstance(mem, StructuredMemory) else None,
        created_before=date_before if isinstance(mem, StructuredMemory) else None,
    )

    page["date_from"] = _args(args, "date_from", None)
    page["date_to"] = _args(args, "date_to", None)
    return page


def _tool_search_memory_by_date(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Search rows created on one calendar date in an IANA timezone."""
    if (
        not isinstance(mem, StructuredMemory)
        or "created_at" not in mem.store.columns(mem.table)
    ):
        raise ValueError(
            f"Memory {mem.name!r} does not store creation dates; choose a structured memory."
        )

    timezone_name = str(_args(args, "timezone", "UTC") or "UTC").strip() or "UTC"
    created_from, created_before, from_iso, before_iso = _calendar_day_bounds(
        _args(args, "date", None), timezone_name
    )
    page_in = max(1, int(_args(args, "page", 1) or 1))
    size_in = max(1, min(200, int(_args(args, "limit", 25) or 25)))
    q = str(_args(args, "query", "") or "").strip()
    user = _args(args, "user_id", None)

    page = read_memory(
        agent,
        mem.name,
        page=page_in,
        size=size_in,
        user=user,
        q=q or None,
        created_from=created_from,
        created_before=created_before,
    )
    page["date"] = str(_args(args, "date", ""))
    page["timezone"] = timezone_name
    page["date_range"] = {"from": from_iso, "before": before_iso}
    return page


def _heartbeat_page(agent: CharacterAgent, *, query: str = "", limit: Any = 10) -> dict:
    """Read heartbeat reports through the shared GUI/MCP adapter."""
    mem = agent.memories.get("heartbeat")
    if not isinstance(mem, StructuredMemory) or mem.name != "heartbeat":
        return _json_error(
            f"Character {agent.character_name!r} does not have heartbeat memory."
        )
    size = max(1, min(200, int(limit or 10)))
    return read_memory(
        agent,
        "heartbeat",
        page=1,
        size=size,
        q=query or None,
    )


def _tool_list_heartbeats(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """List the latest heartbeat reports, newest first."""
    return _heartbeat_page(agent, limit=_args(args, "limit", 10))


def _tool_search_heartbeats(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Hybrid-search heartbeat reports without changing recall counters."""
    query = str(_args(args, "query", "") or "").strip()
    if not query:
        return _json_error("query is required.")
    return _heartbeat_page(
        agent,
        query=query,
        limit=_args(args, "limit", 10),
    )


def _world(agent: CharacterAgent) -> WorldMemory:
    world = agent.world_memory
    if world is None or not world.enabled:
        raise ValueError(f"Character {agent.character_name!r} does not have WorldMemory enabled.")
    return world


def _tool_get_world_state(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    return _world(agent).snapshot(commit=False).to_dict()


def _tool_search_world_records(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    query = str(_args(args, "query", "") or "").strip()
    if not query:
        raise ValueError("query is required.")
    return {"records": _world(agent).search_visible(query, limit=int(_args(args, "limit", 4)))}


def _tool_list_world_events(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    return {"events": _world(agent).list_events(limit=int(_args(args, "limit", 100)))}


def _tool_advance_world(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    world = _world(agent)
    events = world.advance(_args(args, "now", None))
    agent.persist_structured()
    return {"events": [vars(event) for event in events], "snapshot": world.snapshot().to_dict()}


def _tool_apply_world_command(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    world = _world(agent)
    values = dict(args)
    values.setdefault("actor_id", world.observer_id)
    values.setdefault("source", "mcp")
    event = world.apply_command(world.validate_command_dict(values))
    agent.persist_structured()
    return {"event": vars(event), "snapshot": world.snapshot().to_dict()}


def _tool_upsert_world_fact(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    world = _world(agent)
    row_id = world.add_fact(
        str(_args(args, "content", "") or ""),
        subject_id=_args(args, "subject_id", None),
        location_id=_args(args, "location_id", None),
        visibility=str(_args(args, "visibility", "known")),
        importance=_clip(_args(args, "importance", 0.6)),
        source="mcp",
        dedupe_key=_args(args, "dedupe_key", None),
    )
    agent.persist_structured()
    return {"id": row_id}


def _tool_set_world_features(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    world = _world(agent)
    values = _args(args, "values", {})
    if not isinstance(values, dict):
        raise ValueError("values must be an object of feature -> true/false/null.")
    resolved = world.set_features(values, actor_id=_args(args, "actor_id", None))
    agent.persist_structured()
    return {"resolved_features": resolved, "actor_id": _args(args, "actor_id", None)}


# --------------------------------------------------------------------------- #
# Write tools — one set per memory type
# --------------------------------------------------------------------------- #
# Each handler asserts that ``mem`` matches the expected backend, so a
# generic ``memory="user_facts"`` argument on ``add_fact`` is enforced at
# the type boundary (rather than silently writing to the wrong table).

def _tool_add_fact(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_facts":
        return _json_error(f"add_fact only writes to user_facts; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    content = str(_args(args, "content", "") or "").strip()
    if not user_id or not content:
        return _json_error("user_id and content are required.")
    row_id = mem.add_fact(
        user_id=user_id, content=content,
        type=str(_args(args, "type", "general")),
        importance=_clip(_args(args, "importance", 0.5)),
        confidence=_clip(_args(args, "confidence", 0.5)),
    )
    _persist_after_write(agent, mem)
    return {"id": row_id, "memory": mem.name}


def _tool_update_fact(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_facts":
        return _json_error(f"update_fact only writes to user_facts; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    row = mem.get_row(row_id)
    if row is None:
        return _json_error(f"No fact with id={row_id}.")
    if _args(args, "content", None):
        row["content"] = str(args["content"]).strip()
    if _args(args, "type", None):
        row["type"] = str(args["type"])
    if _args(args, "importance", None) is not None:
        row["importance"] = _clip(args["importance"])
    if _args(args, "confidence", None) is not None:
        row["confidence"] = _clip(args["confidence"])
    mem.update_row(row)
    _persist_after_write(agent, mem, updated_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "updated": True}


def _tool_delete_fact(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_facts":
        return _json_error(f"delete_fact only writes to user_facts; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    if mem.get_row(row_id) is None:
        return _json_error(f"No fact with id={row_id}.")
    mem.delete_row(row_id)
    _persist_after_write(agent, mem, removed_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "deleted": True}


def _tool_add_directive(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_directives":
        return _json_error(f"add_directive only writes to user_directives; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    content = str(_args(args, "content", "") or "").strip()
    if not user_id or not content:
        return _json_error("user_id and content are required.")
    keywords = _args(args, "keywords", None) or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    row_id = mem.add_directive(
        user_id=user_id, content=content,
        importance=_clip(_args(args, "importance", 0.5)),
        retrieval_keywords=list(keywords),
    )
    _persist_after_write(agent, mem)
    return {"id": row_id, "memory": mem.name}


def _tool_update_directive(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_directives":
        return _json_error(f"update_directive only writes to user_directives; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    row = mem.get_row(row_id)
    if row is None:
        return _json_error(f"No directive with id={row_id}.")
    if _args(args, "content", None):
        row["content"] = str(args["content"]).strip()
    if _args(args, "importance", None) is not None:
        row["importance"] = _clip(args["importance"])
    if _args(args, "keywords", None) is not None:
        kws = args["keywords"]
        if isinstance(kws, str):
            kws = [k.strip() for k in kws.split(",") if k.strip()]
        row["retrieval_keywords"] = json.dumps(list(kws), ensure_ascii=False)
    mem.update_row(row)
    _persist_after_write(agent, mem, updated_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "updated": True}


def _tool_delete_directive(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_directives":
        return _json_error(f"delete_directive only writes to user_directives; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    if mem.get_row(row_id) is None:
        return _json_error(f"No directive with id={row_id}.")
    mem.delete_row(row_id)
    _persist_after_write(agent, mem, removed_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "deleted": True}


def _tool_add_episode(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "episodic":
        return _json_error(f"add_episode only writes to episodic; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    summary = str(_args(args, "summary", "") or "").strip()
    if not user_id or not summary:
        return _json_error("user_id and summary are required.")
    row_id = mem.add_episode(
        user_id=user_id, summary=summary,
        importance=_clip(_args(args, "importance", 0.5)),
        emotional_shift=emotion_vector(
            _args(args, "emotional_shift", {}),
            allowed_axes=getattr(mem, "emotion_baseline", None),
        ),
    )
    _persist_after_write(agent, mem)
    return {"id": row_id, "memory": mem.name}


def _tool_update_episode(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "episodic":
        return _json_error(f"update_episode only writes to episodic; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    row = mem.get_row(row_id)
    if row is None:
        return _json_error(f"No episode with id={row_id}.")
    if _args(args, "summary", None):
        row["summary"] = str(args["summary"]).strip()
    if _args(args, "importance", None) is not None:
        row["importance"] = _clip(args["importance"])
    if _args(args, "emotional_shift", None) is not None:
        row["emotional_shift"] = encode_emotion_vector(
            args["emotional_shift"],
            allowed_axes=getattr(mem, "emotion_baseline", None),
        )
    mem.update_row(row)
    _persist_after_write(agent, mem, updated_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "updated": True}


def _tool_delete_episode(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "episodic":
        return _json_error(f"delete_episode only writes to episodic; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    if mem.get_row(row_id) is None:
        return _json_error(f"No episode with id={row_id}.")
    mem.delete_row(row_id)
    _persist_after_write(agent, mem, removed_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "deleted": True}


def _tool_add_heartbeat(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "heartbeat":
        return _json_error(f"add_heartbeat only writes to heartbeat; this memory is {mem.name!r}.")
    summary = str(_args(args, "summary", "") or "").strip()
    if not summary:
        return _json_error("summary is required.")
    row_id = mem.add_entry(
        summary=summary,
        kind=str(_args(args, "kind", "discovery")),
        importance=_clip(_args(args, "importance", 0.5)),
    )
    _persist_after_write(agent, mem)
    return {"id": row_id, "memory": mem.name}


def _tool_update_heartbeat(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "heartbeat":
        return _json_error(f"update_heartbeat only writes to heartbeat; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    row = mem.get_row(row_id)
    if row is None:
        return _json_error(f"No heartbeat entry with id={row_id}.")
    if _args(args, "summary", None):
        row["summary"] = str(args["summary"]).strip()
    if _args(args, "kind", None):
        row["kind"] = str(args["kind"])
    if _args(args, "importance", None) is not None:
        row["importance"] = _clip(args["importance"])
    mem.update_row(row)
    _persist_after_write(agent, mem, updated_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "updated": True}


def _tool_delete_heartbeat(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "heartbeat":
        return _json_error(f"delete_heartbeat only writes to heartbeat; this memory is {mem.name!r}.")
    row_id = _args(args, "id", None)
    if row_id is None:
        return _json_error("id is required.")
    row_id = int(row_id)
    if mem.get_row(row_id) is None:
        return _json_error(f"No heartbeat entry with id={row_id}.")
    mem.delete_row(row_id)
    _persist_after_write(agent, mem, removed_ids=(row_id,))
    return {"id": row_id, "memory": mem.name, "deleted": True}


def _tool_set_user_summary(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, StructuredMemory) or mem.name != "user_summary":
        return _json_error(f"set_user_summary only writes to user_summary; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    summary = str(_args(args, "summary", "") or "").strip()
    if not user_id or not summary:
        return _json_error("user_id and summary are required.")
    aliases = _args(args, "aliases", None) or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    row_id = mem.add_or_update(
        user_id=user_id,
        name=str(_args(args, "name", user_id)).strip(),
        aliases=list(aliases),
        summary=summary,
        importance=_clip(_args(args, "importance", 1.0), default=1.0),
    )
    # ``add_or_update`` already synchronized this profile's hybrid entry; just
    # flush the structured indexes to disk so the change survives a restart.
    _persist_only(agent)
    return {"id": row_id, "memory": mem.name, "user_id": user_id}


def _tool_set_user_emotion(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, EmotionStatus):
        return _json_error(f"set_user_emotion only writes to emotion; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    if not user_id:
        return _json_error("user_id is required.")
    deltas = _args(args, "deltas", None)
    comment = _args(args, "comment", None)
    current_mood = _args(args, "current_mood", None)
    if isinstance(deltas, str):
        try:
            deltas = json.loads(deltas)
        except (TypeError, ValueError):
            return _json_error(f"deltas must be a JSON object, got {deltas!r}.")
    if isinstance(deltas, dict) and deltas:
        mem.update(user_id, {k: float(v) for k, v in deltas.items() if k in mem.user_dims})
    if isinstance(comment, str) and comment.strip():
        mem.set_user_comment(user_id, comment)
    if current_mood is not None:
        mem.set_current_mood(current_mood)
    return {
        "user_id": user_id,
        "state": mem.get_user_state(user_id),
        "current_mood": mem.get_current_mood(),
        "comment": mem.get_user_comment(user_id),
    }


def _tool_get_user_emotion(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    if not isinstance(mem, EmotionStatus):
        return _json_error(f"get_user_emotion only reads from emotion; this memory is {mem.name!r}.")
    user_id = _args(args, "user_id", None) or ""
    if not user_id:
        return _json_error("user_id is required.")
    return {
        "user_id": user_id,
        "baseline": dict(mem.baseline),
        "current_mood": mem.get_current_mood(),
        "state": mem.get_user_state(user_id),
        "comment": mem.get_user_comment(user_id),
    }


def _character_emotion(agent: CharacterAgent) -> EmotionStatus | dict:
    """Return the character-wide emotion memory or an MCP error result."""
    mem = agent.memories.get("emotion")
    if not isinstance(mem, EmotionStatus):
        return _json_error(
            f"Character {agent.character_name!r} does not have emotion memory."
        )
    return mem


def _tool_set_character_emotion(
    agent: CharacterAgent, mem: Any, args: dict
) -> dict:
    """Replace the character's persisted current mood."""
    emotion = _character_emotion(agent)
    if isinstance(emotion, dict):
        return emotion
    current_mood = _args(args, "current_mood", None)
    if current_mood is None:
        return _json_error("current_mood is required.")
    updated = emotion.set_current_mood(current_mood)
    return {
        "character": agent.character_name,
        "baseline": dict(emotion.baseline),
        "current_mood": updated,
    }


def _tool_get_character_emotion(
    agent: CharacterAgent, mem: Any, args: dict
) -> dict:
    """Return the character's resting baseline and persisted current mood."""
    emotion = _character_emotion(agent)
    if isinstance(emotion, dict):
        return emotion
    return {
        "character": agent.character_name,
        "baseline": dict(emotion.baseline),
        "current_mood": emotion.get_current_mood(),
    }


def _tool_add_character_info(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Append a chunk to ``character_info`` (RAG).

    Caveat: ``character_info`` is re-built from
    ``<character>/Information/*.md`` at agent startup, so anything you
    append here is **session-scoped** — it will be overwritten on the next
    ``agent.rebuild()`` unless you also drop a markdown file into the
    character directory. Persistent edits belong on disk.
    """
    if not isinstance(mem, RAGMemory) or mem.name != "character_info":
        return _json_error(f"add_character_info only writes to character_info; this memory is {mem.name!r}.")
    text = str(_args(args, "text", "") or "").strip()
    if not text:
        return _json_error("text is required.")
    mem.hybrid.add_documents(
        [Chunk(text=text, source=str(_args(args, "source", "mcp")), metadata={})]
    )
    try:
        mem.persist(os.path.join(agent.save_directory, "info_index"))
    except Exception as e:
        warnings.warn(
            f"persist for {mem.name!r} failed: {e!r}; the new chunk is in "
            f"the in-RAM index but not flushed to disk.",
            stacklevel=2,
        )
    return {"memory": mem.name, "added": 1, "total": mem.hybrid.count}


def _tool_add_dialogue(agent: CharacterAgent, mem: Any, args: dict) -> dict:
    """Append a chunk to ``dialogue_style`` (RAG). Same persistence caveat
    as :func:`_tool_add_character_info` — overwritten on a fresh
    ``agent.rebuild()`` from ``<character>/Dialogues/*.md``.
    """
    if not isinstance(mem, RAGMemory) or mem.name != "dialogue_style":
        return _json_error(f"add_dialogue only writes to dialogue_style; this memory is {mem.name!r}.")
    text = str(_args(args, "text", "") or "").strip()
    if not text:
        return _json_error("text is required.")
    mem.hybrid.add_documents(
        [Chunk(text=text, source=str(_args(args, "source", "mcp")), metadata={})]
    )
    try:
        mem.persist(os.path.join(agent.save_directory, "dialogue_index"))
    except Exception as e:
        warnings.warn(
            f"persist for {mem.name!r} failed: {e!r}; the new chunk is in "
            f"the in-RAM index but not flushed to disk.",
            stacklevel=2,
        )
    return {"memory": mem.name, "added": 1, "total": mem.hybrid.count}


# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #
# Scope ``_CHAR`` means the tool takes no ``memory`` argument (only the
# character-level agent). All others require a matching ``memory`` from the
# agent's memory dict.

_CHAR = "__character__"
_TOOLS: list[dict[str, Any]] = []
_HANDLER_INFO: dict[str, tuple[Callable[..., dict], bool]] = {}
# Live reference to the per-character :class:`MemorySync` map, set by
# :func:`build_router`. Stored in a one-element list so :func:`build_router`
# can rebind it without ``global``. The ``refresh_memory`` tool reads it to
# force an immediate cache reload.
_SYNC_MONITORS_REF: list[dict[str, MemorySync]] = [{}]


def _register(
    name: str,
    description: str,
    input_schema: dict,
    handler: Callable[..., dict],
    *,
    needs_memory: bool = True,
    categories: tuple[str, ...] = ("memory",),
) -> None:
    """Add a tool: ``handler`` gets ``(agent, mem, args)`` and returns a dict.

    ``needs_memory=False`` registers a character-level or provider-neutral
    tool; the dispatcher's explicit memory-lookup is skipped for it.
    """
    category_set = frozenset(categories)
    unknown = category_set - TOOL_CATEGORIES
    if not category_set or unknown:
        raise ValueError(
            f"Tool {name!r} has invalid categories {sorted(category_set)}; "
            f"available categories are {sorted(TOOL_CATEGORIES)}."
        )
    _TOOLS.append(
        {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "categories": category_set,
        }
    )
    _HANDLER_INFO[name] = (handler, needs_memory)


def _register_provider_tool(
    tool_type: type[Tool], *, bound: bool, categories: tuple[str, ...]
) -> None:
    """Expose one provider-neutral library tool through MCP unchanged."""

    def handler(agent: CharacterAgent, mem: Any, args: dict) -> dict:
        return _run_provider_tool(tool_type, agent, args, bound=bound)

    _register(
        tool_type.name,
        tool_type.description,
        tool_type.parameters,
        handler,
        needs_memory=False,
        categories=categories,
    )


# Read ---- ---------------------------------------------------------------- #
_register(
    "list_memories",
    "List every memory on the bound character with its title, kind, "
    "record count, and known user_ids (sidebar overview).",
    {"type": "object", "properties": {}, "required": []},
    _tool_list_memories,
    needs_memory=False,
    categories=("core",),
)

_register(
    "refresh_memory",
    "Force an immediate reload of this character's in-RAM memory caches "
    "(structured hybrid indexes + the knowledge graph). Call this when "
    "another process using the same character (e.g. the Discord bot, the "
    "CLI) has just written memories and you need search/retrieval to "
    "reflect them without waiting for the background sync poller "
    "(CM_SYNC_INTERVAL, default 3 s). Returns ``reloaded`` (true when a "
    "reload ran, false when the on-disk state was already current).",
    {"type": "object", "properties": {}, "required": []},
    _tool_refresh_memory,
    needs_memory=False,
    categories=("core",),
)

_register(
    "graph_overview",
    "High-level knowledge-graph stats for the character: total nodes, edges, "
    "per-kind counts, and known users. Errors if the character does not have "
    "the knowledge_graph memory enabled.",
    {"type": "object", "properties": {}, "required": []},
    _tool_graph_overview,
    needs_memory=False,
    categories=("kg",),
)

_register(
    "search_knowledge_graph",
    "Activation search over the character's knowledge graph. Returns the top "
    "nodes by activation (records) plus the activation-weighted subgraph "
    "(nodes + edges) so a client can render it. `query` drives spreading "
    "activation; omit it for the whole graph. `user_id` biases that user's "
    "PersonNode. `limit` caps nodes (1..300, default 50); `hops` (0..2) "
    "expands the returned subgraph around the top nodes.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 300, "default": 50},
            "hops": {"type": "integer", "minimum": 0, "maximum": 2, "default": 1},
        },
        "required": [],
    },
    _tool_search_knowledge_graph,
    needs_memory=False,
    categories=("kg",),
)

_register(
    "deduplicate_knowledge_graph",
    "One-time person/alias dedup of an already-built knowledge graph. Collapses "
    "any PersonNode that is actually the character (by name or any declared "
    "alias) into the singular 'self' node, then folds PersonNodes that share a "
    "name/alias into one survivor (edges are rewired, nothing is lost). Use to "
    "clean up a graph built before the self-dedup existed, or after editing the "
    "character's aliases. No LLM cost; the result is persisted. Returns before/"
    "after node counts and the merge report.",
    {"type": "object", "properties": {}, "required": []},
    _tool_deduplicate_knowledge_graph,
    needs_memory=False,
    categories=("kg",),
)

# Immutable conversation-event retrieval. These registrations reuse the same
# provider-neutral classes as CharacterAgent's internal function-calling loop.
_register_provider_tool(
    SearchConversationEvents, bound=True, categories=("events",)
)
_register_provider_tool(GetConversationEvents, bound=True, categories=("events",))
_register_provider_tool(GetEventNeighbors, bound=True, categories=("events",))

# Focused graph inspection complements the existing visualization-oriented
# search_knowledge_graph MCP tool above.
_register_provider_tool(GetKnowledgeGraphNodes, bound=True, categories=("kg",))
_register_provider_tool(GetKnowledgeGraphNeighbors, bound=True, categories=("kg",))

# Time arithmetic is useful in both source-event and graph retrieval flows.
_register_provider_tool(
    ResolveTimeRange, bound=False, categories=("events", "kg")
)
_register_provider_tool(
    CalculateTimeDifference, bound=False, categories=("events", "kg")
)

_register(
    "search_memory",
    "Search one memory for the given query. Omit `query` to page all rows. "
    "`user_id` narrows the result to one participant when the memory is "
    "per-user. `date_from` and `date_to` (ISO 8601 timestamps) bound on the "
    "row's `created_at` and apply only to memories that carry one "
    "(user_facts, user_directives, episodic, heartbeat, user_summary); "
    "they are silently ignored for the others (character_info, "
    "dialogue_style, emotion). `limit` caps the page size (1..200, default 25).",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "date_from": {"type": "string", "description": "ISO 8601 (e.g. 2024-01-15T00:00:00Z). Lower bound on created_at."},
            "date_to": {"type": "string", "description": "ISO 8601 (e.g. 2024-02-01T23:59:59Z). Upper bound on created_at."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
            "page": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["memory"],
    },
    _tool_search_memory,
)

_register(
    "search_memory_by_date",
    "Search one structured memory for records created on a specific calendar "
    "date. `date` uses YYYY-MM-DD. `timezone` is an IANA timezone (default "
    "UTC), so the day boundary remains correct across daylight-saving changes. "
    "Optional `query` and `user_id` narrow the results; `limit` caps the page "
    "size (1..200, default 25).",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "date": {
                "type": "string",
                "format": "date",
                "description": "Calendar date in YYYY-MM-DD format.",
            },
            "timezone": {
                "type": "string",
                "default": "UTC",
                "description": "IANA timezone, e.g. UTC or Europe/Rome.",
            },
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
            "page": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["memory", "date"],
    },
    _tool_search_memory_by_date,
)

# user_facts ---- ---------------------------------------------------------- #
_register(
    "add_fact",
    "Add a fact about a user to `user_facts`. `importance` (0..1, default 0.5) "
    "sets the base importance — facts at or above the agent's "
    "`sticky_threshold` are always in the prompt. `confidence` (0..1) is "
    "the model's certainty. `type` is a free-form label (e.g. \"preference\", "
    "\"occupation\").",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "user_id": {"type": "string"},
            "content": {"type": "string"},
            "type": {"type": "string", "default": "general"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        },
        "required": ["memory", "user_id", "content"],
    },
    _tool_add_fact,
)
_register(
    "update_fact",
    "Partially update a `user_facts` row by id. Only the fields you supply are touched.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "id": {"type": "integer"},
            "content": {"type": "string"},
            "type": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["memory", "id"],
    },
    _tool_update_fact,
)
_register(
    "delete_fact",
    "Delete a `user_facts` row by id.",
    {
        "type": "object",
        "properties": {"memory": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["memory", "id"],
    },
    _tool_delete_fact,
)

# user_directives ---- ----------------------------------------------------- #
_register(
    "add_directive",
    "Add a standing instruction to `user_directives`. `keywords` (a list of "
    "terms, or a comma-separated string) boost recall when they appear in a "
    "user's query, even if the hybrid search would have missed the row.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "user_id": {"type": "string"},
            "content": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["memory", "user_id", "content"],
    },
    _tool_add_directive,
)
_register(
    "update_directive",
    "Partially update a `user_directives` row by id.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "id": {"type": "integer"},
            "content": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["memory", "id"],
    },
    _tool_update_directive,
)
_register(
    "delete_directive",
    "Delete a `user_directives` row by id.",
    {
        "type": "object",
        "properties": {"memory": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["memory", "id"],
    },
    _tool_delete_directive,
)

# episodic ---- ------------------------------------------------------------ #
_register(
    "add_episode",
    "Add a memory of what happened to `episodic`. `emotional_shift` is a "
    "sparse object mapping configured emotion axes to 0..1 intensities. "
    "`importance` (0..1) is the base importance.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "user_id": {"type": "string"},
            "summary": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "emotional_shift": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                "default": {},
            },
        },
        "required": ["memory", "user_id", "summary"],
    },
    _tool_add_episode,
)
_register(
    "update_episode",
    "Partially update an `episodic` row by id.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "id": {"type": "integer"},
            "summary": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "emotional_shift": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "required": ["memory", "id"],
    },
    _tool_update_episode,
)
_register(
    "delete_episode",
    "Delete an `episodic` row by id.",
    {
        "type": "object",
        "properties": {"memory": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["memory", "id"],
    },
    _tool_delete_episode,
)

# heartbeat ---- ----------------------------------------------------------- #
_register(
    "list_heartbeats",
    "List the character's latest heartbeat reports in newest-first order. "
    "`limit` controls how many reports are returned (1..200, default 10).",
    {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 10,
            },
        },
        "required": [],
    },
    _tool_list_heartbeats,
    needs_memory=False,
    categories=("memory", "heartbeat"),
)
_register(
    "search_heartbeats",
    "Semantic + lexical search over the character's heartbeat reports. "
    "Returns the best matching reports without modifying recall counters; "
    "`limit` is 1..200 (default 10).",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 10,
            },
        },
        "required": ["query"],
    },
    _tool_search_heartbeats,
    needs_memory=False,
    categories=("memory", "heartbeat"),
)
_register(
    "add_heartbeat",
    "Add an entry to `heartbeat` (the character's autonomous journal). "
    "`kind` is \"discovery\" (default) or \"action\".",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "summary": {"type": "string"},
            "kind": {"type": "string", "default": "discovery"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        },
        "required": ["memory", "summary"],
    },
    _tool_add_heartbeat,
    categories=("memory", "heartbeat"),
)
_register(
    "update_heartbeat",
    "Partially update a `heartbeat` row by id.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "id": {"type": "integer"},
            "summary": {"type": "string"},
            "kind": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["memory", "id"],
    },
    _tool_update_heartbeat,
    categories=("memory", "heartbeat"),
)
_register(
    "delete_heartbeat",
    "Delete a `heartbeat` row by id.",
    {
        "type": "object",
        "properties": {"memory": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["memory", "id"],
    },
    _tool_delete_heartbeat,
    categories=("memory", "heartbeat"),
)

# user_summary ---- ------------------------------------------------------- #
_register(
    "set_user_summary",
    "Add or merge a per-user profile in `user_summary`. There is exactly one "
    "row per user; new `aliases` are unioned with the stored ones, and "
    "`name` / `summary` overwrite the previous values (the index is rebuilt).",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "user_id": {"type": "string"},
            "name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
        },
        "required": ["memory", "user_id", "summary"],
    },
    _tool_set_user_summary,
)

# emotion ---- ------------------------------------------------------------- #
_register(
    "set_character_emotion",
    "Replace the character's own current mood with an absolute snapshot over "
    "the configured baseline axes. Values are clamped to 0..1; unknown axes "
    "are rejected, and omitted configured axes are reset to zero.",
    {
        "type": "object",
        "properties": {
            "current_mood": {
                "type": "object",
                "description": "Absolute character-wide mood over configured emotion axes.",
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
        "required": ["current_mood"],
    },
    _tool_set_character_emotion,
    needs_memory=False,
)
_register(
    "get_character_emotion",
    "Return the character's configured resting baseline and persisted current mood.",
    {"type": "object", "properties": {}, "required": []},
    _tool_get_character_emotion,
    needs_memory=False,
)
_register(
    "set_user_emotion",
    "Update a user's emotion state in `emotion`. Supply `deltas` (an object "
    "of dim → signed step) to nudge the numeric dimensions, or `comment` to "
    "replace the relationship descriptor. Allowed dims are the agent's "
    "configured per-user dims (default: affection, valence, trust).",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "user_id": {"type": "string"},
            "deltas": {
                "type": "object",
                "description": "Mapping of dim (e.g. affection) → signed step in [-1, 1].",
                "additionalProperties": {"type": "number"},
            },
            "comment": {"type": "string", "description": "Relationship label (e.g. \"colleague\")."},
            "current_mood": {
                "type": "object",
                "description": "Absolute character-wide mood over configured baseline axes.",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "required": ["memory", "user_id"],
    },
    _tool_set_user_emotion,
)
_register(
    "get_user_emotion",
    "Return the user's emotion state (numeric dims + relationship comment) "
    "and the character's baseline and current mood.",
    {
        "type": "object",
        "properties": {"memory": {"type": "string"}, "user_id": {"type": "string"}},
        "required": ["memory", "user_id"],
    },
    _tool_get_user_emotion,
)

# RAG writes (session-scoped — see the comments on the handlers). ---- #
_register(
    "add_character_info",
    "Append a chunk to the `character_info` index. Session-scoped: the index "
    "is normally re-built from <character>/Information/*.md on agent "
    "startup, so writes here are useful for in-session edits but are "
    "overwritten on the next rebuild unless you also drop a markdown file "
    "into the character directory.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "text": {"type": "string"},
            "source": {"type": "string", "default": "mcp"},
        },
        "required": ["memory", "text"],
    },
    _tool_add_character_info,
)
_register(
    "add_dialogue",
    "Append a chunk to the `dialogue_style` index. Same filesystem caveat "
    "as `add_character_info`: overwritten from <character>/Dialogues/*.md "
    "on the next rebuild.",
    {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "text": {"type": "string"},
            "source": {"type": "string", "default": "mcp"},
        },
        "required": ["memory", "text"],
    },
    _tool_add_dialogue,
)

# private world ----------------------------------------------------------- #
_register(
    "get_world_state", "Return the observer-filtered projected world state.",
    {"type": "object", "properties": {}, "required": []},
    _tool_get_world_state, needs_memory=False, categories=("world",),
)
_register(
    "search_world_records", "Search facts and events visible to the observer.",
    {"type": "object", "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 4},
    }, "required": ["query"]},
    _tool_search_world_records, needs_memory=False, categories=("world",),
)
_register(
    "list_world_events", "List public or witnessed world events, newest first.",
    {"type": "object", "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
    }, "required": []},
    _tool_list_world_events, needs_memory=False, categories=("world",),
)
_register(
    "advance_world", "Commit deterministic world simulation through an optional epoch time.",
    {"type": "object", "properties": {"now": {"type": "number"}}, "required": []},
    _tool_advance_world, needs_memory=False, categories=("world",),
)
_register(
    "apply_world_command", "Apply a validated command to a configured actor.",
    {"type": "object", "properties": {
        "kind": {"type": "string", "enum": sorted(["move", "start_activity", "eat", "sleep", "wake", "schedule"])},
        "actor_id": {"type": "string"}, "at": {"type": "number"},
        "location_id": {"type": "string"}, "activity_kind": {"type": "string"},
        "activity": {"type": "string"}, "until": {"type": "number"},
        "payload": {"type": "object"}, "dedupe_key": {"type": "string"},
    }, "required": ["kind"]},
    _tool_apply_world_command, needs_memory=False, categories=("world",),
)
_register(
    "upsert_world_fact", "Add an authored or learned fact to the world ledger.",
    {"type": "object", "properties": {
        "content": {"type": "string"}, "subject_id": {"type": "string"},
        "location_id": {"type": "string"},
        "visibility": {"type": "string", "enum": ["public", "known", "local", "private"]},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "dedupe_key": {"type": "string"},
    }, "required": ["content"]},
    _tool_upsert_world_fact, needs_memory=False, categories=("world",),
)
_register(
    "set_world_features", "Patch world defaults or actor tri-state overrides.",
    {"type": "object", "properties": {
        "actor_id": {"type": "string"},
        "values": {"type": "object", "additionalProperties": {"type": ["boolean", "null"]}},
    }, "required": ["values"]},
    _tool_set_world_features, needs_memory=False, categories=("world",),
)


def _build_schemas(
    agent: CharacterAgent,
    categories: Optional[frozenset[str]] = None,
) -> list[dict[str, Any]]:
    """Clone the registered tools and inject the live memory-name enum.

    The schemas above carry ``"memory": {"type": "string"}`` as a placeholder;
    ``tools/list`` returns these schemas with the real ``enum`` populated so
    LLM introspection surfaces real, schema-valid choices.
    """
    enum = sorted(agent.memories.keys())
    out = []
    for tool in _TOOLS:
        if categories is not None and not (tool["categories"] & categories):
            continue
        schema = {**tool["inputSchema"]}
        props = dict(schema.get("properties", {}))
        if "memory" in props:
            props["memory"] = {**props["memory"], "enum": enum}
        schema["properties"] = props
        out.append({"name": tool["name"], "description": tool["description"], "inputSchema": schema})
    return out


# --------------------------------------------------------------------------- #
# JSON-RPC 2.0 dispatcher
# --------------------------------------------------------------------------- #
# We speak a small subset of MCP's Streamable HTTP transport:
# ``initialize``, ``tools/list``, ``tools/call``, ``ping``, and the
# ``notifications/initialized`` ack. Errors flow through ``_RpcError`` →
# JSON-RPC error envelopes; tool failures become ``isError: true`` payloads.

class _RpcError(Exception):
    """A JSON-RPC error to send back to the client."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _dispatch(
    payload: Any,
    agent: CharacterAgent,
    categories: Optional[frozenset[str]] = None,
) -> Optional[dict[str, Any]]:
    """Process a single parsed JSON-RPC envelope; return the response,
    or ``None`` for notifications (the caller returns no wire response).

    Tool failures are returned as ``_json_error`` (MCP's ``isError: true``
    shape) rather than raised, so a single bad tool call doesn't poison the
    rest of a batch. JSON-RPC-level errors (``_RpcError``) are still raised
    and surfaced at the dispatcher boundary.
    """
    if not isinstance(payload, dict):
        raise _RpcError(-32600, "Invalid Request: root must be an object.")
    if payload.get("jsonrpc") != "2.0":
        raise _RpcError(-32600, 'Invalid Request: jsonrpc must be "2.0".')
    method = payload.get("method")
    if not isinstance(method, str):
        raise _RpcError(-32600, "Invalid Request: method must be a string.")
    has_id = "id" in payload
    params = payload.get("params") or {}

    if has_id and method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {}},
        }
    if method == "notifications/initialized":
        return None
    if has_id and method == "ping":
        return {}
    if has_id and method == "tools/list":
        return {"tools": _build_schemas(agent, categories)}
    if has_id and method == "tools/call":
        return _dispatch_tool_call(params, agent, categories)

    if has_id:
        raise _RpcError(-32601, f"Method not found: {method!r}")
    return None


def _dispatch_tool_call(
    params: Any,
    agent: CharacterAgent,
    categories: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise _RpcError(-32600, "tools/call params must be an object.")
    name = params.get("name")
    args = params.get("arguments") or {}
    if not isinstance(name, str) or not name:
        raise _RpcError(-32602, "tools/call requires a non-empty `name`.")
    if not isinstance(args, dict):
        raise _RpcError(-32602, "tools/call `arguments` must be an object.")
    handler_entry = _HANDLER_INFO.get(name)
    if handler_entry is None:
        raise _RpcError(-32602, f"Unknown tool: {name!r}.")
    if categories is not None:
        tool = next(tool for tool in _TOOLS if tool["name"] == name)
        if not (tool["categories"] & categories):
            raise _RpcError(
                -32602,
                f"Tool {name!r} is not enabled by ?tools="
                f"{','.join(sorted(categories))}.",
            )
    handler, needs_memory = handler_entry
    return _invoke(handler, agent, name, args, needs_memory)


def _invoke(
    handler: Callable[..., dict],
    agent: CharacterAgent,
    tool_name: str,
    args: dict,
    needs_memory: bool,
) -> dict[str, Any]:
    """Resolve the memory backing the tool call, then dispatch.

    ``needs_memory`` (registered alongside the tool) controls whether
    ``args["memory"]`` is required — ``list_memories`` registers with
    ``needs_memory=False`` so the dispatcher path is uniform for every tool.
    Tool failures (missing rows, mismatched memory type, …) are returned as
    ``isError: true`` payloads — only JSON-RPC protocol violations raise
    :class:`_RpcError`.
    """
    if not needs_memory:
        try:
            return _as_tool_result(handler(agent, None, args))
        except (RuntimeError, TypeError, ValueError) as e:
            return _json_error(str(e))

    mem_name = args.get("memory")
    mem = agent.memories.get(mem_name)
    if mem is None:
        return _json_error(
            f"Missing or unknown memory {mem_name!r}. Available: {sorted(agent.memories)}."
        )
    try:
        return _as_tool_result(handler(agent, mem, args))
    except (RuntimeError, TypeError, ValueError) as e:
        return _json_error(str(e))


# --------------------------------------------------------------------------- #
# FastAPI router
# --------------------------------------------------------------------------- #
def build_router(
    agent_registry: dict[str, CharacterAgent],
    sync_monitors: Optional[dict[str, MemorySync]] = None,
) -> APIRouter:
    """Return a FastAPI router that serves the MCP endpoint for this app.

    Mount it on the existing app in :file:`api.py`; the route is ``POST /mcp``
    and reads the bound character from the ``?character=`` query parameter.
    ``?tools=core,memory,heartbeat,events,kg`` optionally restricts the advertised and
    callable tool categories; omit it or use ``tools=all`` for the full set.

    ``sync_monitors`` is the live per-character :class:`MemorySync` map (shared
    with the admin router) so the ``refresh_memory`` tool can force an
    immediate cache reload. Optional for callers that don't run the poller; in
    that case ``refresh_memory`` reports ``reloaded: false``.

    When the host app enables API-key auth (``CM_API_KEY`` / ``--api-key``,
    see :mod:`.auth`), this endpoint is gated like every other route: clients
    pass ``Authorization: Bearer <key>`` (most streamable-HTTP MCP client
    configs have a ``headers`` field) or ``X-API-Key``, or fall back to the
    ``?api_key=`` query parameter. A failed check is a plain HTTP ``401``,
    not a JSON-RPC envelope.
    """
    # Keep a live reference to the monitors dict the tool handler reads. We
    # store the dict itself (not a copy) so monitors added/removed by the
    # admin router (create/delete character) are visible here too.
    _SYNC_MONITORS_REF[0] = sync_monitors or {}
    router = APIRouter()

    @router.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        """Single MCP endpoint: JSON-RPC 2.0 over the Streamable HTTP
        transport, JSON mode. Every request carries ``?character=<name>``.

        Batch requests are supported per JSON-RPC 2.0 (top-level array);
        each element is dispatched independently. Notifications (no ``id``)
        produce no wire response — if a batch was *all* notifications, we
        return ``202 No Content`` per the spec.
        """
        agent, error_response = _require_character(request, agent_registry)
        if error_response is not None:
            return error_response
        try:
            categories = _parse_tool_categories(request.query_params.get("tools"))
        except ValueError as e:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32602, "message": str(e)},
                },
                status_code=400,
            )
        try:
            raw = await request.body()
            payload = json.loads(raw.decode("utf-8") or "null")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}},
                status_code=400,
            )

        batch = payload if isinstance(payload, list) else [payload]
        responses: list[dict] = []
        for item in batch:
            if not isinstance(item, dict):
                # Per JSON-RPC 2.0: invalid batch members return a single
                # envelope back to the client.
                responses.append({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request."}})
                continue
            try:
                result = _dispatch(item, agent, categories)
            except _RpcError as e:
                responses.append({"jsonrpc": "2.0", "id": item.get("id"), "error": {"code": e.code, "message": e.message}})
                continue
            if result is None:
                continue  # notification — no response.
            responses.append({"jsonrpc": "2.0", "id": item.get("id"), "result": result})

        if not responses:
            return Response(status_code=202)
        body = responses if isinstance(payload, list) else responses[0]
        return JSONResponse(body)

    @router.get("/mcp")
    async def mcp_get() -> Response:
        """GET on the MCP endpoint is reserved for the SSE channel of the
        Streamable HTTP transport. This server runs in JSON mode only — the
        client should POST a JSON-RPC 2.0 body. We surface a 405 + JSON
        explanation so the failure mode is human-readable rather than a
        mysterious hang."""
        return JSONResponse(
            {
                "error": (
                    "This MCP server runs in JSON mode. POST a JSON-RPC 2.0 body "
                    "to /mcp?character=<name>[&tools=<categories>]."
                ),
            },
            status_code=405,
        )

    return router


# --------------------------------------------------------------------------- #
# Extension point
# --------------------------------------------------------------------------- #
# To support the SSE channel of MCP's Streamable HTTP transport, wrap the
# ``mcp_post`` body with ``sse_starlette.sse.EventSourceResponse`` when the
# request carries ``Accept: text/event-stream`` and stream JSON-RPC messages
# back as SSE events. All our tools return synchronously today, so JSON mode
# is sufficient; this is the small addition required if a client demands SSE.
