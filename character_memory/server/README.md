# CharacterMemory server (`character_memory.server`)

A small FastAPI app that exposes the `CharacterAgent` over HTTP using a
two-step "thin client" chat flow:

1. **`POST /context`** — load (or create) a chat for a character + user,
   store the user's message, and return the assembled memory context.
2. **`POST /save`** — store the assistant answer the client generated and run
   memory extraction so the character learns from the exchange.

This lets a client use its own LLM/model and only rely on the server for
memory recall and learning. The core library does **not** depend on FastAPI;
`fastapi` and `uvicorn` are pulled in only by the optional `server` extra.

The bundled WebUI at **`/gui`** is also a memory observatory and character
workshop. SQLite-backed memories (`user_facts`, `user_directives`, `episodic`,
`heartbeat`, `user_summary`) and per-user emotion state can be added, edited,
and deleted directly from the Archive view. RAG chunks and the knowledge graph
are intentionally read-only because their source of truth is the character's
files; edit those under Workshop → Files and rebuild the indexes instead.

---

## Install

```bash
pip install charactermemory[server]
```

## Run

```bash
# either the console script:
charactermemory-server

# or explicitly with uvicorn:
uvicorn character_memory.server:app --reload
```

A few flags are available on the console script (the env-var equivalents are
listed below and also work when launching through uvicorn directly):

| Flag                     | Default     | Purpose                                            |
|--------------------------|-------------|----------------------------------------------------|
| `--host HOST`            | `0.0.0.0`   | Bind host.                                         |
| `--port PORT`            | `8000`      | Bind port.                                         |
| `--reload` / `--no-reload` | on        | Toggle uvicorn auto-reload.                        |
| `--rebuild-kg [NAME…]`   | _unset_     | Rebuild a character's KG at startup (see below).   |
| `--api-key KEY`          | _unset_     | Require this API key on every endpoint (see below).|

The server reads its configuration from environment variables (the same ones
the rest of the library uses):

| Variable           | Default            | Purpose                                                  |
|--------------------|--------------------|----------------------------------------------------------|
| `OPENAI_BASE_URL`  | OpenAI API         | Chat-completions endpoint used for extraction.           |
| `OPENAI_API_KEY`   | _empty_            | API key for the above.                                   |
| `OPENAI_MODEL`     | `gpt-5.4-mini`     | Model used for extraction.                               |
| `OPENAI_EMBEDDINGS_BASE_URL` | OpenAI API | Embeddings endpoint.                             |
| `OPENAI_EMBEDDINGS_MODEL`    | `text-embedding-ada-002` | Embeddings model.                         |
| `CM_ASSETS_DIR`    | `./assets`         | Root folder scanned for character subdirectories.        |
| `CM_SAVE_DIR`      | `./.cm_servers`    | Where each character's SQLite store + indexes live.      |
| `CM_REBUILD_KG`    | _empty_            | Comma-separated character names to rebuild at startup, or `all`. |
| `CM_API_KEY`       | _empty_            | Require this API key on every endpoint (comma-separated list allowed). |

Both `CM_ASSETS_DIR` and `CM_SAVE_DIR` are relative to the **current working
directory** (the server has no notion of a repo root once installed).

### API key authentication

Optional, off by default. Set `CM_API_KEY` (env var, `.env`, or the
`--api-key` flag) to require a key on **every** endpoint — the thin-client
API (`/`, `/context`, `/save`), the memory browser and admin endpoints
(`/api/...`), and the MCP endpoint (`POST /mcp`):

```bash
CM_API_KEY=change-me charactermemory-server
# or several keys (any one of them passes — handy for rotating or issuing
# one per client):
CM_API_KEY=gui-key,mcp-key charactermemory-server
# or equivalently:
charactermemory-server --api-key change-me
```

A request presents its key in any of three forms (first one found wins):

| Form                                | Typical client                                    |
|-------------------------------------|---------------------------------------------------|
| `Authorization: Bearer <key>` header | HTTP API clients, MCP clients (`headers` config) |
| `X-API-Key: <key>` header            | the browser GUI's fetches                        |
| `?api_key=<key>` query parameter     | SSE `EventSource` and header-less clients        |

```bash
curl -H 'Authorization: Bearer change-me' http://localhost:8000/
curl 'http://localhost:8000/?api_key=change-me'
```

Details:

- Unauthenticated requests get a plain `401` with a `WWW-Authenticate:
  Bearer` header (what streamable-HTTP MCP clients expect — not a JSON-RPC
  envelope).
- `/gui` and its `/gui/static/*` assets stay **public**: the page shell
  ships in the package and holds no user data. It loads, detects the `401`
  on its first data call, prompts for the key (🔑 button in the top bar)
  and attaches it to every request from then on. The key is stored in the
  browser's `localStorage`.
- The `?api_key=` fallback exists because `EventSource` cannot send
  headers; query strings can end up in access logs, so prefer the headers
  whenever the client supports them.
- MCP clients that support custom headers pass the key like this:
  `{"url": "http://host:8000/mcp?character=Kurisu", "headers": {"Authorization": "Bearer change-me"}}`.
- Keys are compared with `hmac.compare_digest`; this is a shared-secret
  gate for exposing the server beyond localhost, not a hardened
  multi-tenant auth system.

### Characters

At startup the server scans `CM_ASSETS_DIR` and builds one `CharacterAgent`
per subdirectory. Each folder name becomes the character name clients send
in requests. For example:

```
assets/
└── Kurisu/          -> character name "Kurisu"
    ├── Information/
    ├── Dialogues/
    └── ...
```

Use `GET /` to list the characters the server has loaded.

### Rebuilding a knowledge graph

A character that opts into the knowledge-graph memory (via a
`.knowledge_graph` marker file in its folder or `CM_KG_CHARACTERS`) loads its
graph from the persisted `kg_index` on a plain start — it is **not** rebuilt.
To force a fresh build — e.g. after editing the wiki / `Information/*.md`,
changing the extractor, or wiping the store — pass `--rebuild-kg`:

```bash
# rebuild Kurisu's KG at startup, then serve (reload is disabled by default
# so the expensive LLM extraction pass runs exactly once):
charactermemory-server --rebuild-kg kurisu

# rebuild several characters:
charactermemory-server --rebuild-kg Kurisu Mayuri

# rebuild every KG-enabled character:
charactermemory-server --rebuild-kg
```

Names are matched case-insensitively. A character requested via `--rebuild-kg`
that does not have the KG memory enabled is logged and skipped (it is not
silently enabled). Without `--reload`, auto-reload is turned off automatically
when `--rebuild-kg` is used; pass `--reload` explicitly to override that.

The equivalent environment variable is `CM_REBUILD_KG` (comma-separated names,
or `all`), which also works when launching through uvicorn directly.

---

## Endpoints

### Memory browser and editor

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/memories/{character}` | List memory systems, counts, users, and editability. |
| `GET` | `/api/memories/{character}/{memory}` | Page/search records and return the WebUI form schema. |
| `POST` | `/api/memories/{character}/{memory}` | Add a record using `{ "values": { ... } }`. |
| `PUT` | `/api/memories/{character}/{memory}/{record_id}` | Update the submitted fields on a record. |
| `DELETE` | `/api/memories/{character}/{memory}/{record_id}` | Remove a record. |

Structured-memory writes refresh and persist the hybrid retrieval index. The
API returns `405` for derived/read-only memories.

### `GET /`

List available characters.

**Response** `200`
```json
{ "characters": ["Kurisu"] }
```

---

### `POST /context`

Resolve the chat (creating it if `chat_id` is missing or unknown), persist
the user turn, and return the assembled memory context for the character +
user.

**Request body**

| Field       | Type     | Required | Notes                                                        |
|-------------|----------|----------|--------------------------------------------------------------|
| `character` | string   | yes      | A subfolder of `assets/` (e.g. `"Kurisu"`).                  |
| `user`      | string   | yes      | The user this chat belongs to.                               |
| `message`   | string   | yes      | The user's latest message.                                   |
| `chat_id`   | string   | no       | Existing chat id. If absent or unknown, a new chat is created. |

**Response** `200` — `ContextResponse`
```json
{
  "chat_id": "9b3f1c2a4d5e6f708192...",
  "context": {
    "character_info": "## Character Information\n...",
    "dialogue_style": "## Example Exchanges (style reference)\n...",
    "user_facts": "## Known facts about user\n...",
    "user_directives": "## Standing instructions\n...",
    "episodic": "## Past episodes\n...",
    "heartbeat": "## Recent activity\n...",
    "emotion": "## Emotional state\n..."
  }
}
```

The `context` object maps each enabled, non-empty memory name to its
rendered section. Keys that are empty/disabled are omitted. Typical keys:

| Key                | Memory contents                                         |
|--------------------|---------------------------------------------------------|
| `character_info`   | Background lore / wiki passages about the character.    |
| `dialogue_style`   | Example exchanges used as a style reference.            |
| `user_facts`       | Facts the character has learned about this user.        |
| `user_directives`  | Standing instructions the user has given.               |
| `episodic`         | Past episodes with this user.                           |
| `heartbeat`        | Character-scoped recent activity / discoveries.         |
| `emotion`          | Current emotional state vector.                         |

**Errors**

| Status | When                                                                    |
|--------|-------------------------------------------------------------------------|
| `404`  | `character` is unknown, or `chat_id` was supplied but doesn't exist.    |
| `403`  | `chat_id` exists but belongs to a different `user`.                     |

---

### `POST /save`

Persist the assistant answer for a chat and run memory extraction over it.

**Request body**

| Field     | Type   | Required | Notes                                                |
|-----------|--------|----------|------------------------------------------------------|
| `chat_id` | string | yes      | The chat id returned by `/context`.                  |
| `answer`  | string | yes      | The assistant answer to persist.                     |

**Response** `200` — `SaveResponse`
```json
{
  "ok": true,
  "chat_id": "9b3f1c2a4d5e6f708192...",
  "extracted": true
}
```

`extracted` reports whether memory extraction actually fired for this turn
(the agent throttles extraction by `MemoryConfig.extract_interval`, so a
single turn may not always trigger learning). Forcing the client's answer
through this endpoint is what makes the character remember the exchange.

**Errors**

| Status | When                                       |
|--------|--------------------------------------------|
| `404`  | `chat_id` is unknown to any character.     |

---

## Typical client flow

```text
┌────────┐                       ┌────────┐
│ Client │                       │ Server │
└───┬────┘                       └───┬────┘
    │  POST /context                 │
    │  {character, user, message,    │
    │   chat_id?}                    │
    │────────────────────────────────>│  load/create chat,
    │                                │  persist user turn,
    │                                │  build memory context
    │  {chat_id, context}            │
    │<────────────────────────────────│
    │                                │
    │  (call your LLM with `context`)│
    │                                │
    │  POST /save                    │
    │  {chat_id, answer}             │
    │────────────────────────────────>│  persist assistant turn,
    │                                │  run extraction
    │  {ok, chat_id, extracted}      │
    │<────────────────────────────────│
```

### `curl`

```bash
# 1. Get context (creates a chat on the first call).
curl -X POST http://localhost:8000/context \
  -H 'Content-Type: application/json' \
  -d '{
        "character": "Kurisu",
        "user": "michael",
        "message": "Hi, I am Michael, the new lab assistant."
      }'
# -> {"chat_id": "9b3f1c2a...", "context": {...}}

# 2. (Client generates its own answer using `context`.)

# 3. Save the answer so the character learns from it.
curl -X POST http://localhost:8000/save \
  -H 'Content-Type: application/json' \
  -d '{
        "chat_id": "9b3f1c2a...",
        "answer": "Welcome, Michael. Daru mentioned you would be joining."
      }'
# -> {"ok": true, "chat_id": "9b3f1c2a...", "extracted": true}
```

### Python (`requests`)

```python
import requests

base = "http://localhost:8000"
chat_id = None

def turn(message: str) -> str:
    global chat_id
    body = {"character": "Kurisu", "user": "michael", "message": message}
    if chat_id:
        body["chat_id"] = chat_id
    ctx = requests.post(f"{base}/context", json=body).json()
    chat_id = ctx["chat_id"]

    answer = my_llm(ctx["context"], message)  # your own model call

    requests.post(f"{base}/save", json={"chat_id": chat_id, "answer": answer})
    return answer
```

---

## Notes

- **Chat ids are globally unique** UUIDs, so `/save` does not require the
  character name — it locates the chat across all loaded characters.
- **Persistence:** the user message is stored by `/context` and the
  assistant answer by `/save`, so the chat history is durable across
  server restarts. Structured-memory indexes are persisted to
  `CM_SAVE_DIR/<character>/` and flushed on shutdown.
- **This is an example server**, not a hardened production server: auth is
  opt-in via `CM_API_KEY` / `--api-key` (see "API key authentication" above)
  and off by default, and there is no rate limiting or concurrency control
  around the underlying SQLite store. For multi-worker deployments, give each
  character a single writer.

---

## Memory browser GUI

A read-only single-page GUI for browsing what a character has learned. Open it
in a browser at:

```
http://localhost:8000/gui
```

The page is served by the same FastAPI app (no separate frontend server, no
build step) and talks to the JSON endpoints below. It is **modular per memory
type**: each memory (`user_facts`, `episodic`, `emotion`, `character_info`, …)
has its own card renderer, keyed by memory name in `app.js` (`RENDERERS`), with
a per-`kind` fallback (`structured` / `rag` / `emotion` / `generic`) so a new
memory type still renders before it gets a tailored view.

Features:

- **Sidebar** — one entry per memory with a live record count; filter by name.
- **Pagination** — large memories page `size` records at a time (default 25,
  max 100) with windowed page numbers; pages are cached client-side so
  back/forward is instant.
- **Search** — per-memory query box (debounced). Prefers the memory's hybrid
  (semantic) retrieval and **falls back to lexical matching when the embedding
  server is unreachable**, so search still works offline.
- **User filter** — for per-user memories (facts / directives / episodes /
  emotion), narrow to one user.
- **Type-specific views** — confidence/importance/effective bars for facts,
  signed emotional-shift bars for episodes, per-user emotion bars + a baseline
  banner, rendered markdown for wiki chunks, etc.
- Skeleton loaders + request cancellation keep it feeling snappy.
- **Live recall monitor** — open `/gui?tab=live&character=Kurisu` to watch
  successful `POST /context` requests arrive over SSE. Each request shows the
  exact recalled items in collapsible memory drawers and, when the knowledge
  graph contributed, the captured activation field. The monitor keeps the
  latest 25 events per character in memory for the current server session; it
  does not add a second durable chat log.

Keyboard: `/` focuses search, `Esc` clears it.

If the server runs with `CM_API_KEY` set, the GUI prompts for the key on
the first rejected request (or via the 🔑 button in the top bar), stores it
in `localStorage` and attaches it to every request — headers for fetches,
`?api_key=` for the SSE live stream.

### `GET /gui`

The HTML page. Static assets (`styles.css`, `app.js`) are served from
`/gui/static/`.

Use `?tab=live&character=<name>` to deep-link directly to the live recall
monitor. The browser reconnects automatically if the SSE connection drops.

---

### `GET /api/context-events/{character}`

Server-Sent Events stream for the live recall monitor. The stream replays the
current in-memory session history (up to 25 events), then emits one `context`
event after each successful `POST /context`. Events include the request
metadata, the weighted retrieval query, each recalled memory's rendered section
and items, and a capped graph payload built from the activation trace captured
during that same recall. `POST /context` itself remains unchanged.

This broker is process-local and intended for the documented single-worker
server. It is not a durable or multi-worker event transport.

---

### `GET /api/memories/{character}`

Sidebar overview: every memory with its record count and known users.

**Response** `200`
```json
{
  "character": "Kurisu",
  "memories": [
    { "name": "user_facts", "title": "User Facts", "kind": "structured",
      "enabled": true, "count": 31, "users": ["francesco"] },
    { "name": "character_info", "title": "Character Information", "kind": "rag",
      "enabled": true, "count": 93, "users": [] },
    { "name": "emotion", "title": "Emotion", "kind": "emotion",
      "enabled": true, "count": 1, "users": ["francesco"] }
  ]
}
```

**Errors** — `404` if `character` is unknown.

---

### `GET /api/memories/{character}/{memory}`

One page of records for a memory.

**Query params**

| Param  | Type | Default | Notes                                                       |
|--------|------|---------|-------------------------------------------------------------|
| `page` | int  | `1`     | 1-based (`≥ 1`).                                            |
| `size` | int  | `25`    | Page size (`1..100`).                                       |
| `user` | str  | _none_  | Filter to one user id (per-user memories only).             |
| `q`    | str  | _none_  | Search query. Semantic when the embedder is up, else lexical. |

**Response** `200`
```json
{
  "character": "Kurisu", "memory": "user_facts", "title": "User Facts",
  "kind": "structured", "enabled": true,
  "page": 1, "size": 25, "total": 31, "pages": 7,
  "search": false, "query": "", "user": null,
  "users": ["francesco"],
  "records": [
    {
      "id": 1, "user_id": "francesco",
      "text": "[user_name] The user's name is Francesco.",
      "score": 0.47,
      "fields": { "id": 1, "type": "user_name", "content": "…",
                  "confidence": 0.9, "importance": 0.5, "…": "…" },
      "meta": { "effective": 0.47, "importance": 0.5,
                "recall_count": 13, "created_at": 1783757914.7,
                "last_recalled": 1783758429.8 }
    }
  ],
  "extra": {}
}
```

Each `record` carries: `text` (default display string), `fields` (the raw,
type-specific columns a custom renderer can read), `score` (search relevance in
search mode, effective-importance otherwise), and `meta` (decay/recency
bookkeeping for structured memories). For `emotion`, `extra.baseline` holds the
character's resting-state vector.

**Errors**

| Status | When                                            |
|--------|-------------------------------------------------|
| `404`  | `character` or `memory` is unknown.             |
| `422`  | `page` < 1 or `size` outside `1..100`.          |

> Search over the structured/RAG memories ranks up to 200 hits and paginates
> within that window; the embedding server must be reachable for semantic
> search (lexical fallback otherwise).

---

## MCP endpoint

The same FastAPI app exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io)
endpoint at `POST /mcp?character=<name>[&tools=<categories>]` that lets MCP clients
(Claude Desktop, MCP Inspector, the Python `mcp` client, …) **read and
edit** a character's memories. It speaks JSON-RPC 2.0 — the JSON-mode
subset of MCP's Streamable HTTP transport. No new dependency: the
protocol surface we need (`initialize`, `tools/list`, `tools/call`,
`ping`, `notifications/initialized`) is implemented directly on top of
FastAPI in [`mcp.py`](./mcp.py).

When the server runs with `CM_API_KEY` set (see "API key authentication"
above), this endpoint requires the key too: pass an
`Authorization: Bearer <key>` header — most streamable-HTTP MCP clients have
a `headers` config field for it — or `X-API-Key`, or the `?api_key=`
query parameter for header-less clients. A failed check is a plain HTTP
`401`.

### `POST /mcp?character=<name>[&tools=<categories>]`

The query parameter `character` (required) names the `CharacterAgent`
every tool call operates against — it must match a folder scanned by
`CM_ASSETS_DIR` at startup (`GET /` lists them).

The optional `tools` parameter selects a comma-separated union of tool
categories. It filters both `tools/list` and `tools/call`, so a client cannot
invoke a tool hidden by its configured URL. Omit it, or use `tools=all`, for
the complete registry.

| Category | Tools |
|---|---|
| `core` | Memory overview and cache refresh. |
| `memory` | Generic memory search plus structured/RAG read-write operations. |
| `heartbeat` | Heartbeat journal list, search, and CRUD tools. |
| `events` | Immutable event search/fetch/neighbors plus temporal calculations. |
| `kg` | Graph overview/search/deduplication, node fetch/neighbors, plus temporal calculations. |

Tools may belong to more than one category: `resolve_time_range` and
`calculate_time_difference` are included by either `events` or `kg`, and the
heartbeat list/search/CRUD tools are included by both `memory` and `heartbeat` for
backwards compatibility. For example,
`/mcp?character=Kurisu&tools=heartbeat` exposes only the heartbeat journal
tools. Category names are case-insensitive and duplicates are ignored. An
empty or unknown selection returns HTTP 400 with JSON-RPC code `-32602`;
calling a registered tool outside the selected categories returns a normal
JSON-RPC `-32602` error envelope.

The body is a JSON-RPC 2.0 envelope (single object, or an array for
batches). Every successful / failed tool call returns a regular JSON
envelope; the route never returns FastAPI's `{"detail": …}` shape
even on URL-level errors so MCP clients can parse it.

**Envelopes** (single):

```json
{ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
    "name": "search_memory",
    "arguments": { "memory": "user_facts", "query": "coffee", "limit": 5 }
}}
```

```json
{ "jsonrpc": "2.0", "id": 1, "result": {
    "content": [{ "type": "text", "text": "{\"memory\":\"user_facts\",...}" }],
    "isError": false
}}
```

Errors come back either as a **JSON-RPC error envelope** (protocol
problems: `code` is one of `-32700 / -32600 / -32601 / -32602 / -32603`)
or as a **tools/call result with `isError: true`** (tool-level failure:
missing row, wrong memory type, bad timestamp, …).

### Tools

`tools/list` returns the schemas; each tool is precisely typed with a
JSON-schema so LLM introspection surfaces real choices (the `memory`
field is an enum of the loaded agent's memories). New tools are added in
`mcp.py` by one `_register(…)` call — methods + schemas + handlers
register themselves at module import.

| Tool                    | Writes to         | Notes                                                                       |
|-------------------------|-------------------|------------------------------------------------------------------------------|
| `list_memories`         | _character_       | Sidebar overview: every memory with title, kind, count, known users.        |
| `refresh_memory`        | _character_       | Force an immediate cache reload after another process writes.               |
| `search_memory`         | _read-only_       | Optional `query`, `user_id`, ISO-8601 `date_from` / `date_to`, `limit`.     |
| `search_memory_by_date` | _read-only_       | Required calendar `date`; optional IANA `timezone`, `query`, `user_id`, `limit`. |
| `search_conversation_events` | _read-only_ | Search raw immutable events and extracted aliases with optional occurrence bounds. |
| `get_conversation_events` | _read-only_ | Fetch complete events by stable event IDs.                                  |
| `get_event_neighbors`   | _read-only_       | Expand immediately preceding/following events in the same chat.             |
| `resolve_time_range`    | _read-only_       | Resolve relative expressions into half-open occurrence bounds.              |
| `calculate_time_difference` | _read-only_ | Calculate whole elapsed days, weeks, months, or years.                       |
| `graph_overview`        | _read-only_       | KG node/edge counts and known users.                                         |
| `search_knowledge_graph` | _read-only_      | Activation search returning records and a visualization subgraph.           |
| `get_knowledge_graph_nodes` | _read-only_   | Fetch graph nodes by stable node IDs.                                        |
| `get_knowledge_graph_neighbors` | _read-only_ | Expand a node across directly connected typed edges.                       |
| `deduplicate_knowledge_graph` | `knowledge_graph` | Merge duplicate person/self nodes and persist the graph.                |
| `add_fact`              | `user_facts`      | `user_id`, `content`, `type`, `importance`, `confidence`.                  |
| `update_fact`           | `user_facts`      | `id`, plus any subset of fields to overwrite.                                |
| `delete_fact`           | `user_facts`      | `id`.                                                                       |
| `add_directive`         | `user_directives` | `user_id`, `content`, `importance`, `keywords`.                            |
| `update_directive`      | `user_directives` | `id`, plus any subset.                                                       |
| `delete_directive`      | `user_directives` | `id`.                                                                       |
| `add_episode`           | `episodic`        | `user_id`, `summary`, `importance`, sparse vector `emotional_shift` (configured axes, `0..1`). |
| `update_episode`        | `episodic`        | `id`, plus any subset.                                                       |
| `delete_episode`        | `episodic`        | `id`.                                                                       |
| `add_heartbeat`         | `heartbeat`       | `summary`, `kind` (`discovery` \| `action`), `importance`.                  |
| `list_heartbeats`       | _read-only_       | Latest heartbeat reports, newest first; optional `limit`.                    |
| `search_heartbeats`     | _read-only_       | Hybrid search over heartbeat reports; required `query`, optional `limit`.    |
| `update_heartbeat`      | `heartbeat`       | `id`, plus any subset.                                                       |
| `delete_heartbeat`      | `heartbeat`       | `id`.                                                                       |
| `set_user_summary`      | `user_summary`    | `user_id`, `summary`, optional `name`, `aliases`, `importance`.              |
| `set_character_emotion` | `emotion`         | Absolute character-wide `current_mood`; no `memory` or `user_id` required.   |
| `get_character_emotion` | _read-only_       | Character resting baseline and persisted current mood.                       |
| `set_user_emotion`      | `emotion`         | `user_id`, optional signed `deltas`, absolute `current_mood`, and optional `comment`. |
| `get_user_emotion`      | `emotion`         | `user_id`. Returns baseline + current mood + per-user dims + relationship comment. |
| `add_character_info`    | `character_info`  | `text`, optional `source`. Session-scoped — see caveat below.               |
| `add_dialogue`          | `dialogue_style`  | Same caveat as `add_character_info`.                                         |

`search_memory` accepts a `date_from` / `date_to` timestamp range. The
dedicated `search_memory_by_date` tool accepts a calendar date
(`YYYY-MM-DD`) and an optional IANA timezone (default `UTC`), and searches
the half-open local-day range so daylight-saving transitions are handled
correctly. Date filtering is honoured on memories whose rows carry `created_at`
(`user_facts`, `user_directives`, `episodic`, `heartbeat`,
`user_summary`) and silently ignored on the others
(`character_info`, `dialogue_style`, `emotion`). Bad ISO 8601 input
returns `isError: true`; an inverted range
(`date_from > date_to`) returns `isError: true` rather than an empty
result.

### Write plumbing

Every write tool commits through the memory's own primitive
(`Add.Fact`, `episode`, `Directive`, heartbeat `entry`,
`summary.add_or_update`, …) then calls `StructuredMemory.rebuild_index`
so the hybrid retriever ranks against the new text, and
`agent.persist_structured()` so the change survives a server restart.
Skipped when the embedder is down (a warning is emitted) or when the
memory rebuilds itself (`user_summary.add_or_update`).

The `add_character_info` / `add_dialogue` tools append to the in-RAM
hybrid index directly. **Caveat**: these RAG memories are normally
rebuilt from `<character>/Information/*.md` and
`<character>/Dialogues/*.md` on agent startup, so anything appended via
MCP is session-scoped — persistent edits belong in those files on
disk.

### Errors

| Status | When                                                                   |
|--------|-------------------------------------------------------------------------|
| `400`  | Missing `?character=` query parameter.                                  |
| `400`  | Empty/unknown `?tools=` category selection.                              |
| `404`  | Unknown character (one not in the loaded `AGENTS` dict).               |
| `405`  | `GET /mcp` — JSON-mode only; the SSE channel is not implemented.       |
| `400`  | Body is not valid JSON.                                                |
| `422`  | Body parses but the JSON-RPC envelope is malformed.                    |

JSON-RPC-level errors (`method not found`, `invalid params`, …) are
returned in the response body as `{"jsonrpc":"2.0","id":...,"error":...}`
with the standard codes (`-32601`, `-32602`, …). Tool-level failures
(missing row, cross-table guard, bad date) are returned as `tools/call`
results with `isError: true` — see [`mcp.py`](./mcp.py) for the exact
envelope.

### Example: `search_memory` with date filter

```bash
curl -X POST 'http://localhost:8000/mcp?character=Kurisu' \
  -H 'Content-Type: application/json' \
  -d '{
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {
          "name": "search_memory",
          "arguments": {
            "memory": "user_facts",
            "date_from": "2024-01-01T00:00:00Z",
            "date_to":   "2024-12-31T23:59:59Z",
            "limit": 10
          }
        }
      }'
```

### Example: event-only tool surface

The category selection belongs in the MCP client URL and therefore applies to
every request on that connection:

```bash
curl -X POST 'http://localhost:8000/mcp?character=Kurisu&tools=events' \
  -H 'Content-Type: application/json' \
  -d '{
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {
          "name": "search_conversation_events",
          "arguments": {
            "query": "photography workshop",
            "occurred_from": "2024-01-01T00:00:00Z",
            "limit": 8
          }
        }
      }'
```

Use `tools=events,kg` to expose both retrieval families while excluding the
generic memory CRUD tools.

### Example: `search_memory_by_date`

```bash
curl -X POST 'http://localhost:8000/mcp?character=Kurisu' \
  -H 'Content-Type: application/json' \
  -d '{
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {
          "name": "search_memory_by_date",
          "arguments": {
            "memory": "episodic",
            "date": "2024-07-17",
            "timezone": "Europe/Rome",
            "user_id": "francesco",
            "limit": 10
          }
        }
      }'
```

### Example: round-trip add/update/delete

```python
import requests, json

base = "http://localhost:8000"
char = "Kurisu"
def call(method, params):
    return requests.post(
        f"{base}/mcp?character={char}",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=10,
    ).json()

# Add one fact.
add = call("tools/call", {"name": "add_fact", "arguments": {
    "memory": "user_facts",
    "user_id": "michael",
    "content": "Michael is the new lab assistant",
    "type": "occupation",
    "importance": 0.7,
    "confidence": 0.9,
}})
print(add["result"]["content"][0]["text"])
fid = json.loads(add["result"]["content"][0]["text"])["id"]

# Bump its importance.
call("tools/call", {"name": "update_fact", "arguments": {
    "memory": "user_facts", "id": fid, "importance": 0.95,
}})

# And finally remove it.
call("tools/call", {"name": "delete_fact", "arguments": {
    "memory": "user_facts", "id": fid,
}})
```
