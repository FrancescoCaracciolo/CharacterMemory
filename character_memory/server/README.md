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

Both `CM_ASSETS_DIR` and `CM_SAVE_DIR` are relative to the **current working
directory** (the server has no notion of a repo root once installed).

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

---

## Endpoints

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
- **This is an example server**, not a hardened production server: there is
  no auth, rate limiting, or concurrency control around the underlying SQLite
  store. For multi-worker deployments, give each character a single writer.

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

Keyboard: `/` focuses search, `Esc` clears it.

### `GET /gui`

The HTML page. Static assets (`styles.css`, `app.js`) are served from
`/gui/static/`.

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
