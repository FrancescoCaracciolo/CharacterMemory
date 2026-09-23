# `CharacterAgent` & `Character`

`CharacterAgent` is the orchestrator, the only object most callers touch. It
builds the memories, manages persistent chats, assembles prompts, calls the
LLM, persists the conversation, and schedules learning.

`Character` is the lightweight prompt/extraction helper the agent owns: it
turns a query into rendered memory sections, and drives a single extraction
call across the enabled memories. You usually never instantiate it directly.

Both classes live in `character_memory/` (`agent.py`, `character.py`) and are
re-exported from the package top level:

```python
from character_memory import CharacterAgent, Character, Chat
```

---

## `CharacterAgent`

### Construction & loading

The constructor is **cheap**: it stores paths and templates and opens nothing.
You must then call one of the loaders before using the agent.

```python
agent = CharacterAgent(
    directory="assets/Kurisu",        # the character's files (Information/, Dialogues/)
    name="Kurisu",                    # defaults to the folder name
    save_directory="assets/Kurisu/.cm_data",   # default: <directory>/.cm_data
    prompt_config=PromptConfig(),     # optional; config.yaml can supply this too
    persona="A neuroscience researcher…",      # optional; config.yaml can supply this too
    temporal_resolution_engine=my_engine,       # optional engine override
    reranker=my_reranker,             # optional MemoryReranker default
    token_counter=count_model_tokens, # optional text -> int callable
)
```

There are three loaders, all returning `self` for chaining:

| Method | When to use |
|---|---|
| `load_from_config(path_or_config)` | The standard path. Reads a `config.yaml`, an in-memory `CharacterMemoryConfig`, or the legacy positional sub-configs. |
| `load_from_config(LLMConfig(), EmbeddingConfig(), MemoryConfig())` | Same method, original positional-subconfig style. |
| `load(llm, embedder, memories)` | DIY: supply ready backends + a list of memories (e.g. a custom memory). |

`load_from_config` also accepts `llm=` and `embedder=` overrides if you want
to inject custom backends but keep the config-driven memories.

Temporal resolution is configured under
`CharacterMemoryConfig.temporal_resolution`; the default local engine is
built during loading and shared by every memory. A constructor-injected
`temporal_resolution_engine=` overrides engine construction. The agent
resolves once per `build_context`, `render_prompt`, or `generate_answer` call.
See [Automatic temporal resolution](temporal_resolution.md).

### Indexing

| Method | Effect |
|---|---|
| `build()` | **Load-or-build**: load persisted indexes if present, otherwise chunk + build + persist them. Idempotent. |
| `rebuild()` | Force a full re-chunk + re-index, overwriting everything. |
| `rebuild_knowledge_graph()` | Rebuild only the KG (cheap, targeted). Use for a one-off KG refresh. |
| `persist_structured()` | Flush the structured-memory (and KG) indexes to disk. Call after manual writes. |

> The KG is **loaded**, never silently rebuilt, on a plain `build()`. To force
> a (re)build, call `rebuild_knowledge_graph()` or pass `--rebuild-kg` to the
> server.

### Chat management

Conversations are persisted as rows in the shared `SQLiteStore` (tables
`chats` and `messages`). One `Chat` per conversation; one row per turn.

```python
chat = agent.create_chat(user="michael", title="phonewave intro")
chat  = agent.load_chat(chat_id)              # any chat by id (across users)
chats = agent.list_chats(user="michael")      # optionally filter by user
```

A `Chat` lets you add messages (attributed to a speaker), inspect history, and
report participants:

```python
chat.add_message("user", "Hi!", user_id="michael")  # speaker is recorded
chat.add_message("assistant", "Hello, Michael.")
chat.messages()                # openai-style [{role, content}, ...]
chat.participants()            # distinct human speakers (group chat support)
chat.last_user_message()
chat.unextracted()             # rows not yet fed to the extractor
```

### Generation

```python
agent.generate_answer(
    target,                # Chat | chat_id | raw query str | list[dict] messages
    *,
    stream=False,          # -> str | Iterator[str | TextChunk | ToolCallEvent | ToolResultEvent]
    save=True,             # persist the assistant reply + auto-extract
    user_id="default",     # used for raw-query / message-list targets
    tools=None,            # ToolRegistry | Tool | list[…] — opt-in tool calling
    max_tool_iterations=8, # cap on the model↔tool loop
    tool_choice=None,      # OpenAI convention: "auto" | "none" | "required" | {…}
    budget=3000,           # optional global memory cap; omit to inherit config
    reranker=my_reranker,  # optional per-call MemoryReranker override
)
```

**Target forms**

| `target` | Behaviour |
|---|---|
| `Chat` | Uses the chat's last user message, history and participants. `save=True` persists the reply. |
| `str` (chat id) | Loads the chat; behaves like the `Chat` case. |
| `str` (bare query) | One-shot; no history, single-user (`[user_id]`). |
| `list[dict]` messages | Uses the last user message as the query and the list as history. Single-user. |

With `stream=True` the return is an iterator:

- **No tools** → `str` chunks (legacy plain-text streaming).
- **With tools** → a union of `TextChunk`, `ToolCallEvent`, `ToolResultEvent`
  (see [Tools](tools.md)).

**Persistence invariant:** only the final assistant **text** is written to
the `messages` table. Intermediate tool-call / tool-result messages stay
in-memory for the loop and are never persisted, so chat history and extraction
stay clean.

### Prompt inspection (without calling the LLM)

```python
snapshot = agent.recall(target, budget=3000)  # ContextSnapshot
print(snapshot.memory_token_count)
print(snapshot.sections)                    # {section_name: rendered_section}
print(snapshot.recalls)                     # selected items and diagnostics

# Alternative return shapes; each performs its own recall:
sections = agent.build_context(target, budget=3000)
snapshot = agent.build_context_snapshot(target, budget=3000)
full = agent.render_prompt(target, budget=3000)
```

All accept the same `target` forms as `generate_answer`, plus `user_id=`,
`budget=`, and `reranker=` overrides. Omitted budget inherits
`MemoryConfig.token_budget`; `None` removes the cap and `0` omits memories.
The default is unlimited. Choose one context-building method per turn to
avoid repeated retrieval and reinforcement.

`ContextSnapshot.memory_budget` is the effective cap and `memory_token_count`
counts rendered memory sections, including headers and labels. System text,
history, intermediate prompts, and tool messages are outside it. Budgeted
selection records exposure only for selected items. The HTTP `/context`
endpoint accepts a JSON `budget` field but does not expose these two Python
snapshot fields. See [Memory budget & reranking](memory_budget.md) for counting,
configuration, extension contracts, and HTTP examples.

### Learning (extraction)

Extraction normally fires automatically every `extract_interval` user turns.
You can also drive it manually — both are idempotent and resumable (every
message row carries an `extracted` flag).

```python
agent.extract(chat)         # extract just this chat's unprocessed turns
agent.extract()             # extract every unprocessed turn across all chats
agent.dedup("user_facts")   # sweep one memory (or all) for duplicates
agent.dedup()               # sweep every structured memory
```

### Tools

```python
tools = agent.memory_tools()              # built-in read-only memory self-tools
reply = agent.generate_answer(chat, tools=tools)
# or mix your own:
from character_memory import ToolRegistry, tool
reply = agent.generate_answer(chat, tools=tools + [my_tool])
```

See [Tools](tools.md) for defining your own.

### Lifecycle

```python
agent.close()   # persist structured indexes, then close the SQLite store
```

`CharacterAgent` also supports the context-manager protocol pattern manually;
call `close()` in a `finally` block when running long-lived scripts.

---

## `Character`

A `Character` bundles an identity (`character_name`, `base_instruction`) with
the memory systems it can recall from and write to, plus an optional LLM and
`PromptConfig`. The agent builds one in `_wire_character`; you usually don't
instantiate it, but its API is public.

```python
from character_memory import Character
char = Character(
    character_name="Kurisu",
    base_instruction="A neuroscience researcher…",
    memories=[...],          # the agent's enabled memories
    llm=llm,                 # used for extraction
    prompts=PromptConfig(),
    budget=3000,             # optional default; None is unlimited
    reranker=my_reranker,    # optional default selection strategy
    token_counter=count_model_tokens,  # optional; cl100k_base by default
)
```

What it does for the agent:

- **`recall(query, user_id="default", *, limits=None, participants=None, budget=…, reranker=None)`**
  — returns a `ContextSnapshot`. Omitted `limits` use `MemoryConfig` defaults.
- **`build_context_snapshot(query, user_id, limits, participants, …)`** — same
  result type; the older context methods default missing memory limits to zero.
- **`build_context(query, user_id, limits, participants, temporal_resolution)`** — returns
  `{memory_name: rendered_section}` for every enabled, non-empty memory,
  routing each through the single-user or multi-participant path per its
  `scope`.
- **`render_prompt(query, user_id, limits, participants, temporal_resolution)`** — full system
  block: the system template + every section in `section_order`.
- **`extract(turns, user_id, participants)`** — gathers the `ExtractionSpec`s
  of every participating memory, builds one combined JSON schema + instruction,
  runs a single LLM call, and hands each memory its slice. Returns the raw
  extraction dict plus an `__added__` map of freshly-written items (used by
  dedup / the KG).

`recall`, `build_context_snapshot`, `build_context`, and `render_prompt` also
accept keyword-only `budget` and `reranker` overrides. Omitted budget inherits
the `Character` constructor default; `None` removes the cap.

The renderer respects the prompt config:

- **Headers** — `PromptConfig.<name>_header` (and `_header_multi` for group
  chats), falling back to the memory's `.title`.
- **Section order** — `PromptConfig.section_order`; sections not present are
  skipped.
- **Section template** — `PromptConfig.section_template` (default
  `"## {title}\n{body}"`).
- **System line** — `PromptConfig.system` with `{character_name}` and
  `{base_instruction}` interpolated.

---

## `Chat`

```python
from character_memory import Chat
```

A handle over rows in the shared `SQLiteStore`. The agent owns the store and
hands it to every `Chat`; chats never open a connection of their own.

| Method | Purpose |
|---|---|
| `add_message(role, content, *, user_id=None)` | Persist one turn; `user_id` is the speaker (defaults to owner for user turns, `None` for assistant). |
| `messages()` / `history()` | All turns oldest-first as OpenAI dicts; in a group chat each user dict carries an OpenAI-safe `name`. |
| `messages_with_speakers()` | Same but with the real `user_id` per turn (used by the multi-user extractor). |
| `participants()` | Distinct human speakers, oldest-first (the owner is always included). |
| `last_user_message()` | The most recent user turn, or `None`. |
| `unextracted()` | Rows not yet fed to the extractor, oldest-first. |
| `mark_extracted(ids)` | Flag specific message ids as already extracted. |
| `__len__` | Number of messages. |

The schema (additive — learned memories are untouched):

- `chats` — `id, user_id, title, created_at`
- `messages` — `id, chat_id, role, content, user_id, created_at, extracted`

`user_id` records **who spoke that turn**; legacy rows (NULL) fall back to the
chat owner. The `extracted` flag lets `CharacterAgent.extract` process only
new messages, so extraction is idempotent and resumable.

---

## Common patterns

### "Drive my own LLM, you do memory"

```python
chat.add_message("user", message)
system = agent.render_prompt(chat, budget=3000)    # 1. memory (one recall)
answer = my_llm(system, chat.messages())
chat.add_message("assistant", answer)              # 2. persist
agent.extract(chat); agent.persist_structured()    # 3. learn
```

### "Inject custom backends, keep everything else"

```python
from character_memory import CharacterAgent, OpenAICompatibleLLM, OpenAICompatibleEmbeddings
from my_code import MyRetrieverLLM, MyEmbedder

agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config(
    "assets/Kurisu/config.yaml",
    llm=MyRetrieverLLM(),            # overrides the auto-built OpenAI client
    embedder=MyEmbedder(),
)
agent.build()
```

### "Add a brand-new memory"

```python
from character_memory import CharacterAgent, Memory, OpenAICompatibleLLM, OpenAICompatibleEmbeddings

class MyMemory(Memory):
    name = "my_memory"
    def recall(self, query, user_id, limit, state_changing=True): ...
    def build(self, chunks): ...
    def persist(self, path): ...
    def load(self, path): ...

llm = OpenAICompatibleLLM()
emb = OpenAICompatibleEmbeddings()
agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu").load(llm, emb, memories=[
    # …the standard memories…, 
    MyMemory(),
])
agent.build()
```

See [Memory systems](memories/index.md) and [Custom backends](custom_backends.md)
for the full subclassing story.
