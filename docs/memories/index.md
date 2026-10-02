# Memory systems

For practical examples, start with [Using memories](usage.md): choose a memory,
configure it, learn from chat, add records, inspect recall, and correct facts.
This page describes the interfaces and retrieval behavior for library extension.

A **memory** is anything that can `recall(query, user_id, limit)` and render
itself into a prompt section. Every memory subclasses `Memory` (in
`memory/base.py`). The agent iterates the *enabled* memories, asks each for a
rendered section, and concatenates them into the system prompt.

There are **twelve** built-in memories, each independently enable/disable-able.
They come in three architectural flavors:

| Flavor | Backed by | Examples |
|---|---|---|
| **RAG memory** | a pure `HybridSearch` index over pre-built chunks | `character_info`, `dialogue_style` |
| **Structured memory** | a SQLite table (rows) **plus** a `HybridSearch` index over those rows, with decay + recall bookkeeping | `user_facts`, `user_directives`, `episodic`, `heartbeat`, `user_summary`, `calendar` |
| **Custom** | a bespoke backing store | `emotion`, `knowledge_graph` |

| Memory | Flavor | Scope | Learns from chat? | Page |
|---|---|---|---|---|
| `character_info` | RAG | CHARACTER | no (rebuilt from `Information/*.md`) | [→](character_info.md) |
| `dialogue_style` | RAG | CHARACTER | no (rebuilt from `Dialogues/*.md`) | [→](dialogue_style.md) |
| `user_facts` | Structured | PER_USER | yes | [→](user_facts.md) |
| `user_directives` | Structured | PER_USER | yes | [→](user_directives.md) |
| `episodic` | Structured | PER_USER | yes | [→](episodic.md) |
| `conversation_events` | Structured | PER_USER | no (immutable source records) | [→](../extraction_and_dedup.md) |
| `emotion` | Custom | PER_USER | yes | [→](emotion.md) |
| `world` | Custom | CHARACTER | yes (world directives) | [→](../architecture.md) |
| `heartbeat` | Structured | CHARACTER | no (autonomous loop) | [→](heartbeat.md) |
| `user_summary` | Structured | PER_USER | yes | [→](user_summary.md) |
| `calendar` | Structured | PER_USER | yes | [→](calendar.md) |
| `knowledge_graph` | Custom | PER_USER | indirectly (reads other memories) | [→](knowledge_graph.md) |

---

## The `Memory` ABC

```python
class Memory(ABC):
    name: str = "memory"
    scope: MemoryScope = MemoryScope.PER_USER   # PER_USER | CHARACTER

    def __init__(self, *, enabled: bool = True, name=None): ...

    @abstractmethod
    def recall(self, query, user_id, limit, *, state_changing=True) -> list[MemoryItem]: ...
    @abstractmethod
    def build(self, info_chunks: list[Chunk]) -> None: ...
    @abstractmethod
    def persist(self, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> None: ...

    # Optional hooks (have sensible defaults):
    def format(self, items) -> str: ...
    def build_section(self, query, user_id, limit, state_changing=True, *,
                      temporal_resolution=None, temporal_resolution_engine=None,
                      temporal_weight=None) -> str | None: ...
    def recall_participants(self, query, participants, limit, state_changing=True, *,
                            temporal_resolution=None, temporal_resolution_engine=None,
                            temporal_weight=None) -> list[MemoryItem]: ...
    def build_section_participants(self, query, participants, limit, state_changing=True, *,
                                   temporal_resolution=None, temporal_resolution_engine=None,
                                   temporal_weight=None) -> str | None: ...
    def format_grouped(self, items, participants) -> str: ...
    def format_selection(self, items, participants) -> str: ...
    def prepare_recall(self) -> None: ...
    def record_recall(self, items) -> None: ...
    def get_memories(self, limit=0) -> list[MemoryItem]: ...
    def extraction_spec(self, context=None) -> ExtractionSpec | None: ...
    def apply_extraction(self, value, user_id) -> list[MemoryItem]: ...
```

The four abstract methods are the **recall → render** half and the
**build / persist / load** lifecycle half. Everything else has a sensible
default; override only what you need.

The temporal kwargs are forwarded (to `recall` and, in group chats, to
`recall_participants`) **only when the memory sets
`supports_temporal_resolution = True`**. Opt-in memories must then accept the
three kwargs in their `recall` (and, if overridden, `recall_participants`)
signatures; memories that don't opt in keep the simpler signatures and never
see the kwargs.

For [global budget selection](../memory_budget.md), `format_selection` renders
selected subsets through `format` or `format_grouped` by default. Custom
section renderers can override it. `record_recall` and `prepare_recall` default
to no-ops; stateful memories implement selected exposure in `record_recall`,
while independent lifecycle work belongs in `prepare_recall`.

### Key concepts

- **`MemoryItem`** — a recalled item ready for the prompt: `text`, `score`,
  `kind`, and a `metadata` dict (carries the source row, the speaker, etc.).
- **`scope`** — how the memory relates to chat participants (see
  [Multi-user](../multi_user.md)):
  - `PER_USER` (default): recall once per participant; group results by
    speaker. Facts / directives / episodes / emotion / summary.
  - `CHARACTER`: recall once, ignoring participants. Wiki / dialogues /
    heartbeat / KG self-baseline.
- **`state_changing`** — when `False`, recall is read-only: memories must not
  bump recall-count / last-recalled bookkeeping. Used by the memory self-tools
  and explicit read-only searches so inspection does not skew decay statistics.
  Agent context-building calls can update bookkeeping. With a global budget or
  reranker, candidate retrieval is read-only and `record_recall(items)` records
  only selected exposure afterward.
- **Extraction** — a memory that wants to learn from chat returns an
  `ExtractionSpec(field, schema, instruction, per_user)` from
  `extraction_spec(context)`. The agent builds a single combined schema out of
  every participating memory's spec, runs one LLM call, then hands each memory
  its slice via `apply_extraction(value, user_id)`. Returning `None` opts out.

---

## `StructuredMemory` — the shared base

`UserFactMemory`, `UserDirectiveMemory`, `EpisodicMemory`, `HeartbeatJournal`
and `UserSummaryMemory` all subclass `StructuredMemory` (in
`memory/structured.py`). It owns the **SQLite row ↔ RAG index ↔ decay**
wiring. Subclasses only declare:

- `table` — the SQLite table name.
- `extra_columns` — memory-specific columns (e.g. `content`, `confidence`).
- `text_column` — the column whose text gets embedded + BM25-indexed.
- `row_text(row)` — how a row renders to embedding/BM25 text.
- `row_item(row, score)` — how a row renders to a prompt `MemoryItem`.
- optionally `extraction_spec`, `apply_extraction`, `contradiction_policy`,
  and an `_effective(row)` override (episodic does this for emotional impact).

**Never reimplement `recall` / `add` / `rebuild_index` in a subclass.**

### Columns every structured memory gets for free

```python
COMMON_COLUMNS = {
    "id":            "INTEGER PRIMARY KEY AUTOINCREMENT",
    "user_id":       "TEXT NOT NULL",
    "importance":    "REAL NOT NULL DEFAULT 0.5",
    "created_at":    "REAL NOT NULL",
    "last_recalled": "REAL",
    "recall_count":  "INTEGER NOT NULL DEFAULT 0",
}
```

### Recall algorithm (single-user path)

1. Load the user's rows from SQLite.
2. Run `hybrid.search(query, k=max(limit, candidate_pool), where={"user_id": uid})`
   for BM25+similarity candidates.
3. Add every sticky row to the candidate set, then score all candidates.
4. Rank with bounded relevance repair (below), and—when `limit > 1`—partition
   at most one slot for the best sticky row so sticky rows cannot crowd the
   relevance-driven slots.
5. Bump `recall_count` / `last_recalled` for what was surfaced (skipped when
   `state_changing=False`).

### Bounded familiarity and relevance repair

From `memory/decay.py`:

```
intrinsic  = base · emotion_boost
aged       = intrinsic · exp(-age / (half_life · decay_resistance))
exposure   = recall_count / (recall_count + 10)
familiar   = min(intrinsic, aged + 0.10 · intrinsic · exposure)
ranking    = familiar + relevance · 1.10 · (intrinsic - familiar)

  emotion_boost     = 1 + |emotion_impact|         (EpisodicMemory only)
  decay_resistance = 1 + max(0, emotion_impact) (EpisodicMemory only)
  age = now - (semantic_event_time or created_at)
```

Hybrid RRF supplies relevance in `[0, 1]`. Exposure can restore at most ten
percent of intrinsic salience and never refreshes age; `last_recalled` remains
telemetry. Maximum relevance repairs 110% of the remaining decay loss.
`base` alone decides sticky eligibility.

### The knowledge graph reads these

When the KG memory is enabled, `StructuredMemory` rows are its **source of
truth**: it ingests `user_facts`, `episodic`, `user_summary`, `emotion` and
the wiki through their public APIs and never writes back. See
[Knowledge graph](knowledge_graph.md).

---

## Extraction in one paragraph

Every `extract_interval` user turns, the agent:

1. Asks each enabled memory for an `ExtractionSpec`.
2. Combines them into **one** JSON schema + instruction (one LLM call).
3. Labels the transcript with each turn's real speaker (group chat).
4. Hands each memory its slice via `apply_extraction`.
5. Runs dedup (and mirrors mutations into the KG).

Returning `None` from `extraction_spec` opts a memory out of extraction
entirely. See [Extraction & dedup](../extraction_and_dedup.md) for the full
flow, and each memory's page for its specific schema.

---

## Adding a new memory

The minimum viable memory:

```python
from character_memory import Memory, MemoryItem, MemoryScope, Chunk

class NotesMemory(Memory):
    name = "notes"
    scope = MemoryScope.CHARACTER

    def __init__(self, notes: list[str], *, enabled=True):
        super().__init__(enabled=enabled)
        self._notes = notes

    def recall(self, query, user_id, limit, state_changing=True):
        # naive: substring match
        return [MemoryItem(text=n, score=1.0, kind=self.name)
                for n in self._notes if query.lower() in n.lower()][:limit]

    def build(self, info_chunks: list[Chunk]) -> None: pass
    def persist(self, path: str) -> None: pass
    def load(self, path: str) -> None: pass
```

Then pass it via `CharacterAgent.load(llm, embedder, memories=[…])`. If it
should learn from the conversation, also implement `extraction_spec` and
`apply_extraction`. See [Custom backends](../custom_backends.md) for
patterns, including how to subclass `StructuredMemory` for free SQLite+decay
plumbing.

---

## Standalone usage

Every memory is usable **without** the agent — they are just objects. See the
`examples/` folder:

```bash
python examples/character_info.py     # RAG over the wiki
python examples/dialogue_style.py     # few-shot exchanges
python examples/user_facts.py         # facts + extraction + recall
python examples/user_directives.py    # standing instructions + keyword boost
python examples/episodic.py           # episodes weighted by emotion
python examples/heartbeat.py          # the character's own journal
python -m examples.kurisu             # full agent end-to-end
python -m examples.knowledge_graph    # KG retriever API
```

Each memory's page below reproduces the relevant snippet.
