# Extraction & deduplication

The two out-of-band processes that turn a conversation into durable memory.

## Extraction

Every `MemoryConfig.extract_interval` user turns (default `5`), the agent runs
extraction over the conversation's not-yet-processed messages. It is
**idempotent and resumable** — every `messages` row carries an `extracted`
flag, so re-running never double-counts.

```text
For each enabled memory that returns a non-None ExtractionSpec:
    gather (memory, spec) pairs
Combine every spec into ONE JSON schema + ONE instruction
Run a single chat_structured call over the labelled transcript
For each (memory, spec):
    items = memory.apply_extraction(extracted[spec.field], user_id, chat_id=chat_id)
Track freshly-added items in extracted["__added__"]
Persist structured indexes
Feed added items to the knowledge graph (incremental update)
Run dedup over the added items; mirror mutations into the KG
Flag the processed messages extracted=1
```

The combined-schema trick is the heart of it: even if five memories want to
learn, **one** LLM call produces facts + directives + episodes + emotion
deltas + user summaries in a single JSON object. Each memory then consumes
its slice.

### World-memory boundary

Automatic `world_updates` extraction is limited to the shared setting:
`facts` describe established setting properties or explicitly time-bounded
conditions, while `events` describe completed happenings that affect that
setting. Laboratory equipment, a power outage with a stated duration, and
damage to a shared location qualify. The character's current actions belong
in `actions` and retain their assistant-message provenance requirement.

Personal studying, exam logistics, browsing thesis proposals, conversation
recaps, and failed image delivery are neither world facts nor world events.
Mentioning the character or adding a date does not change this. Personal
milestones qualify only insofar as they establish a concrete shared-setting
change; extract the setting change, not the personal recap.

Worthwhile personal information can go to the top-level user `facts` or
`episodes` fields when enabled and when it meets their selection rules.
Otherwise it is omitted, including when those memories are disabled. Empty
world arrays are expected for ordinary personal conversation. These semantic
instructions guide the model rather than deterministically validating its
classification. They affect future extraction only; existing records and
manually authored world updates are unchanged.

### Driver: `Character.extract`

```python
result = agent.character.extract(
    turns,                 # [{role, content, user_id}, …] — speaker per turn
    user_id="michael",     # the chat's default user (owner / current speaker)
    participants=None,     # [..] with len>1 ⇒ multi-user mode
    chat_id=None,          # [..] chat-scoped memories stamp it onto their rows
)
```

It builds an `ExtractionContext` (character/user names, persona, known facts
to avoid re-extracting) and uses the participating memories' specs. The
returned dict includes an `__added__` map: `{memory_name: [MemoryItem, …]}`
for every memory that wrote rows, so dedup / the KG can act on them without
re-querying.

`chat_id` (optional) identifies the **conversation** extraction ran over.
`Memory.apply_extraction` accepts `chat_id` and chat-scoped memories
(`user_facts`, `episodic`) stamp it onto the row they write; other memories
accept and ignore it. The point is to let the knowledge graph link the
facts and episodes of the same chat via a [`ChatEdge`](knowledge_graph.md).
When `None` (legacy / off-context callers) the column is left `NULL` and the
chat-bridge hyperlink is skipped for that row, matching the pre-`chat_id`
behaviour bit-for-bit.

### Single-user vs multi-user

- **1:1 chat** → each memory's spec is used as-is; per-user items are
  attributed to the single participant.
- **Group chat** (participants > 1) → for every `per_user=True` spec the
  builder injects a `user_id` enum of the participants into the item schema,
  the transcript is labelled with each turn's real speaker, and an
  attribution note is appended. Each per-user memory's `apply_extraction`
  then honours the per-item `user_id`. Emotion returns
  `[{user_id, deltas, comment?}]` in this mode.

### Grounding: known facts

The extractor is grounded with the top known user facts (highest importance
first), so the model does not re-extract what is already stored. In a group
chat the facts of every participant are gathered (each capped at `limit`).

### Manual control

```python
agent.extract(chat)      # extract just this chat's unprocessed turns
agent.extract()          # extract every unprocessed turn across all chats
```

Both are safe to call repeatedly. The HTTP `/save` endpoint calls
`agent.extract(chat)` after persisting the assistant turn.

### Opting out / in

A memory opts out of extraction by returning `None` from
`extraction_spec(context)`. The agent only consults *enabled* memories, so a
disabled memory is never asked. To add a new memory to the extraction batch,
implement `extraction_spec` and thread `chat_id` through `apply_extraction`
— see [Custom backends → Memory](custom_backends.md#5-custom-memory--memory--structuredmemory).

---

## Decay (why some memories fade)

Structured memories (`user_facts`, `user_directives`, `episodic`,
`heartbeat`, `user_summary`) combine intrinsic importance, age, bounded
prompt familiarity, and normalized retrieval relevance:

```
intrinsic = base · emotion_boost
aged      = intrinsic · exp(-age / (half_life · decay_resistance))
familiar  = min(intrinsic, aged + 0.10 · intrinsic · count/(count + 10))
ranking   = familiar + relevance · 1.10 · (intrinsic - familiar)

  emotion_boost     = 1 + |emotion_impact|         (EpisodicMemory only)
  decay_resistance = 1 + max(0, emotion_impact) (EpisodicMemory only)
  age = now - (semantic_event_time or created_at)
```

- Prompt exposure restores at most ten percent of salience and never renews
  the decay clock; `recall_count` and `last_recalled` remain telemetry.
- With room for both, at most one **sticky** fact (importance ≥
  `MemoryConfig.sticky_threshold`, default `0.95`) occupies the reserved
  sticky slot; the remaining slots exclude sticky rows.
- `base` alone decides stickiness, so a deliberately important fact never
  fully vanishes.
- For `EpisodicMemory`, the emotional magnitude amplifies retention and
  positive events additionally resist forgetting.

Tunable via `MemoryConfig.decay_half_life` (default 3 days) and
`MemoryConfig.sticky_threshold`.

---

## Deduplication

Without dedup, extraction would slowly fill a memory with near-duplicates
("Michael is an engineer", "Michael works as an engineer", "Michael's job is
engineering"). The optional `Deduplicator` runs as a **post-extraction** step
on the freshly-added items, and as a **sweep** on demand.

A memory entry is considered a duplicate when it passes every *enabled* gate,
evaluated in escalating order:

1. **Exact** (cheapest) — case-insensitive, stripped string equality.
2. **Similarity** — cosine similarity above
   `DedupConfig.similarity_threshold` (default `0.92`; `None` disables).
3. **LLM judge** — when `DedupConfig.llm_judge` is on, an LLM confirms a
   similarity candidate is genuinely the same information.

If `DedupConfig.consolidate` is on, a confirmed duplicate is **merged** into
one entry (an LLM writes a single concise entry preserving every distinct
fact); otherwise the newer one is dropped.

`DedupConfig.per_user` (default on) only compares entries sharing a
`user_id` during a sweep, so users stay isolated.

### Contradiction resolution

A separate, per-memory-opt-in gate. After dedup finds no duplicate, the
deduplicator can surface semantically close rows that **clash** — "Michael is
a doctor" vs "Michael is an engineer" — and overwrite the older row's text
with the newer one's. Enabled per memory via
`StructuredMemory.contradiction_policy()`:

| Memory | Policy |
|---|---|
| `user_facts` | enabled, default threshold `0.70` |
| `episodic` | enabled, threshold `0.65` (summaries are looser) |
| others | disabled by default |

Timestamps are passed to the judge so it can tell a genuine clash from a
change over time ("user was sad Monday" vs "user was happy Tuesday" is **not**
a contradiction).

### Driving it

```python
# After every extraction round the agent runs dedup over the added items:
agent.extract(chat)

# On-demand sweep of one or every structured memory:
reports = agent.dedup("user_facts")        # one memory
reports = agent.dedup(user_id="michael")   # one user, every memory
reports = agent.dedup()                    # everything
# -> {"user_facts": DedupReport(merged=N, dropped=M, …), …}
```

Returns a `{memory_name: DedupReport}` mapping. The agent uses its configured
`Deduplicator` if dedup is enabled, otherwise builds a fresh default one
on the fly so a one-off sweep is always possible.

### Enabling / tuning

```python
from character_memory import MemoryConfig, DedupConfig

MemoryConfig(dedup=DedupConfig(
    enabled=True,
    similarity_threshold=0.90,
    llm_judge=True,        # needs an LLMClient
    consolidate=True,      # rewrite duplicates into one merged entry
))
```

Or in `config.yaml`:

```yaml
memory:
  dedup:
    enabled: true
    similarity_threshold: 0.90
    llm_judge: true
    consolidate: true
```

---

## Knowledge-graph interaction

When the KG memory is enabled, both processes feed it:

1. After extraction, `kg.retriever.update(added)` ingests the freshly-added
   source rows **incrementally** (the `added` map tells it what's new). The
   same call also re-runs `wire_chat_edges` so a freshly-added fact bridges
   to pre-existing episodes of the same chat — and vice versa. The full
   `build()` runs the bridge pass on the entire graph on first construct.
2. After dedup, `kg.retriever.apply_deduplication(reports)` mirrors the
   mutations — merged/dropped source rows map to graph nodes that need to be
   removed or refreshed.

The graph never writes back to a source memory; modules stay independent.
See [Knowledge graph](knowledge_graph.md) for the `chat` edge kind and
`wire_chat_edges` mechanics.
