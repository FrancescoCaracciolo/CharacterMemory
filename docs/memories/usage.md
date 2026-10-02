# Using memories

The agent handles recall and learning for enabled memories. You can also use
its memory objects directly to seed known information, inspect what was
learned, or correct a record. Start with the [working character example](../getting_started.md)
if you do not yet have a loaded agent.

## Choose where information belongs

| Memory | Put this here | How it gets populated | Default |
|---|---|---|---|
| [`character_info`](character_info.md) | “Ada works at the hilltop observatory.” Stable character lore. | Files in `Information/`, indexed by `build` / `rebuild`. | On |
| [`dialogue_style`](dialogue_style.md) | Examples of how Ada speaks. | `Speaker: text` exchanges in `Dialogues/`. | On |
| [`user_facts`](user_facts.md) | “Alice teaches biology.” A fact about a person. | Chat extraction or `add_fact`. | On |
| [`user_directives`](user_directives.md) | “Keep astronomy explanations short.” A standing instruction. | Chat extraction or `add_directive`. | On |
| [`episodic`](episodic.md) | “Alice and Ada watched a meteor shower together.” A remembered experience. | Chat extraction or `add_episode`. | On |
| `conversation_events` | The original conversation passages behind a memory. | Source records created during extraction, with links to extracted items. See [source events](../extraction_and_dedup.md). | On |
| [`emotion`](emotion.md) | Ada's current mood and her relationship feelings toward Alice. | Extracted emotion changes or explicit state updates. | On |
| [`user_summary`](user_summary.md) | Alice's name, aliases, and a compact rolling profile. | Chat extraction or `add_or_update`. | On |
| [`heartbeat`](heartbeat.md) | “Reviewed last night's observations.” Something Ada did outside a chat. | Your application's background loop calls `add_entry`. | On |
| `world` | Where Ada is, her routines, needs, and private world facts. | `world.yaml`, simulation, world commands, and extracted world directives. See [world configuration](../configuration.md). | Off |
| [`calendar`](calendar.md) | “Meet Alice at 18:00 on Friday.” A dated commitment. | Explicit commitments extracted from chat, calendar APIs/tools, and live world routines. | Off |
| [`knowledge_graph`](knowledge_graph.md) | Connections between people, facts, episodes, places, and entities. | Built from source memories and lore, then updated after extraction. | Off |

A fact describes something true; an episode describes something that happened;
a directive describes how the character should act. A calendar entry holds a
scheduled commitment, while an episode can later record the experience of it.
Conversation events preserve source text rather than replacing these summaries.

Enabling heartbeat provides a journal; your application must run the background
work that writes it. Enabling calendar stores and recalls commitments; delivering
notifications is your application's responsibility.

## Configure the memories before loading

In `assets/Ada/config.yaml`, the keys belong under `memory`:

```yaml
memory:
  enabled_user_facts: true
  enabled_user_directives: true
  enabled_episodic: true
  enabled_emotion: true
  enabled_user_summary: true
  enabled_heartbeat: false
  enabled_calendar: true
  calendar:
    timezone: Europe/Rome
  enabled_world: false
  enabled_knowledge_graph: false
  extract_interval: 5
  user_facts_k: 5
  episodic_k: 4
  conversation_events_k: 0
```

Load this with `agent.load_from_config("assets/Ada/config.yaml")`, followed by
`agent.build()`. Omitted settings retain their defaults. The same fields are
available through `MemoryConfig` for applications that configure in Python.
Apply config changes when constructing/loading the agent; mutating an already
loaded config object does not rewire its memories and retrieval limits.

There are two different controls:

- `enabled_<name>: false` excludes a memory from the agent's automatic recall
  and extraction. It does not delete its saved data. Re-enable it to use that
  data again. Direct memory calls are lower-level operations; the enabled flag
  is not an access-control boundary.
- `<name>_k: 0` keeps the memory available for learning and direct/tool access,
  but leaves it out of automatic prompt retrieval. Positive values bound the
  number of recalled items. The example makes conversation events available
  on demand without inserting original passages into every prompt.

The knowledge graph uses `knowledge_graph_token_budget` instead of a `_k`
item count. It is an additional retriever over source memories; start with
ordinary memories, then enable it when you want connected recall. Its build
and incremental ingestion can involve additional model requests. See
[knowledge graph setup](knowledge_graph.md).

## Let the character learn from a conversation

The following snippets assume `agent` is loaded and built as in Getting started.

```python
chat = agent.create_chat(user="alice")
chat.add_message("user", "I'm a biology teacher. Please keep your explanations short.")
answer = agent.generate_answer(chat)
print(answer)

# Useful for a first-turn demo or the end of a short session:
agent.extract(chat)
```

With default settings, `generate_answer` automatically extracts after every
five user turns in that chat. Explicit `extract(chat)` runs immediately over
recent unprocessed messages. An extraction may produce facts, directives,
episodes, emotion changes, and an updated profile; it need not produce an
entry for every memory. Learned structured indexes are persisted by extraction.

To own the extraction schedule, pass `auto_extract=False` to
`generate_answer`, then call `extract(chat)` yourself. Saving a reply is separate
from learning. Call extraction regularly: the current window is at most
`max(extract_interval * 2, 4)` recent unprocessed message rows per chat, including
assistant messages. It is not an API for importing an arbitrarily long backlog.

Deduplication beyond each memory's basic guards is **off by default**. To enable
the post-extraction deduplicator, set `memory.dedup.enabled: true`; semantic gates,
an optional LLM judge, and consolidation are configured separately. See
[Extraction & dedup](../extraction_and_dedup.md).

## Add information you already know

Use `agent.memories[name]` to access a configured memory. These writes do not
need an LLM extraction call, although RAG-backed additions use the embedder.
For an import, keep the returned row IDs if you will need to edit the records.

```python
facts = agent.memories["user_facts"]
fact_id = facts.add_fact(
    "alice",
    "Alice teaches biology at a secondary school.",
    type="occupation",
    importance=0.8,
    confidence=1.0,
)

agent.memories["user_directives"].add_directive(
    "alice",
    "Keep explanations short unless Alice asks for more detail.",
    importance=0.8,
    retrieval_keywords=["explain", "astronomy"],
)

agent.memories["episodic"].add_episode(
    "alice",
    "Alice and Ada watched a meteor shower from the observatory.",
    importance=0.7,
    emotional_shift={"joy": 0.3},
)

agent.memories["user_summary"].add_or_update(
    "alice",
    name="Alice",
    aliases=["Ali"],
    summary="A biology teacher who enjoys astronomy and prefers concise explanations.",
)
agent.persist_structured()
```

Use `importance` in `[0, 1]` for salience, and `confidence` in `[0, 1]` for
certainty about a fact. An episodic `emotional_shift` is a dictionary of
changes on configured baseline emotion axes, not a single scalar.
`add_or_update` replaces the profile name/summary and merges aliases.

Direct writes bypass the agent's post-extraction deduplication and knowledge
graph ingestion. For a large import, run `agent.dedup()` if configured. If the
knowledge graph is enabled, `agent.rebuild_knowledge_graph()` rebuilds it from
the current sources; this can involve model requests. Finally, persist the
indexes. Do not assume a direct `add_fact` also creates a graph node.

Some memories have their own APIs rather than the structured row interface:

```python
emotion = agent.memories["emotion"]
print(emotion.get_current_mood())
print(emotion.get_user_state("alice"))
emotion.update("alice", {"trust": 0.1})

# Requires heartbeat to be enabled in your configuration for automatic recall.
agent.memories["heartbeat"].add_entry(
    "Reviewed last night's observations.", kind="action", importance=0.6,
)
agent.persist_structured()
```

For dated events, use [`agent.calendar_memory.create_event`](calendar.md);
for exact world state, use `agent.world_snapshot()` and the
[world tools](../tools.md). Consult each memory's page for its write semantics.

## Inspect stored records and recalled context

Stored information and selected prompt context are different views. A memory
can contain hundreds of rows while returning only a few relevant items.

```python
facts = agent.memories["user_facts"]

# All stored facts for this user, independent of query relevance or decay:
for row in facts.all_rows(user_id="alice"):
    print(row["id"], row["content"], row["importance"])

# Search without increasing recall counts or changing last-recalled times:
items = facts.recall(
    "What does Alice do for work?",
    user_id="alice",
    limit=3,
    state_changing=False,
)
for item in items:
    print(item.text, item.score, item.metadata)
```

To inspect the exact sections and items selected by an agent context build:

```python
snapshot = agent.recall(
    "What does Alice do for work?", user_id="alice", budget=3000,
)
print(snapshot.memory_token_count, snapshot.memory_budget)
for name, section in snapshot.sections.items():
    print(name, section)
for name, recall in snapshot.recalls.items():
    print(name, [item.text for item in recall.items])
```

A context build includes memory bookkeeping. When a budget or reranker is
active, only selected items are reinforced. The global budget counts the
rendered sections after per-memory retrieval; it does not expand each memory's
candidate limit. Configure a default with `memory.token_budget` or override it
per call. See [Memory budget & reranking](../memory_budget.md).

For a read-only diagnostic search, use a memory's
`recall(..., state_changing=False)` as above. Agent `recall`, `build_context`,
`build_context_snapshot`, and `render_prompt` each perform retrieval; calling
them successively does not inspect a shared cached result.

Structured retrieval combines query relevance, importance, age, and bounded
recall reinforcement. Sticky importance makes an item eligible outside the
usual search candidates, but does not guarantee that every sticky item fits
within the retrieval limit. Emotional impact also influences episodic ranking.
Low recall rank is not deletion: use `all_rows` to check what remains stored.
See the [retrieval reference](index.md#recall-algorithm-single-user-path).

If a fact does not appear, check the user ID, enabled flag, retrieval limit,
global budget/reranker, whether extraction ran, and the stored row before
adjusting retrieval settings.

## Correct or remove a fact

Use the memory API so edits remain compatible with its backing store. For
`UserFactMemory`, `update_row` and `delete_row` change rows without updating
the search index, so rebuild that memory's index after edits:

```python
facts = agent.memories["user_facts"]
row = facts.get_row(fact_id)  # fact_id was returned by add_fact above.
if row is not None:
    row["content"] = "Alice now teaches biology at a university."
    facts.update_row(row)
    facts.rebuild_index()
    agent.persist_structured()

# To delete that fact instead:
# facts.delete_row(fact_id)
# facts.rebuild_index()
# agent.persist_structured()
```

If the knowledge graph is enabled, refresh its source representation after
manual edits as described above. Removing a fact does not remove the source
chat messages, conversation events, or related episodes; they are separate
records. The WebUI and MCP also expose memory editing APIs.

## Reuse memories across chats and users

`user_id` identifies a person; `chat.id` identifies a conversation. Reusing the
same user ID in a new chat makes that person's learned memories available
without including an old chat's transcript. Changing the ID starts a separate
per-user memory history. Display names and profile aliases do not replace IDs.

For a group chat, attach the actual speaker to each user message:

```python
group = agent.create_chat(user="alice", title="Observatory visitors")
group.add_message("user", "I teach biology.", user_id="alice")
group.add_message("user", "I build telescopes.", user_id="bob")
# Learn with each statement attributed to its actual speaker.
agent.extract(group)
```

Persisted chats let the agent recall for all participants and attribute
extraction to the speakers. Character-scoped lore, dialogue examples, world
state, and heartbeat entries are shared. Raw query strings and bare message
lists use the supplied single `user_id`; use a `Chat` for group attribution.
Knowledge-graph cross-user visibility has a separate
[privacy setting](knowledge_graph.md); see [Multi-user chats](../multi_user.md)
for the complete behavior.

Emotion recall supports the shared temporal arguments in group chats; it
remains a current-state snapshot. A global budget is shared across all speakers
and memories, including emotion dimensions; speaker labels count toward it.

When finished, call `agent.close()`. On restart, load the same character and
state directory. See [Persistence and restarts](../getting_started.md#persistence-and-restarts)
for the library/server path difference and backup guidance.
