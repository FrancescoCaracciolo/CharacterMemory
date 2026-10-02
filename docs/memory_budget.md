# Global memory budget and reranking

A global budget selects whole items from all retrieved memories within one
token cap. It applies to the memory sections of a prompt, across memory types
and all participants in a group chat.

## Recall and inspect selected memories

With an already loaded and built `CharacterAgent`:

```python
from character_memory import ScoreMemoryReranker

result = agent.recall(
    "What did we discuss about robotics?",
    user_id="alice",
    budget=3000,
    reranker=ScoreMemoryReranker(memory_weights={"user_directives": 1.5}),
)
print(result.memory_budget)       # 3000
print(result.memory_token_count)  # <= 3000
print(result.sections)            # rendered sections in prompt order
for name, recall in result.recalls.items():
    for item in recall.items:
        print(name, item.text, item.score, item.metadata)
```

`recall()` returns a `ContextSnapshot`, with the same sections and per-memory
snapshots as `build_context_snapshot()`, plus `memory_token_count` and
`memory_budget`. Item scores and metadata remain those returned by retrieval.
Backend diagnostics describe that retrieval; they may include candidates
which did not survive selection.

Agent targets may be a `Chat`, chat ID, raw query, or message list. Use a
persisted `Chat` to include its history and all speakers. Each call performs
retrieval and can update recall bookkeeping; the snapshot is not a cached
preview of a separate generation call.

The same keyword-only `budget` and `reranker` overrides work on
`build_context`, `build_context_snapshot`, `render_prompt`, and
`generate_answer`, including streaming and tool-enabled generation:

```python
system_prompt = agent.render_prompt(chat, budget=3000)
# Or let the agent generate the reply:
answer = agent.generate_answer(chat, budget=3000)
```

These are alternative ways to build context; normally choose one per turn.

## Defaults and what counts

Configure the default in the character's YAML, or pass
`MemoryConfig(token_budget=3000)` through `load_from_config`:

```yaml
memory:
  token_budget: 3000
  user_facts_k: 8
  knowledge_graph_token_budget: 1000
```

| Per-call value | Python | HTTP JSON | Behavior |
|---|---|---|---|
| Omitted | no `budget` argument | no `budget` field | Inherit `memory.token_budget`; the default configuration is unlimited. |
| Unlimited | `budget=None` | `"budget": null` | Remove the cap for this call. |
| No memories | `budget=0` | `"budget": 0` | Omit all memory sections. |
| Positive integer | `budget=3000` | `"budget": 3000` | Select whole items within that cap. |

The counter includes memory headers, timestamps, speaker labels, formatting,
and blank-line separators between memory sections. It excludes the
persona/system instructions, intermediate `prompt:<id>` blocks, conversation
history, and tool messages. This is not a limit on the complete LLM request.

Existing per-memory limits determine the candidate pool. Most are item
counts; `knowledge_graph_token_budget` limits the graph's rendered body before
global selection. A global budget neither expands these searches nor disables
their limits. Items that do not fit are skipped while selection continues;
unused capacity is allowed. Sticky facts, directives, emotion dimensions, and
world state compete under the same cap, without truncation or guaranteed slots.

When a budget or reranker is active, candidates are retrieved with
`state_changing=False`. Only selected items receive exposure reinforcement.
World auto-advancement runs separately from selection. Unlimited calls without
a reranker retain the existing retrieval and reinforcement behavior.

## Default and custom rerankers

`MemoryReranker` is an abstract base class. `ScoreMemoryReranker` is the default
implementation when a budget is set; it makes no model calls. For each memory,
it divides positive finite scores by that memory's largest positive score,
then applies its configured weight (default `1`). Nonpositive/nonfinite scores
contribute zero, and ties preserve retrieval order. Weights must be finite and
nonnegative. This heuristic does not calibrate semantic relevance across
different backends or guarantee an optimal packing of items.

Subclass `MemoryReranker` to order or filter the available options:

```python
from character_memory import MemoryReranker

class MetadataPriorityReranker(MemoryReranker):
    def rerank(self, query, candidates, *, budget):
        return sorted(
            candidates,
            key=lambda candidate: (
                bool(candidate.item and candidate.item.metadata.get("preferred")),
                candidate.score,
            ),
            reverse=True,
        )

result = agent.recall("robotics", budget=3000, reranker=MetadataPriorityReranker())
```

Each public `MemoryCandidate` exposes:

| Attribute | Meaning |
|---|---|
| `id` | Request-local identifier, independent of database row IDs and graph node IDs. |
| `memory_name` | Source memory. |
| `item` | Original `MemoryItem`; `None` for an atomic body-only custom recall. |
| `score` | Original item score, or zero for body-only results. |
| `rendered_text` | Standalone rendered section, including its header and labels. |
| `token_count` | Token cost of that standalone section. |

Return an ordered subset of the supplied candidate objects, without modifying
them. Duplicate, replaced, or unknown candidates raise `ValueError`. The
character greedily considers them in that order and checks the exact grouped
rendering before accepting each item. Standalone costs are not additive:
headers can be shared and dialogue numbering can change. Selected items retain
their original order for each memory's formatter; sections follow the configured
section order. A reranker can
also filter results when the budget is unlimited.

## Tokenizers and custom memories

Both `CharacterAgent` and `Character` accept constructor defaults `reranker=`
and `token_counter=`. The default counter uses `tiktoken`'s `cl100k_base`;
errors propagate rather than falling back to an approximation. Supply a
deterministic `text -> nonnegative int` callable for another model's tokenizer,
returning zero for empty text. For example, with your tokenizer already loaded:

```python
from character_memory import CharacterAgent, ScoreMemoryReranker

agent = CharacterAgent(
    directory="assets/Ada",
    reranker=ScoreMemoryReranker(),
    token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
)
```

Direct `Character` callers can set `budget=` in the constructor. Its `recall`
method accepts per-memory `limits=` and explicit `participants=`; omitted
recall limits use `MemoryConfig` defaults. Its older context-building methods
still use their supplied limits, defaulting missing memory limits to zero.

Stateful custom memories should implement `record_recall(items)` to record
selected exposure. `StructuredMemory` already implements this shared behavior.
Override `format_selection(items, participants)` if custom section rendering
differs from the normal `format`/`format_grouped` path; it must handle subsets
without retrieving again. `prepare_recall()` is for lifecycle work independent
of exposure. These hooks have defaults and add no new abstract requirements.

## HTTP `/context`

Send the optional `budget` field in the JSON request body:

```bash
curl -X POST http://localhost:8000/context \
  -H 'Content-Type: application/json' \
  -d '{"character":"Ada","user":"alice","message":"What did we discuss about robotics?","budget":3000}'
```

The override applies only to this request and does not change configuration.
Negative numbers, booleans, strings, and floating-point values return HTTP
`422` before a chat or message is written. Omission, `null`, and `0` follow the
table above. Python library/client calls reject invalid budgets with `ValueError`.

The response shape is unchanged: `context`, `context_order`, `context_text`,
and `memories` describe the selected contributions, alongside `chat_id`.
Intermediate prompt blocks remain in the context but outside the cap.
`memory_token_count` and `memory_budget` are Python `ContextSnapshot` fields;
the HTTP response does not currently expose them.

The bundled HTTP client preserves the difference between omission and null:

```python
from character_memory import CharacterMemoryClient

client = CharacterMemoryClient("http://localhost:8000")
result = client.context("Ada", "alice", "Tell me about robotics", budget=3000)
print(result.context_text)
```

`get_context` is an alias with the same arguments. The HTTP endpoint accepts
the budget, not a serialized reranker or tokenizer; inject those defaults into
the server's `CharacterAgent` in Python when customizing server setup.
