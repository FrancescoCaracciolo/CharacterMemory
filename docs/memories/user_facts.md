# `user_facts` — per-user facts with confidence + decay

| | |
|---|---|
| **Class** | `UserFactMemory` (`memory/user_facts.py`) |
| **Flavor** | Structured memory (SQLite rows + `HybridSearch` index) |
| **Scope** | `PER_USER` |
| **Learns from chat?** | **Yes** — extracted every `extract_interval` turns |
| **Config toggle** | `MemoryConfig.enabled_user_facts` |
| **Retrieval size** | `MemoryConfig.user_facts_k` (default `5`) |

## What it does

Stores **facts about a user** — occupation, preferences, relationships, goals
— and any general facts the user stated. Each row has a `type`
(e.g. `preference`, `occupation`, `user_name`), the `content` itself, a
`confidence` in `[0, 1]`, plus the shared importance / decay fields.

This is the workhorse "what do I know about this person" memory.

## Row schema

| Column | Type | Notes |
|---|---|---|
| `id`, `user_id`, `importance`, `created_at`, `last_recalled`, `recall_count` | common | every structured memory |
| `type` | TEXT | free-form label (`general`, `occupation`, `preference`, …) |
| `content` | TEXT | the fact, one self-contained full sentence |
| `confidence` | REAL | the model's certainty, `0..1` |
| `chat_id` | TEXT (nullable) | the conversation the fact was learned in; `NULL` ⇒ legacy / single-user. Carried into the knowledge graph's `FactNode.chat_id` so a `ChatEdge` pairs this fact with the other facts and episodes of the same chat. |

## Recall

Standard `StructuredMemory.recall` combines BM25+similarity candidates for the
user with sticky-fact eligibility (importance ≥ `sticky_threshold`). Ranking
combines relevance, importance, decay, and bounded reinforcement; a sticky
slot is reserved when the retrieval limit allows it. The agent's
[global budget/reranker](../memory_budget.md) may still exclude any candidate,
including sticky facts. Rendered as:

```
<content> (type: <type>, confidence: 0.XX)
```

## Contradiction policy

`UserFactMemory` opts **into** contradiction resolution: stable facts
(occupation, preferences, …) that assert incompatible current truths —
"Michael is a doctor" vs "Michael is an engineer" — are surfaced by the
deduplicator's contradiction gate and the older row's text is overwritten.
See [Extraction & dedup](../extraction_and_dedup.md).

## Extraction

`extraction_spec` declares the `facts` field:

```jsonc
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {"type": "string"},
      "content": {"type": "string"},
      "importance": {"type": "number"},
      "confidence": {"type": "number"}
    },
    "required": ["type", "content", "importance", "confidence"]
  }
}
```

Instruction (paraphrased): *"stable facts about {user} (occupation,
preferences, relationships, goals) or general facts {user} stated; each
content must be one self-contained full sentence about {user};
importance 0-1; confidence 0-1."*

`apply_extraction` skips exact-duplicate `content` for that user (and, in a
group chat, attributes each item to the participant the LLM named).

## Section header

`"What you remember about this user"` (singular) /
`"What you remember about these users"` (group chat). Overridable via
`PromptConfig.user_facts_header[_multi]`.

## Standalone usage

```python
from character_memory import UserFactMemory, HybridSearch, OpenAICompatibleEmbeddings, SQLiteStore

mem = UserFactMemory(SQLiteStore("memory.db"), HybridSearch(OpenAICompatibleEmbeddings()))
mem.add_fact("michael", "Michael is a nuclear engineer called in by Daru.",
             type="occupation", importance=0.8, confidence=0.9)

for it in mem.recall("What is michael's job?", user_id="michael", limit=1):
    print(f"- {it.text}  (score={it.score:.3f})")
```

Extraction without the agent:

```python
from character_memory.memory.extract import Extractor, build_extraction
schema, instruction = build_extraction([mem.extraction_spec()])
extracted = Extractor(llm).extract(conversation, schema=schema, instruction=instruction)
mem.apply_extraction(extracted["facts"], user_id="michael")
```

## Direct manipulation

```python
mem.add_fact(user_id, content, *, type="general", importance=0.5, confidence=0.5, chat_id=None)
mem.all_rows(user_id=None)      # every row, or one user's
mem.get_row(row_id); mem.update_row(row); mem.delete_row(row_id)
mem.rebuild_index()
```

`chat_id` is forwarded into the knowledge graph's `FactNode.chat_id` so the
[`ChatEdge`](knowledge_graph.md) pass links this fact with the other facts
and episodes of the same conversation (wiki-derived nodes have no
`chat_id` and stay outside the chat-bridge subgraph).

The MCP server and WebUI expose `add_fact` / `update_fact` / `delete_fact`.
