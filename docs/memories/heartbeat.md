# `heartbeat` — the character's autonomous journal

| | |
|---|---|
| **Class** | `HeartbeatJournal` (`memory/heartbeat.py`) |
| **Flavor** | Structured memory |
| **Scope** | `CHARACTER` (the journal is the character's own log; ignores `user_id`) |
| **Learns from chat?** | No — written by the character's autonomous loop, not by extraction |
| **Config toggle** | `MemoryConfig.enabled_heartbeat` |
| **Retrieval size** | `MemoryConfig.heartbeat_k` (default `4`) |

## What it does

A structured log of the character's **autonomous discoveries and actions** —
things the character found or did *on its own*, outside any conversation.
Think of it as the character's diary / scratchpad: "read an article on
neural decoding", "drafted a reply to the lab group", "found a new paper on
memory consolidation".

This memory is **not** populated by ordinary chat extraction. Instead, an
autonomous "browsing" / background loop the host application runs writes
here via `add_entry`. The `UserDirectiveMemory` is the typical trigger:
users add a directive like *"when I say 'run a heartbeat', read `HEARTBEAT.md`
and follow it"*, and the application's heartbeat runner executes those
instructions and logs the outcome here.

## Row schema

| Column | Type | Notes |
|---|---|---|
| common columns | | every structured memory (`user_id` is always `"_self"`) |
| `summary` | TEXT | what the character found / did |
| `kind` | TEXT | `"discovery"` (default) or `"action"` |

## Recall

The journal ignores `user_id` (character-scoped). `recall`:

1. Loads the most recent rows (up to `candidate_pool`).
2. Runs `hybrid.search(query, k=limit)` over those rows.
3. Scores by effective importance with decay and takes `limit`.

There are **no sticky rows** in the journal.

## No extraction

`HeartbeatJournal` does **not** implement `extraction_spec` — it returns
`None` by default, so the extractor never tries to write to it. All writes
come from your application's heartbeat loop.

## Section header

`"Recent discoveries / actions of yours"`. Overridable via
`PromptConfig.heartbeat_header`.

## Standalone usage

```python
from character_memory import HeartbeatJournal, HybridSearch, OpenAICompatibleEmbeddings, SQLiteStore

mem = HeartbeatJournal(SQLiteStore("memory.db"), HybridSearch(OpenAICompatibleEmbeddings()))
mem.add_entry("Read an article on neural decoding.", kind="discovery", importance=0.6)
mem.add_entry("Drafted a reply to the lab group.",   kind="action",     importance=0.4)
mem.add_entry("Found a new paper on memory consolidation.", kind="discovery", importance=0.7)
mem.rebuild_index()

for it in mem.recall("anything about memory research?", user_id="_self", limit=2):
    print(f"- {it.text}  (score={it.score:.3f})")
```

## Direct manipulation

```python
mem.add_entry(summary, *, kind="discovery", importance=0.5, user_id="_self")
mem.all_rows(); mem.get_row(id); mem.update_row(row); mem.delete_row(id)
```

MCP tools: `list_heartbeats` / `search_heartbeats` / `add_heartbeat` /
`update_heartbeat` / `delete_heartbeat`. The list tool returns newest reports
first; search uses the journal's hybrid index with the server's lexical fallback.
