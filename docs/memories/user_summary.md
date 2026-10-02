# `user_summary` — a single rolling profile per user

| | |
|---|---|
| **Class** | `UserSummaryMemory` (`memory/user_summary.py`) |
| **Flavor** | Structured memory |
| **Scope** | `PER_USER` |
| **Learns from chat?** | **Yes** — extraction refreshes the single row |
| **Config toggle** | `MemoryConfig.enabled_user_summary` |
| **Retrieval size** | `MemoryConfig.user_summary_k` (default `2`) |

## What it does

A **single consolidated profile per user**: their `name`, every `aliases`
(nicknames / other names they go by — stored as a JSON list), and a `summary`
— a quick, self-contained description of who they are (interests, role,
personality, key facts).

Unlike the other structured memories, each user owns **exactly one row**.
Extraction **refreshes** the row instead of appending, so the profile stays
consolidated rather than fragmenting into many factlets. Refresh extraction is
snapshot-aware: it receives the current profile as a baseline and must return
the complete merged profile, preserving details that are not contradicted by
the new conversation. The summary is **always injected** (sticky by default:
`default_importance = 1.0`) whenever the user is present in the conversation.

## Row schema

| Column | Type | Notes |
|---|---|---|
| common columns | | every structured memory |
| `name` | TEXT | the user's real name |
| `aliases` | TEXT | JSON array of alternate names |
| `summary` | TEXT | free-text profile |
| `updated_at` | REAL | last profile refresh time (managed automatically) |

## Updates merge

`add_or_update(user_id, name, aliases, summary, *, importance=1.0)`:

- If no row exists for the user → inserts one.
- If a row exists → **unions** the new `aliases` with the stored ones
  (deduped), overwrites `name` / `summary` with the complete merged snapshot,
  preserves decay bookkeeping, updates `updated_at`, and rebuilds the per-user
  hybrid index so it stays in sync with the single row.

Aliases are normalised via `_parse_aliases`, which accepts a JSON list, a
Python list, or a comma-separated string.

## Extraction

`extraction_spec` declares the `user_summaries` field:

```jsonc
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name":     {"type": "string"},
      "aliases":  {"type": "array", "items": {"type": "string"}},
      "summary":  {"type": "string"}
    },
    "required": ["name", "aliases", "summary"]
  }
}
```

Instruction (paraphrased): *"for each distinct person, a single consolidated
profile with their `name`, every `aliases`, and a `summary`: a quick,
self-contained description of who they are. Use the current stored profile as
the preservation baseline and return the complete merged snapshot. Produce one
item per person, written from {user}'s perspective using the real name."*

`apply_extraction` calls `add_or_update` per item, attributing to the right
participant in a group chat.

## Section header

`"Summary of this user"` (singular) /
`"Summaries of these users"` (group chat). Overridable via
`PromptConfig.user_summary_header[_multi]`.

## Standalone usage

```python
from character_memory import UserSummaryMemory, HybridSearch, OpenAICompatibleEmbeddings, SQLiteStore

mem = UserSummaryMemory(SQLiteStore("memory.db"), HybridSearch(OpenAICompatibleEmbeddings()))
mem.add_or_update("michael", "Michael", ["Mike"],
                  "A nuclear engineer who helped with the phonewave.")

print(mem.get_summary("michael"))
# -> {'id': 1, 'user_id': 'michael', 'name': 'Michael',
#     'aliases': '["Mike"]', 'summary': 'A nuclear engineer …', ...}

for it in mem.recall("who is michael?", user_id="michael", limit=1):
    print(f"- {it.text}  (score={it.score:.3f})")
```

## Direct manipulation

```python
mem.add_or_update(user_id, name, aliases, summary, *, importance=1.0)
mem.get_summary(user_id)        # the stored row, or None
```

MCP tool: `set_user_summary` (add or merge). Because `add_or_update` already
rebuilds this memory's single-row index, the MCP write path skips the
redundant `rebuild_index` and only flushes to disk.
