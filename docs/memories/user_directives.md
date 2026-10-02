# `user_directives` — per-user standing instructions + keyword boosts

| | |
|---|---|
| **Class** | `UserDirectiveMemory` (`memory/user_directives.py`) |
| **Flavor** | Structured memory |
| **Scope** | `PER_USER` |
| **Learns from chat?** | **Yes** |
| **Config toggle** | `MemoryConfig.enabled_user_directives` |
| **Retrieval size** | `MemoryConfig.user_directives_k` (default `4`) |

## What it does

Stores **standing instructions** a user has asked the character to follow —
"always answer me formally", "when I ask you to build a website, activate
the frontend-design skill", "when I say 'run a heartbeat', read
`HEARTBEAT.md` and follow it". Each directive carries a list of
`retrieval_keywords`: terms whose presence in a user query **boosts recall**
even when the hybrid search would have missed the row.

Think of it as durable user-side policy, surfaced exactly when it matters.

## Row schema

| Column | Type | Notes |
|---|---|---|
| common columns | | every structured memory |
| `content` | TEXT | the instruction, one self-contained full sentence |
| `retrieval_keywords` | TEXT | JSON array of trigger terms |

## Recall (keyword boost)

`recall` runs the standard structured path first, then performs a **keyword
boost pass** over this user's rows: any directive whose keywords appear in
the (lowercased) query is added to the candidate set, up to `limit`. So even
if BM25+similarity missed a directive, the right keywords can make it a
candidate. A [global budget/reranker](../memory_budget.md) can still exclude a
retrieved directive from the final prompt; keyword matches do not bypass the cap.

```python
mem.add_directive(
    "alice",
    "When you are creating a website, activate the skill frontend-design",
    importance=0.8,
    retrieval_keywords=["design", "website"],
)
mem.add_directive(
    "alice",
    "When I tell you run a heartbeat, read HEARTBEAT.md and run the instructions in it",
    importance=0.50,
    retrieval_keywords=["heartbeat"],
)

# "Make me a website with all the things you like" → surfaces the first
# directive thanks to the "website" keyword, even though hybrid recall
# ranked other rows higher.
for it in mem.recall("Make me a website with all the things you like",
                     user_id="alice", limit=1):
    print(f"- {it.text}  (score={it.score:.3f})")
```

## Extraction

`extraction_spec` declares the `directives` field:

```jsonc
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "content": {"type": "string"},
      "importance": {"type": "number"},
      "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["content", "importance", "keywords"]
  }
}
```

Instruction (paraphrased): *"standing instructions {user} asked {char} to
follow; each content must be one self-contained full sentence; importance
0-1; keywords: terms that should trigger retrieval."*

`apply_extraction` dedups exact-`content` matches and attributes items to the
right participant in a group chat.

## Section header

`"Standing instructions from this user"` (singular) /
`"Standing instructions from these users"` (group chat). Overridable via
`PromptConfig.user_directives_header[_multi]`.

## Direct manipulation

```python
mem.add_directive(user_id, content, *, importance=0.5, retrieval_keywords=None)
mem.all_rows(user_id=None); mem.get_row(id); mem.update_row(row); mem.delete_row(id)
```

MCP tools: `add_directive` / `update_directive` / `delete_directive`.
