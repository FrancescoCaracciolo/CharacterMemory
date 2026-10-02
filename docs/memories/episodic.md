# `episodic` — events the character experienced, weighted by emotion

| | |
|---|---|
| **Class** | `EpisodicMemory` (`memory/episodic.py`) |
| **Flavor** | Structured memory |
| **Scope** | `PER_USER` |
| **Learns from chat?** | **Yes** |
| **Config toggle** | `MemoryConfig.enabled_episodic` |
| **Retrieval size** | `MemoryConfig.episodic_k` (default `4`) |

## What it does

Stores **notable events** the character experienced with a user — written as
full sentences from the *character's* point of view ("Kurisu and Michael
tested the phonewave together late at night"). Each episode carries a sparse
`emotional_shift` vector whose configured axes identify the emotions that
caused the event's impact. Values are clamped to `0..1`; scalar shifts are
invalid.

The vector's raw impact and similarity to the current mood feed back into
retention via the decay formula; emotionally intense, mood-congruent episodes
rank higher.

## Row schema

| Column | Type | Notes |
|---|---|---|
| common columns | | every structured memory |
| `summary` | TEXT | what happened, one full sentence from the character's POV |
| `emotional_shift` | TEXT (JSON object) | sparse configured emotion axis → intensity (`0..1`) |
| `chat_id` | TEXT (nullable) | the conversation the episode was learned in; `NULL` ⇒ legacy / single-user. Carried into the knowledge graph's `EpisodeNode.chat_id` so a `ChatEdge` pairs this episode with the other facts and episodes of the same chat. |
| `source_message_ids` | TEXT (JSON array) | source-message provenance |
| `occurred_at` | REAL (nullable) | when the episode happened; derived from source messages and used by temporal recall |

## Effective importance (decay)

`EpisodicMemory` computes raw impact as the clamped L2 norm and compares the
vector with the persisted current mood using cosine similarity:

```
impact             = min(1, sqrt(sum(component²)))
similarity         = cosine(emotional_shift, current_mood)  # 0 for zero vectors
emotion_boost      = 1 + impact
decay_resistance   = 1 + impact
adjusted_half_life = half_life · decay_resistance
intrinsic          = importance · emotion_boost · (1 + similarity)
aged               = intrinsic · exp(-age / adjusted_half_life)
familiar           = min(intrinsic, aged + 0.10 · intrinsic · count/(count + 10))
ranking            = familiar + relevance · 1.10 · (intrinsic - familiar)
```

So a strongly positive or negative event sticks around longer than a neutral
one of the same base importance, and positive events additionally resist
forgetting.

## Contradiction policy

Episodic summaries opt into contradiction resolution but at a **lower
similarity bar** (`0.65` vs the default `0.70`) because summaries are looser
than bare facts. Timestamps are critical: "user was sad Monday" vs "user was
happy Tuesday" is a change over time, **not** a contradiction, and the judge
needs the timestamps to tell them apart.

## Extraction

`extraction_spec` declares the `episodes` field:

```jsonc
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "importance": {"type": "number"},
      "emotional_shift": {
        "type": "object",
        "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "required": ["summary", "importance", "emotional_shift"]
  }
}
```

Instruction (paraphrased): *"notable things that happened between {char} and
{user}, written as full sentences from {char}'s point of view;
`emotional_shift` identifies which configured emotions caused the event's
impact for {char}."*

## Section header

`"Episodes you've shared with this user"` (singular) /
`"Episodes you've shared with these users"` (group chat). Overridable via
`PromptConfig.episodic_header[_multi]`.

## Standalone usage

```python
from character_memory import EpisodicMemory, HybridSearch, OpenAICompatibleEmbeddings, SQLiteStore

mem = EpisodicMemory(SQLiteStore("memory.db"), HybridSearch(OpenAICompatibleEmbeddings()))
mem.add_episode("michael", "Kurisu and Michael tested the phonewave together late at night.",
                importance=0.85, emotional_shift={"joy": 0.7, "surprise": 0.2})

for it in mem.recall("phonewave testing", user_id="michael", limit=1):
    print(f"- {it.text}  (score={it.score:.3f})")
```

## Direct manipulation

```python
mem.add_episode(user_id, summary, *, importance=0.5, emotional_shift={}, chat_id=None)
mem.all_rows(user_id=None); mem.get_row(id); mem.update_row(row); mem.delete_row(id)
```

`chat_id` is forwarded into the knowledge graph's `EpisodeNode.chat_id` so
the [`ChatEdge`](knowledge_graph.md) pass links this episode with the other
facts and episodes of the same conversation (wiki-derived nodes have no
`chat_id` and stay outside the chat-bridge subgraph). The HTTP server's
`/save` and `/extract` endpoints already supply `chat_id` to extraction.
`occurred_at` is migrated/backfilled from `source_message_ids` for existing
databases, and can be passed explicitly to `add_episode` for imported events.

MCP tools: `add_episode` / `update_episode` / `delete_episode`.
