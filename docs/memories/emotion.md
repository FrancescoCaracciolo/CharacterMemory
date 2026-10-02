# `emotion` — baseline vector + per-user relationship dims

| | |
|---|---|
| **Class** | `EmotionStatus` (`memory/emotion.py`) |
| **Flavor** | Custom (a bespoke SQLite table, **not** a `StructuredMemory`) |
| **Scope** | `PER_USER` (per-user dims; baseline is CHARACTER-ish — surfaced once) |
| **Learns from chat?** | **Yes** — absolute current mood + signed relationship deltas |
| **Config toggle** | `MemoryConfig.enabled_emotion` |
| **Retrieval size** | n/a (every dim is always surfaced) |

## What it does

Tracks **how the character feels** on three layers:

1. A fixed **baseline** — user-independent: `joy`, `sadness`, `anger`,
   `anxiety`, `surprise`, `neutral`. This is the character's resting state.
2. A persisted, character-wide **current mood** snapshot (same axes, initially
   equal to the baseline). Each extraction replaces it absolutely.
3. A set of **per-user dimensions** (default `affection`, `valence`,
   `trust`) — how the character feels *toward each specific user*, clamped to
   `[-1, 1]` and updated by signed deltas. Plus a per-user **`comment`** — a
   short relationship descriptor (colleague / friend / rival / mentor /
   partner / conflicting / …).

The mood axes and relationship dimensions are configurable via `MemoryConfig.emotion_baseline` and
`MemoryConfig.emotion_user_dims`.

## Storage

One row per user in the `emotion` table (PK `user_id`). The `state` column
holds a JSON blob of that user's dims + comment. The baseline lives in config;
the persisted `current_mood` snapshot is stored in `emotion_state`.
`SQLiteStore.upsert` is called with
`pk="user_id"`.

## Recall

Direct `recall` returns the full state, ignoring the per-memory `limit` and
without decay. When a [global budget or reranker](../memory_budget.md) is active,
emotion dimensions compete with other candidates; only selected dimensions
appear in the prompt:

```
baseline.joy=0.20
current.joy=0.20
sadness=0.10
…
affection(toward michael)=0.40
valence(toward michael)=0.30
trust(toward michael)=0.20
relationship(toward michael)=colleague
```

In a **group chat**, `recall_participants` surfaces the baseline **once**,
then each participant's per-user dims + comment (the only `PER_USER` memory
that needs this special handling — otherwise the baseline would be duplicated
per participant). `format_grouped` renders it as:

```
Baseline:
- joy=0.20
- sadness=0.10

Toward michael:
- affection=0.40
- valence=0.30
- relationship=colleague

Toward okabe:
- affection=-0.10
- …
```

## Extraction

`EmotionStatus.extraction_spec` returns an object with a complete absolute
`current_mood` vector and `users: [{user_id, deltas, comment?}]`. The mood
uses configured baseline axes with `0..1` values; relationship deltas remain
signed and are emitted only for affected users.

The `comment` is a coarse label, not a delta. To avoid the extractor
rewriting it every turn with synonyms, it is **only emitted** when the
conversation establishes or clearly changes the relationship — otherwise the
key is omitted so a stable label is preserved. See the `_COMMENT_NOTE`
constant in the source.

`apply_extraction`:

- Every extraction first replaces `current_mood`, then applies
  `update(user_id, deltas)` (preserves the comment), then
  `set_user_comment(user_id, comment)` if one was provided.
- Group chat: iterates the list, applying each entry to its own user.

## Section header

`"Emotional state"` (singular) /
`"Emotional state (baseline + toward each user)"` (group chat). Overridable
via `PromptConfig.emotion_header[_multi]`.

## Direct manipulation

```python
mem.update("michael", {"affection": 0.1, "trust": 0.05})     # signed deltas, clamped to [-1, 1]
mem.set_user_state("michael", {"affection": 0.5, "valence": 0.3, "trust": 0.4})  # replace dims
mem.set_user_comment("michael", "trusted colleague")         # replace label
state  = mem.get_user_state("michael")                       # numeric dims only
comment = mem.get_user_comment("michael")                    # label or ""
```

The `emotion_note` template in `PromptConfig` interpolates `{baseline}` and
`{user_state}` if you want to inject a one-liner above the section.

## MCP tools

`set_character_emotion` replaces the character's own persisted current mood;
`get_character_emotion` returns that mood and the configured resting baseline.
Neither requires a `memory` or `user_id` argument.

`set_user_emotion` (optional `deltas`, absolute `current_mood`, and optional
`comment`) and `get_user_emotion` (returns baseline + current mood + per-user
dims + comment) remain available for relationship state.
