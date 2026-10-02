# Multi-User (Group Chat) Handling

This document explains how `character_memory` hosts **conversations with
several human speakers** — a *group chat* — end to end. It is kept up to date
as the implementation evolves.

The short version: a chat persists one row per turn with the *actual speaker*
recorded alongside the content, every `Memory` declares whether it remembers
things per speaker or per character, and the orchestrator (the
`CharacterAgent`) fans recall and extraction out to fit. A 1:1 chat is bit-for-
bit identical to the legacy single-user path, so existing code keeps working.

## 1. Conceptual model

A `Chat` is a room. Several humans can join it, each one identified by a
`user_id` string. The chat owner (the user that created it) is just one of the
participants — distinguished only by being guaranteed to be in the
participant list whether they have spoken yet or not. The character speaks
separately (model outputs, `role="assistant"`) and is *not* a participant.

When the agent answers, it sees **the participants**, not "the user", and
every per-user memory is asked for the slice that belongs to each one. The
character's own knowledge (wiki, example dialogues, heartbeat) is asked
exactly once and contributes a single section to the prompt.

## 2. Chat persistence — who spoke each turn

The shared `SQLiteStore` (the same `memory.db` that holds the structured
memories) carries the chat tables. From `character_memory/chat.py`:

```python
_CHAT_COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "user_id": "TEXT NOT NULL",          # the chat OWNER (creator)
    "title": "TEXT NOT NULL DEFAULT ''",
    "created_at": "REAL NOT NULL",
}

_MESSAGE_COLUMNS: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "chat_id": "TEXT NOT NULL",
    "role": "TEXT NOT NULL",
    "content": "TEXT NOT NULL",
    "user_id": "TEXT",                   # the SPEAKER of this turn (NULL on legacy rows)
    "created_at": "REAL NOT NULL",
    "extracted": "INTEGER NOT NULL DEFAULT 0",
}
```

`chats.user_id` is the **owner** — the user who created the chat.
`messages.user_id` is the **speaker** of that particular turn. For a 1:1 chat
both columns hold the same string; for a group chat the per-row `user_id`
records who actually spoke.

`Chat` exposes the participant set the orchestrator needs:

```python
def participants(self) -> list[str]:
    """Distinct human speakers in this chat, oldest-first.

    Speakers come from the ``user_id`` of user-role messages; legacy rows
    with a NULL ``user_id`` count as the chat owner. The owner is always
    included. For a 1:1 chat this returns ``[self.user_id]``.
    """
```

Note three deliberate behaviours:

1. **Oldest-first ordering**: `seen.insert(0, self.user_id)` puts the chat
   owner at the front only if they would otherwise be missing — so the order
   is "owner (if present), then the first other speaker, then …" rather than
   a re-sort.
2. **Legacy-`NULL` fallback**: rows written before the `messages.user_id`
   column existed have `NULL` there, and `uid = r.get("user_id") or
   self.user_id` treats them as the owner. The migration is additive and
   idempotent:

   ```python
   # _ChatBackend.__init__
   if "user_id" not in self.store.columns("messages"):
       self.store.execute("ALTER TABLE messages ADD COLUMN user_id TEXT")
   ```

3. **Assistant turns are unowned**: `Chat.add_message("assistant", …)`
   defaults `user_id=None` for assistant messages; only human turns carry a
   `user_id`. This is why `participants()` filters on `role="user"`.

### OpenAI-safe speaker labels

When the chat is sent back to a Chat-Completions-style API, every user turn
in a multi-user chat gets an OpenAI `name` field naming the speaker. The
Chat-Completions API only accepts `^[A-Za-z0-9_-]+` for that field, so
`character_memory` filters the `user_id` through it transparently:

```python
# OpenAI-style `name` field only allows alphanumerics, underscore, hyphen.
_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]+")

def _safe_name(user_id: str) -> Optional[str]:
    name = _NAME_SAFE.sub("", user_id or "").strip("_-")
    return name or None
```

A Discord user named `**Dark_Knight 99**` becomes `Dark_Knight99` for the
extraction prompt — the library never crashes the model on a quirky
identifier.

## 3. Memory scopes — the per-user / character split

The core abstraction is `MemoryScope`, declared as a class attribute on every
memory (see `character_memory/memory/base.py`):

```python
class MemoryScope(str, Enum):
    """How a memory relates to the participants of a conversation."""

    PER_USER = "per_user"    # recall once per participant, group by speaker
    CHARACTER = "character" # recall once, ignore participants entirely
```

Each memory declares its own scope. The agent reads it and decides what to
do. The standard memories and their scopes:

| Memory | Scope | Stores… |
|---|---|---|
| `character_info`   | `CHARACTER` | the character's wiki/story markdown chunks |
| `dialogue_style`   | `CHARACTER` | example exchanges, used as a style reference |
| `heartbeat`        | `CHARACTER` | the character's own discoveries / actions |
| `user_facts`       | `PER_USER`  | facts about a user (and general facts) |
| `user_directives`  | `PER_USER`  | standing instructions *from* a user |
| `episodic`         | `PER_USER`  | events the character experienced with a user |
| `emotion`          | `PER_USER`  | per-user emotion dims + relationship comment |
| `user_summary`     | `PER_USER`  | one consolidated profile per user |
| `knowledge_graph`  | `PER_USER`  | graph whose PersonNodes are users; biased by `user_id` on retrieve |

Declarative, not hard-coded — adding a new memory is just a question of
setting (or omitting) the right `scope = …`. The orchestrator changes
nothing.

## 4. Multi-user recall — fans out and groups

The optional [global memory budget](memory_budget.md) is shared across all
participants and memory types, after the existing privacy filtering and
per-memory retrieval. It is not allocated separately per speaker. Participant
labels count toward the cap, and no speaker is guaranteed a reserved share.
Use `agent.recall(chat, budget=3000)` or pass the same budget to generation or
HTTP `/context`. Selected items retain their speaker metadata.

Every `Memory` keeps a single-user `recall(query, user_id, limit, …)` method
as its workhorse. The `Memory` base class layers three convenience methods
on top of it (also in `memory/base.py`):

```python
def recall_participants(self, query, participants, limit, state_changing=True, *,
                        temporal_resolution=None, temporal_resolution_engine=None,
                        temporal_weight=None):
    """Recall items for the conversation's participants.

    * PER_USER  scope: recall once per participant (each gets up to `limit`);
      the speaker is carried in each item's metadata.
    * CHARACTER scope: recall once (the memory is not about any one
      participant); the first participant is passed to `recall` as a dummy
      `user_id` and ignored by the implementation.
    """
    if not participants:
        return []
    if len(participants) == 1:
        return self._recall_with_temporal(query, participants[0], limit,
                                          state_changing=state_changing, ...)
    if self.scope is MemoryScope.CHARACTER:
        return self._recall_with_temporal(query, participants[0], limit,
                                          state_changing=state_changing, ...)
    items: list[MemoryItem] = []
    for uid in participants:
        items.extend(self._recall_with_temporal(query, uid, limit,
                                                state_changing=state_changing, ...))
    return items
```

The `...` in each branch stand for the three temporal kwargs, passed through
unchanged. `_recall_with_temporal` forwards them to `recall` **only when the
memory sets `supports_temporal_resolution = True`** — otherwise it makes the
plain legacy `recall` call, so subclasses that predate (or opt out of)
temporal resolution keep working.

The **single-participant branch is the fast path** and is identical to the
legacy single-user call — that is what guarantees 1:1 chats are bit-for-bit
unchanged.

`build_section_participants` is the rendering pipeline. It is what
`Character.build_context` calls when more than one participant is present:

```python
# memory/base.py (build_section_participants_result, simplified)
def build_section_participants(self, query, participants, limit, state_changing=True, *,
                               temporal_resolution=None, temporal_resolution_engine=None,
                               temporal_weight=None):
    if not participants:
        return None
    if len(participants) == 1:
        # Identical to the legacy single-user path.
        return self.build_section(query, participants[0], limit, state_changing=state_changing)
    if not self.enabled:
        return None
    if self.supports_temporal_resolution:
        items = self.recall_participants(query, participants, limit,
                                         state_changing=state_changing,
                                         temporal_resolution=temporal_resolution,
                                         temporal_resolution_engine=temporal_resolution_engine,
                                         temporal_weight=temporal_weight)
    else:
        items = self.recall_participants(query, participants, limit,
                                         state_changing=state_changing)
    if not items:
        return None
    if self.scope is MemoryScope.CHARACTER:
        return self.format(items)
    return self.format_grouped(items, participants)
```

The temporal kwargs reach an overridden `recall_participants` **only when the
memory sets `supports_temporal_resolution = True`**; non-opt-in overrides keep
the pre-temporal `(query, participants, limit, state_changing)` signature.
`EmotionStatus` — the one PER_USER memory that overrides
`recall_participants` — accepts and ignores the kwargs: current emotion is a
snapshot, so temporal search does not apply to its dimensions.

The default `format_grouped` buckets each `MemoryItem` by the speaker stored
in its metadata and renders an `About {uid}:` block per participant. The
order is *participant order from the call*, then any remaining speakers
trailing — so the layout is stable across turns.

### Section headers go plural

`Character.build_context` swaps to a `*_header_multi` header field on
`PromptConfig` when a section is rendered for more than one user, falling
back to the singular header otherwise. From `prompts.py`:

```python
user_facts_header:      str  = "What you remember about this user"
user_facts_header_multi: str  = "What you remember about these users"

user_directives_header:      str  = "Standing instructions from this user"
user_directives_header_multi: str  = "Standing instructions from these users"

episodic_header:      str  = "Episodes you've shared with this user"
episodic_header_multi: str  = "Episodes you've shared with these users"

emotion_header:      str  = "Emotional state"
emotion_header_multi: str  = "Emotional state (baseline + toward each user)"
```

Override any of these from your `config.yaml` to rename the rendered
section.

### The Emotion special case

`EmotionStatus` is the only `PER_USER` memory whose baseline is
user-independent — the character's resting `joy / sadness / anger /…
` belongs to *the character*, not to any participant. To avoid that baseline
being duplicated under every user's block, `EmotionStatus` overrides
`recall_participants`/`format_grouped` (`character_memory/memory/emotion.py`):

```python
# recall once per participant, but tag each item with the speaker so the
# grouper can route it; baseline items get {"emotion": "baseline"} (no user_id).
for k, v in self.baseline.items():
    items.append(MemoryItem(text=f"{k}={v:.2f}", … metadata={"emotion": "baseline"}))
for uid in participants:
    state = self.get_user_state(uid)
    for k, v in state.items():
        items.append(MemoryItem(text=f"{k}={v:.2f}", … metadata={"user_id": uid}))
```

```python
# format_grouped override:
baseline: list[MemoryItem] = []
by_user: dict[str, list[MemoryItem]] = {}
for it in items:
    uid = it.metadata.get("user_id")
    if isinstance(uid, str) and uid:
        by_user.setdefault(uid, []).append(it)
    else:
        baseline.append(it)

blocks = []
if baseline:
    blocks.append("Baseline:\n" + "\n".join(f"- {it.text}" for it in baseline))
for uid in order:
    body = "\n".join(f"- {it.text}" for it in by_user[uid])
    blocks.append(f"Toward {uid}:\n{body}")
```

The rendered prompt therefore reads:

```
Baseline:
- neutral=0.50
- joy=0.20
- sadness=0.10

Toward francesco:
- affection=0.42
- valence=0.10
- relationship=friend

Toward michael:
- affection=0.15
- valence=-0.05
- relationship=colleague
```

Note that the per-user dim text is bare (`affection=0.42`, *not*
`affection(toward francesco)=0.42` as the single-user `recall()` would
emit): the `Toward {uid}:` block header already disambiguates the speaker.

Similarly, the `KnowledgeGraphMemory` (`scope = PER_USER`) biases the query
by `user_id` on `retrieve(user_id=…)`, so the character's own PersonNode +
that user's relation edges get first-class activation.

## 5. Multi-user extraction — per-speaker attribution

A 1:1 chat's extraction path is unchanged. The same is **almost** true for
group chats — the change is that the prompt + schema now stamp each
extracted item with the participant the item is about, so a fact about
Alice never gets filed under Bob's profile.

### The `ExtractionContext` is the switch

`Character._extraction_context` populates the `ExtractionContext`'s
`participants` field. `Character.extract` passes that context to
`build_extraction`. From `character.py`:

```python
def extract(self, turns, user_id="default", llm=None, participants=None):
    parts = participants if participants else None
    context = self._extraction_context(user_id, participants=parts)
    participating = [
        (mem, spec)
        for mem in self.memories
        if mem.enabled
        for spec in [mem.extraction_spec(context)]
        if spec is not None
    ]
    …
```

`ExtractionContext.multi_user` is the boolean switch (`len(participants) > 1`)
that flips the prompt render into multi-user mode.

### Schemas gain a `user_id` enum

LLM-extraction calls blend the `ExtractionSpec`s from every participating
memory into one combined JSON schema. The multi-user augmentation lives in
`_augment_per_user_schema` (`character_memory/memory/extract.py`):

```python
def _augment_per_user_schema(schema, participants):
    if not participants or schema.get("type") != "array":
        return schema
    items = schema.get("items")
    if not isinstance(items, dict) or items.get("type") != "object":
        return schema
    new_items = dict(items)
    props = dict(new_items.get("properties") or {})
    props["user_id"] = {"type": "string", "enum": list(participants)}
    new_items["properties"] = props
    required = list(new_items.get("required") or [])
    if "user_id" not in required:
        required.append("user_id")
    new_items["required"] = required
    new_schema = dict(schema)
    new_schema["items"] = new_items
    return new_schema
```

Every `ExtractionSpec` declares whether its items are per-user via the
`per_user: bool` flag (see `character_memory/memory/base.py`). The
augmentor only touches schemas whose `per_user=True` *and* we are in
multi-user mode *and* the schema is an array of objects. Non-array
schemas pass through unchanged, and so do array schemas whose items are
not objects (e.g. a spec whose items are bare strings).

If the spec already declares a `user_id` property in its items, the
augmentor **overrides** it with the enum-restricted version rather than
adding a second field. This is exactly what the `emotion` spec relies on
— its multi-user schema is `[{user_id, deltas, comment?}]` and the
augmentor simply rewrites the `user_id` entry in place.

Five structured-memory subclasses set `per_user=True`: `user_facts`,
`user_directives`, `episodic`, `user_summary`, and `emotion` (whose spec
flips into the array-of-objects form only when
`len(participants) > 1`). The JSON-object a per-user field turns into
inside the combined extraction schema is:

```json
{ "type": "object",
  "properties": { "type":        {"type":"string"},
                  "content":     {"type":"string"},
                  "importance":  {"type":"number"},
                  "confidence":  {"type":"number"},
                  "user_id":     {"type":"string",
                                  "enum": ["Alice","Bob","francesco"]} },
  "required": ["type","content","importance","confidence","user_id"] }
```

### Multi-user instruction note

`PromptConfig.extraction_multi_note` is appended to the extraction
instruction whenever `context.multi_user` is true. From `prompts.py`:

```python
extraction_multi_note: str = (
    "This conversation has several participants: {participants}. For each "
    "per-user field, set the item's `user_id` to the participant the item is "
    "about (one of the listed names). Only attribute an item to someone when "
    "the conversation actually establishes it about them."
)
```

`build_extraction` interpolates `{participants}` with the comma-joined
real names from the `ExtractionContext`.

### Transcript labelling

`Extractor.extract` labels each user turn in the transcript with its real
speaker. From `character_memory/memory/extract.py`:

```python
multi = bool(context and context.multi_user)
for t in turns:
    if t.get("role") == "user":
        speaker = (t.get("user_id") or user_name) if multi else user_name
        lines.append(f"{speaker}: {content}")
    else:
        lines.append(f"{char_name}: {content}")
```

A group-chat transcript therefore reads `Alice: …`, `Bob: …`,
`Kurisu: …`, `Alice: …` rather than a generic `User:` / `Character:` dance.

### Per-memory write-back

Each structured-memory subclass uses the per-item `user_id` on `apply_extraction`,
falling back to the caller's default `user_id` when the item omits it. From
`user_facts.py`:

```python
def apply_extraction(self, value, user_id):
    added: list[MemoryItem] = []
    for f in value or []:
        content = (f.get("content") or "").strip()
        if not content or self._has_text(user_id, content, "content"):
            continue
        uid = str(f.get("user_id") or user_id)
        if uid != user_id and self._has_text(uid, content, "content"):
            continue
        row_id = self.add_fact(uid, content, …)
        …
```

The same pattern is in `user_directives.py`, `episodic.py`, and
`user_summary.py` (the summary memory *requires* a `user_id` — its spec's
`instruction` explicitly says "produce one item per person").

The `EmotionStatus` make the same path work for its `[{user_id, deltas,
comment?}]` array:

```python
def apply_extraction(self, value, user_id):
    if isinstance(value, list):                       # multi-user
        for entry in value:
            uid = str(entry.get("user_id") or user_id)
            deltas = entry.get("deltas")
            if isinstance(deltas, dict):
                self.update(uid, deltas)
            comment = entry.get("comment")
            if isinstance(comment, str) and comment.strip():
                self.set_user_comment(uid, comment)
    elif isinstance(value, dict):                     # single-user
        …
```

The `comment` is still emitted only when the relationship genuinely changes,
so a stable `friend`/`colleague` label is preserved across turns.

## 6. Target resolution — every API surface

`CharacterAgent._resolve_target` (in `agent.py`) is what computes the
participants list once, before any memory is queried:

```python
def _resolve_target(self, target, user_id):
    """Return (query, user_id, prior_messages, participants) for a target.

    - Chat               -> (last user msg or "", chat.user_id, history, chat.participants())
    - chat id str        -> same, after load_chat
    - raw query str      -> (target, user_id, [], [user_id])
    - list[dict] msgs    -> (last user content or "", user_id, target, [user_id])

    Participants come from a persisted Chat (every human speaker in it).
    Raw query / message-list targets have no stored speakers, so they are
    treated as single-user with ``[user_id]``.
    """
```

Three API surfaces take a target: `build_context`, `render_prompt`, and
`generate_answer`. Each calls `_resolve_target`, fans out (1 participant
→ legacy path, >1 → participants-aware path), and propagates the
participants to `Character.build_context` / `Extractor.extract`.

The practical consequence:

* `agent.generate_answer(chat_id)` and `agent.generate_answer(chat)` →
  participants come from `Chat.participants()`.
* `agent.generate_answer(messages_list, user_id="alice")` →
  participants are `[user_id]`. There is no persisted `Chat` to fall back
  on, so callers who want group behaviour must pass a `Chat` (or its id).
* `agent.build_context(chat_id, user_id=…)` → same as above; the `user_id`
  argument is the **default user** / chat owner for that call.

### Per-chat auto-extraction stays correct

`CharacterAgent._extract_chat` runs on the configured `extract_interval`
of user turns. It always calls `Chat.participants()` so a group chat keeps
its real participant set — meaning every batch the extractor sees is
freshly labelled and every per-user item is correctly attributed:

```python
def _extract_chat_unlocked(self, chat):
    window = max(self._extract_interval() * 2, 4)
    rows = chat.unextracted()
    if not rows:
        return
    # chat.id is threaded into apply_extraction so chat-scoped memories
    # (user_facts, episodic) stamp their rows with it; the knowledge graph
    # then links those rows via ChatEdge.
    self._extract_messages(rows[-window:], chat.user_id, participants=chat.participants(), chat_id=chat.id)
```

`extract()` (no argument) loops over all chats **per chat, not per user**,
so each chat keeps its own participant set for an extraction call.

## 7. Cross-cutting concerns

### Deduplication is per-user automatically

`Deduplicator.dedup_items` and `Deduplicator.sweep` both narrow candidate
rows via the memory's RAG index with `where={"user_id": user_id}` (see
`character_memory/memory/dedup.py`). A fact about Alice is therefore only
compared to Alice's other facts — there is no cross-talk between
participants, even when both have a fact called *"likes pizza"*. The
self-test for a duplicate ("Alice has an exam on the 17th of July") also
runs only against Alice's existing rows.

### Knowledge graph nodes per participant

Every `user_summary` row becomes a `PersonNode`, keyed
`person:<user_id>`. Stable IDs mean Person↔Person edges
(`TransitionEdge`, `CoOccurrenceEdge`) are meaningful — Alice's
relationship with Bob never leaks into Alice's relationship with Carol.

Knowledge-graph ingestion reads the source memories through their public
APIs and never mutates them. The graph is **per-character, shared across
users**, so co-occurrence and transition edges are meaningful across the
whole group. For per-user attribution details see `docs/knowledge_graph.md`
(section 1: *Nodes* / *Edges*, section 2: *Ingestion* — the
"Context-aware, relevance-filtered extraction" subsection).

### Prompt-level transparency

For every section the prompt template is the same; only the header and the
body content change. A user sumary rendered for two participants reads:

```
## Summaries of these users
About francesco:
- Francesco (aka Frank): close collaborator, working on the memory library.

About michael:
- Michael (aka Mike): worked together on the Acchan Discord bot.
```

…whereas the same memory rendered for one participant reads:

```
## Summary of this user
- Francesco (aka Frank): close collaborator, working on the memory library.
```

The `<memory_name>_header_multi` plural headers in `PromptConfig` are
separate fields from the singular ones; override just the plural when you
want different wording in group chats.

## 8. Important gotchas

1. **Legacy `NULL` `user_id`** — older `memory.db` files predate the
   `messages.user_id` column. The additive `ALTER TABLE messages ADD
   COLUMN user_id TEXT` migration in `_ChatBackend.__init__` runs once on
   startup, guarded by `PRAGMA table_info`. Reads treat `NULL` as the
   chat owner. Fresh DBs skip the migration silently.

2. **OpenAI-safe `name` filtering** — `Chat.messages()` (used when sending
   the transcript back to the LLM) filters every group-chat user turn's
   `user_id` through `_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]+")`. The
   resulting `name` field is never sent stripped empty: a `name=None`
   silences the field rather than send `""`.

3. **Assistant turns have `user_id=None`** — in the schema that means an
   `_extract_messages` reader (`r.get("user_id")`) for the LLM transcript
   never mistakes an assistant turn for a participant speaker. `Extractor.
   extract` only labels `role="user"` turns.

4. **Single-participant fast path is exact** — keep it that way. Any
   override of `recall_participants` should still return the same items
   the legacy `recall` (with `len(participants)==1`) returns; otherwise a
   1:1 chat will silently drift from the legacy behaviour.

5. **Default `user_id="default"` is a string** — counting "default" as a
   *real* user is a footgun. Either pass the chat owner consistently on
   the agent's surface, or accept that raw-message-list / bare-query
   targets produce participants equal to `[user_id]` (and one participant
   is still single-user).

6. **Emotion's baseline is not a participant** — `EmotionStatus` overrides
   `recall_participants` / `format_grouped` precisely so the baseline
   appears once. Don't try to "fix" this by moving the baseline into
   per-user dims; that would duplicate it under every participant.

7. **Dedup is per user** — `Deduplicator.sweep(None)` groups rows by
   distinct `user_id` (because `DedupConfig.per_user=True` is the
   default); passing `user_id=…` sweeps a single user. There is no global
   "sweep every user's facts together" path; don't expect one.

8. **`HeartbeatJournal` ignores `user_id` on purpose** — its rows are
   tagged `user_id="_self"` on insert; recall drops the user filter and
   queries the whole table. Combined with `scope = CHARACTER` this means
   a group chat's heartbeat section is the same as a 1:1 chat's, shared
   across every user. This is deliberate and not a bug.

## 9. End-to-end data flow

A single user turn in a group chat with `n` participants:

1. **Client** → `POST /context {character, user, message, chat_id?}`. The
   `user` field is the **current speaker**. A chat id is an unguessable
   room key: any caller holding it may post as any speaker, which is what
   enables group chats.
2. **Server** (`character_memory/server/api.py`) locates or creates the
   `Chat`, then `chat.add_message("user", message, user_id=req.user)`. The
   per-turn `user_id` is what makes each participant's messages
   attributable.
3. **Server** calls `agent.build_context(chat)` → `CharacterAgent._resolve_
   target(chat)` → `chat.participants()`. The participants list includes
   every distinct speaker in the chat (and the owner first if they spoke).
4. **`Character.build_context`** sees `len(participants) > 1` and uses
   `Memory.build_section_participants` per memory. `PER_USER` memories
   recall once per participant and the result is grouped by speaker;
   `CHARACTER` memories recall once and contribute a flat section.
5. **Client** uses the rendered context to make its own LLM call.
6. **Client** → `POST /save {chat_id, answer}`. The assistant answer is
   persisted with `chat.add_message("assistant", answer)`. Once the chat has
   `extract_interval` user turns since its last extraction (or when the
   client calls `POST /extract`), the extractor loads each memory's `ExtractionSpec`, builds one
   combined JSON schema with `user_id` enums on `per_user=True` specs,
   renders the multi-user extraction note + per-speaker transcript, makes
   one `chat_structured` LLM call, and hands each memory the items it
   asked for.
7. **Each memory** runs its `apply_extraction(value, user_id=chat.
   user_id)`. Per-user memories attribute every item by `item["user_id"]`
   (or fall back to the chat owner if the LLM omitted it). The
   deduplicator compacts the freshly-added items against each memory's
   existing rows. The knowledge graph ingests the new items and mirrors
   dedup mutations.
8. **Persist**: `agent.persist_structured()` writes the rewritten
   hybrid indexes; the KG is persisted alongside.

## 10. Verified behaviour

The implementation is exercised inline (there is no formal test suite — see
`CLAUDE.md`). The following behaviours are confirmed:

- **Single-user is unchanged**: a 1:1 chat takes the
  `len(participants)==1` fast path everywhere (`recall_participants`,
  `build_section_participants`, the extraction `multi_user` switch). The
  prompt header stays singular; the schema has no `user_id` enum augment.
  All five layers — recall, rendering, extraction, dedup, knowledge graph
  — are bit-for-bit the legacy single-user path.
- **Multi-user recall is grouped**: `Memory.build_section_participants`
  yields an `About {uid}:` block per participant in the order they were
  given; `CHARACTER` memories contribute a single flat section regardless
  of participant count.
- **Multi-user extraction attributes correctly**: `user_facts`,
  `user_directives`, `episodic`, and `user_summary` all read each item's
  `user_id` (the LLM-stamped speaker) and fall back to the chat owner
  when absent. Misattribution requires the LLM to disagree with the
  prompt — the schema's required `user_id` enum helps prevent that.
- **Emotion folds the baseline once**: a group chat renders the baseline
  block followed by one `Toward {uid}:` block per participant (no
  duplicates).
- **Dedup stays per-user**: `Deduplicator.dedup_items` and
  `Deduplicator.sweep` both narrow candidates with
  `where={"user_id": user_id}`. Cross-user duplication is impossible by
  construction.
- **Knowledge graph nodes are stable**: a user gets a `person:<user_id>`
  node scoped per-character; per-user facts/episodes get `FactEdge` /
  `EpisodeEdge` to that PersonNode. Co-occurrence and transition edges
  connect participants meaningfully (`Alice ↔ Bob` is a real edge) and
  deduplication of a fact removes/refreshes its node in lockstep
  (see `docs/knowledge_graph.md`).
- **Chat persistence migration**: a `memory.db` without the
  `messages.user_id` column picks it up on first run via an `ALTER TABLE`
  guarded by `PRAGMA table_info`; legacy reads return `NULL` and the
  helper code falls back to the chat owner.
- **OpenAI-safe `name`**: a `Chat.messages()` payload passes through
  `_safe_name` so a quirky `user_id` like `**Dark_Knight 99**` becomes
  `Dark_Knight99` in the eventual LLM transcript; fully scrubbed names
  become `None` and are dropped from the message dict.
- **Kurisu / group-chats**: the bundled Kurisu character keeps the
  configurable `enabled_knowledge_graph` flag; multi-user extraction
  works over `assets/Kurisu/.cm_data/memory.db` once any second user
  writes a turn.

## 11. Quick reference — the API

```python
from character_memory import (
    CharacterAgent,                              # build/render/extract
    CharacterMemoryConfig,
    Chat,                                        # persisted conversation
    Memory, MemoryItem, MemoryScope,             # base + scope
    ExtractionSpec, ExtractionContext,           # extraction API
)

# Construct once
agent = (
    CharacterAgent(directory="assets/Kurisu", name="Kurisu")
    .load_from_config("assets/Kurisu/config.yaml")
)

# Create / load a chat — participants are inferred from stored messages.
chat = agent.create_chat("francesco", title="Brainstorming")   # owner=francesco
chat.add_message("user", "Hi Kurisu!", user_id="francesco")
chat.add_message("user", "I'm Michael.", user_id="michael")    # 2nd speaker
chat.add_message("user", "Quick note — I'm Alice.", user_id="alice")  # 3rd
# chat.participants() -> ['francesco', 'michael', 'alice']

# Generate an answer. _resolve_target(chat) returns participants
# ['francesco','michael','alice']; group-chat extraction runs automatically.
reply = agent.generate_answer(chat)

# Per-chat extraction (idempotent — the chat's `extracted` flag
# marks processed messages):
agent.extract(chat)

# A raw query without a Chat is treated as single-user. Pass a Chat
# (or chat id) to enable group behaviour.
context = agent.build_context(chat, user_id="francesco")
prompt = agent.render_prompt(chat, user_id="francesco")
```
