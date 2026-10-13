# `groups` — persistent group memory with configurable privacy

| | |
|---|---|
| **Class** | `GroupMemory` + `GroupDirectory` (`memory/groups.py`) |
| **Flavor** | Structured memory (directory + rolling profiles) |
| **Scope** | `CHARACTER` |
| **Learns from chat?** | **Yes** — extraction refreshes each group's rolling summary |
| **Config toggle** | `MemoryConfig.enabled_groups` (default **off**) |
| **Retrieval size** | `MemoryConfig.groups_k` (default `4` eligible group profiles) |
| **Default privacy** | `MemoryConfig.group_privacy` (default `members`) |

## What it does

A **persistent group** spans multiple chats and remembers three things: the
users observed in it, a rolling description of its conversations, and the
origin of extracted facts and episodes. Groups are scoped to a character:
`group_id` is the canonical identifier, and several chats may reference the
same one.

```python
from character_memory import CharacterAgent, CharacterMemoryConfig

agent = CharacterAgent(
    "assets/Kurisu",
    config=CharacterMemoryConfig.memory(
        enabled_groups=True, groups_k=4, group_privacy="members",
    ),
)
planning = agent.create_chat("alice", group_id="lab", title="Planning")
followup = agent.create_chat("bob", group_id="lab", title="Follow-up")
```

Chat creation binds a chat to its group **once**; later requests naming a
different identifier are rejected. Historical chats are never assigned
automatically, and creating a chat does not by itself add anyone to the
roster — **only saved messages from human speakers do** (`save=false`
retrieval changes nothing).

## Group identity

The `groups` table stores one row per group: `display_name`, an optional
`privacy` override (`NULL` inherits the character's configured policy), and
the rolling `summary`. Supplying a new identifier creates the row once,
atomically; re-ensuring an existing group never resets another chat's
metadata.

`GroupDirectory.observed_users(group_id)` derives the roster from the
distinct `user_id`s on saved `role='user'` messages across all chats linked
to the group. There is no join/leave tracking, membership administration, or
external roster sync, and no membership periods — a recorded speaker has
access to the group's entire stored history under `members`.

## Rolling summary extraction

On the normal extract interval, the batched extractor refreshes the summary
of the chat's group using the previous summary plus the new messages. The
instruction asks for recurring topics and ongoing group context, derives
names from recorded speakers rather than model guesses, and returns an empty
summary when nothing changed. Profiles are indexed lazily: the hybrid index
rebuilds only when the profile rows (summary, display name, roster) changed
since the last recall.

## Provenance stamping

Facts and episodes extracted in a linked chat are stamped with its
`group_id` while retaining `chat_id` and message provenance. Manual
`add_fact` / `add_episode` calls accept an optional `group_id`, as do the
MCP `add_fact` / `add_episode` tools (a new identifier materialises the
directory row on first use). Recalled rows carry a source-group label.

## Privacy modes

The character default (`group_privacy`) applies to every group without an
override; per-group overrides win. Eligibility is evaluated against the
current **audience**: in a linked group conversation that is the group's
observed roster, in any other conversation it is the chat's participants.

| Mode | Availability |
|---|---|
| `none` | Group summaries, facts and episodes can be recalled across conversations. |
| `members` | Every recipient must have spoken in the source group. A private conversation has one recipient. |
| `group_only` | Recall only while conversing in the source group. |

Eligible group facts and episodes are shared regardless of their **subject**:
Alice can recall a fact about Bob learned in their shared group, while Bob's
ungrouped private facts keep their existing user scoping. During recall, the
current group's profile comes first, then other eligible profiles within
`groups_k` and the prompt budget, and recalled facts/episodes are labelled
with their source group.

Eligibility is applied **before** candidate selection, sticky inclusion,
reranking and recall reinforcement, and a missing audience identity never
grants unrestricted access. The same restrictions apply to knowledge-graph
projections of grouped evidence (source-derived text, relationships,
traversal, diagnostics); existing KG privacy continues to govern ungrouped
knowledge. Deduplication and contradiction resolution — including
exact-match suppression — never merge restricted content into an ungrouped
row or another group's row: they partition by `(user_id, group_id)`, and
graph consolidation preserves source distinctions while visibility caches
invalidate on privacy or roster changes.

**Coverage boundary:** these rules cover group profiles, facts, episodes and
their graph projections. User summaries, directives, conversation-history
memories and other systems retain their existing sharing behavior; this is
not a universal conversation-privacy guarantee.

Disabling `enabled_groups` stops profile extraction and recall but **does
not** bypass privacy on already tagged facts and episodes.

## Library interface

`CharacterAgent` exposes the administrative surface (deliberately
unrestricted — privacy governs recall, not inspection):

- `agent.list_groups()` — every group with roster, privacy and linked chats.
- `agent.get_group(group_id)` — one group plus its facts and episodes.
- `agent.update_group(group_id, display_name=..., privacy=...)` — edit
  metadata; `privacy=None` clears the override. The observed roster is not
  editable.
- `agent.create_chat(user, group_id=...)` — bind a new chat to a group.

## Server surfaces

- **HTTP**: `POST /context` accepts an optional `group_id` (binding new
  chats, validated against existing ones) and returns the resolved
  identifier. `GET /api/groups/{character}`,
  `GET /api/groups/{character}/{group_id}` and
  `PATCH /api/groups/{character}/{group_id}` expose the directory; the
  admin mini-chat (`POST /api/admin/characters/{name}/chat`) accepts
  `group_id` when creating a chat.
- **MCP**: `list_groups`, `get_group` and `update_group` (category
  `groups`); `add_fact` / `add_episode` accept `group_id`.
- **Client**: `CharacterMemoryClient.context(group_id=...)` (the response
  carries `group_id`), plus `list_groups` / `get_group` / `update_group`.
- **WebUI**: a groups browser with roster, summary, linked chats, facts and
  episodes; display-name and privacy editing; group selection when starting
  a chat from the configurator.

## Persistence

One row per group in the shared structured store, created through additive
migrations (SQLite and PostgreSQL alike); legacy chats and rows stay
ungrouped. Index metadata rebuilds through the standard structured-memory
mechanisms, so wiping `.cm_data` resets profiles with everything else while
deleting only `memory.db` clears learned group state.
