# Calendar memory

`CalendarMemory` stores dated commitments for the character and users. It
supports one-off events and whole weekly routine series in an IANA timezone.
Shared events have one authoritative row and an `attendees` list, so a group
chat sees the same occurrence only once.

Calendar rows live in the normal structured-memory SQLite store as
`calendar_events`, with a `calendar_index/` search index. The memory is off by
default:

```python
from character_memory import CalendarConfig, CharacterMemoryConfig, MemoryConfig

config = CharacterMemoryConfig(memory=MemoryConfig(
    enabled_calendar=True,
    calendar=CalendarConfig(timezone="Europe/Rome"),
))
```

The character's private `WorldMemory` can be attached as a live source with
`import_world_routines: true` (the default when the world is enabled). World
routines are projected into concrete calendar occurrences at recall time; they
are not copied into `calendar_events` and remain read-only from the calendar.

```python
calendar = agent.calendar_memory
calendar.create_event(
    "alice",
    title="Dentist",
    start_at="2026-08-20T09:00:00+02:00",
    end_at="2026-08-20T10:00:00+02:00",
    attendees=["_self"],
)
calendar.create_event(
    "_self",
    title="Morning run",
    kind="routine",
    weekdays=[0, 2, 4],
    start_local="07:30",
    duration_minutes=45,
)
```

`search_events()` uses a nearby window (24 hours in the past and 14 days in
the future by default), then combines text relevance with temporal proximity.
Use `start`/`before` for an explicit range and `owners` to limit visibility to
the character, participants, or events where they are attendees.

During normal agent recall, an expression such as “ieri” or “next Friday”
widens that candidate window to include the resolved interval and gives every
overlapping occurrence an additional temporal relevance score. It remains an
additional ranking signal, not a hard filter; textually relevant nearby events
can still be returned.

Calendar extraction participates in the same automatic extraction call as
facts and episodes. It only creates, updates, or cancels explicit commitments
and requires the supporting transcript message IDs. Model-facing calendar
write tools stage changes until the final assistant message is committed;
MCP exposes equivalent `search_calendar_events`, `create_calendar_event`,
`update_calendar_event`, and `cancel_calendar_event` tools.

The WebUI's Calendar page is an agenda view with date navigation, search,
owner filtering, and editing for persisted events. Live world-routine cards are
marked read-only.
