# Automatic temporal resolution

Temporal resolution lets recall understand questions such as **“Cosa hai
fatto ieri?”**, **“What happened last week?”**, or **“昨日何をしましたか？”**.
The phrase is resolved to a half-open UTC interval and memories whose semantic
event timestamp overlaps that interval receive an additional ranking signal.
It is enabled by default and does not make a network request.

## Default flow

For each recall turn, `CharacterAgent`:

1. builds the normal history-aware query (current user message plus recent
   user messages weighted by `retrieval_recency_decay`);
2. resolves all temporal expressions **once** with the configured engine;
3. passes the same immutable `TemporalResolution` to every memory;
4. unions semantic candidates with timestamp matches and ranks them together.

The reference time is the latest user message's `occurred_at`, then its
`created_at`, then the current time. This matters when importing an old chat:
“yesterday” is relative to when that source message happened, not when it was
imported.

No expression, a disabled engine, or an engine error is fail-open: recall is
the same semantic/BM25 path as before.

## Engines

### `dateparser` (default)

`DateParserTemporalResolutionEngine` uses precompiled locale dictionaries from
`dateparser`; it uses no LLM and no embeddings. The common relative-expression
path normally completes in well under a few milliseconds after construction.
The built-in language set covers 30 languages, including Italian, English,
Spanish, French, German, Portuguese, Slavic languages, Arabic, Hebrew, Hindi,
Chinese, Japanese, Korean, Indonesian, Vietnamese, and Thai.

Set `languages` to ISO language codes to reduce cold-start work, or to `["*"]`
to load every dateparser locale.

### `llm` (opt-in)

`LLMTemporalResolutionEngine` makes exactly one `LLMClient.chat()` call per
recall turn, asks for all ranges in one JSON response, and validates/parses the
JSON locally. It deliberately does not use `chat_structured()`, because a
structured client may retry malformed output and violate the one-call
contract. The LLM endpoint is configured separately from the answer/extraction
model.

## Configuration

```yaml
temporal_resolution:
  enabled: true
  engine: dateparser             # dateparser | llm | registered custom name
  timezone: Europe/Rome          # IANA timezone used for calendar boundaries
  weight: 1.0                    # 0..1 contribution to recall ranking
  languages: [it, en]            # [] = broad fast defaults; ["*"] = all locales
  llm:                            # inert unless engine: llm
    base_url: http://127.0.0.1:9999/v1
    api_key: anything
    model: temporal-model
    timeout: 30
    max_tokens: 768
```

The same fields are available through `TemporalResolutionConfig` and
`TemporalLLMConfig`. Environment variables use
`CM_TEMPORAL_RESOLUTION_*` and `CM_TEMPORAL_LLM_*`; see
[Configuration](configuration.md).

## Direct memory use

The timestamp-aware built-ins also support the explicit API proposed for
standalone memory calls:

```python
from character_memory import DateParserTemporalResolutionEngine

items = memory.recall(
    "Cosa hai fatto ieri?",
    user_id="alice",
    limit=5,
    temporal_resolution=True,
    temporal_resolution_engine=DateParserTemporalResolutionEngine(["it"]),
)
```

`temporal_resolution=True` resolves once for that memory. For several
memories, resolve once yourself and pass the result to each call:

```python
resolution = engine.resolve(query, reference_time=source_timestamp,
                            timezone="Europe/Rome")
items = memory.recall(query, "alice", 5, temporal_resolution=resolution)
```

Agent-level injection is also supported:

```python
agent = CharacterAgent(
    "assets/Kurisu",
    temporal_resolution_engine=MyTemporalEngine(),
)
```

## Timestamp policy

Only timestamps that mean **when the remembered event happened** participate.
Storage timestamps are not treated as event time for profiles or facts.

| Memory/source | Temporal interval |
|---|---|
| `conversation_events` | `occurred_at` (backfilled from source message `created_at`) |
| `episodic` | `occurred_at`, derived from `source_message_ids` |
| `heartbeat` | journal `created_at` |
| `world` | `occurred_at` for immutable `record_type=event` only |
| `calendar` | occurrence `start_at`–`end_at`, including recurring/live sources |
| `knowledge_graph` | episode timestamp and projected timestamp-aware source rows |
| facts, directives, summaries | no temporal match by default |

Old episodic databases are migrated additively: an `occurred_at` column is
created and existing rows are backfilled when source messages are available.

## Ranking and diagnostics

For structured memories, semantic and temporal relevance are in `[0, 1]`:

```text
weighted_temporal = config.weight × temporal_match
combined = 1 - (1 - semantic) × (1 - weighted_temporal)
```

This makes a full timestamp overlap comparable to a strong semantic match,
while partial independent signals reinforce each other. It is not a hard
date filter. Calendar occurrences and graph nodes use the same principle in
their native ranking models.

Recalled `MemoryItem.metadata` exposes `semantic_relevance`,
`temporal_relevance`, `combined_relevance`, `temporal_expression`,
`temporal_range_start`, `temporal_range_end`, and `temporal_grain` when
applicable. `ContextSnapshot.temporal_resolution` contains the single
turn-level resolution used by all memories.

## Custom engines

Subclass the public ABC and optionally register a factory name:

```python
from character_memory import (
    TemporalResolutionEngine,
    register_temporal_resolution_engine,
)

class MyTemporalEngine(TemporalResolutionEngine):
    name = "my_engine"

    def resolve(self, message_or_history, *, reference_time=None, timezone="UTC"):
        ...  # return TemporalResolution

register_temporal_resolution_engine("my_engine", MyTemporalEngine)
```

External `Memory` subclasses remain source-compatible. Set
`supports_temporal_resolution = True` and accept the three keyword arguments
only when that memory has a meaningful event timestamp.

