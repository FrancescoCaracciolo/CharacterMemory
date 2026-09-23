# Configuration

There are three layers of configuration, all optional — every knob has a
sensible default.

1. **Environment variables / `.env`** — the LLM/embedding endpoints and API
   keys. Read at import time by `config.py`'s tiny `.env` loader.
2. **`config.yaml`** (per character) — the single source of truth for a
   character: persona, prompts, every sub-config, aliases. Read by
   `CharacterAgent.load_from_config(path)`.
3. **In Python** — construct the dataclasses yourself and pass them in.

Only the fields listed under environment variables have environment mappings.
Use YAML or Python for other settings, including the global memory budget.
Explicit configuration overrides the corresponding environment defaults.

---

## Environment variables

For guided server configuration, run `charactermemory-server-setup` from the
directory where you will start `charactermemory-server`. The wizard saves `.env`,
preserves unrelated settings, and offers optional connection tests. See the
[setup walkthrough](../character_memory/server/README.md#guided-setup).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Chat-completions endpoint. |
| `OPENAI_API_KEY` | _empty_ | API key for the above. |
| `OPENAI_MODEL` | `gpt-5.4-mini` | Chat model used for answers + extraction. |
| `OPENAI_EMBEDDINGS_BASE_URL` | `https://api.openai.com/v1` | Embeddings endpoint. |
| `OPENAI_EMBEDDINGS_MODEL` | `text-embedding-ada-002` | Embeddings model. |
| `OPENAI_EMBEDDINGS_API_KEY` | `OPENAI_API_KEY` | Separate embeddings key; explicitly empty means no key rather than fallback. |
| `CM_STORAGE_BACKEND` | `sqlite` | Structured storage: `sqlite` or `postgres`. |
| `CM_DATABASE_URL` | _empty_ | PostgreSQL connection URL; requires the `postgres` extra. |
| `CM_DATABASE_NAMESPACE` | automatic per character | PostgreSQL isolation namespace; normally leave unset on the server. |
| `CM_RETRIEVAL_BACKEND` | `hybrid` | Local FAISS/BM25 or `postgres` (requires PostgreSQL storage and pgvector >= 0.8). |
| `CM_HOST` | `0.0.0.0` | Server console command bind host; `--host` takes precedence. |
| `CM_PORT` | `8000` | Server console command port; `--port` takes precedence. |
| `CM_TEMPORAL_RESOLUTION_ENABLED` | `true` | Master switch for automatic temporal recall. |
| `CM_TEMPORAL_RESOLUTION_ENGINE` | `dateparser` | `dateparser`, `llm`, or a registered engine. |
| `CM_TEMPORAL_RESOLUTION_TIMEZONE` | `UTC` | IANA timezone for calendar boundaries. |
| `CM_TEMPORAL_RESOLUTION_WEIGHT` | `1.0` | Temporal ranking contribution, clamped to `0..1`. |
| `CM_TEMPORAL_RESOLUTION_LANGUAGES` | _empty_ | Comma-separated ISO codes; empty uses broad fast defaults. |
| `CM_TEMPORAL_LLM_BASE_URL` | `OPENAI_BASE_URL` | Dedicated opt-in temporal LLM endpoint. |
| `CM_TEMPORAL_LLM_API_KEY` | `OPENAI_API_KEY` | Dedicated temporal endpoint key. |
| `CM_TEMPORAL_LLM_MODEL` | `OPENAI_MODEL` | Temporal model used only with `engine=llm`. |
| `CM_TEMPORAL_LLM_TIMEOUT` | `30` | Temporal request timeout in seconds. |
| `CM_TEMPORAL_LLM_MAX_TOKENS` | `768` | Maximum output tokens for the one-call JSON response. |
| `CM_ASSETS_DIR` | `./assets` | Root scanned for character subdirectories (server). |
| `CM_SAVE_DIR` | `./.cm_servers` | Where each character's SQLite store + indexes live (server). |
| `CM_KG_CHARACTERS` | _empty_ | Comma-separated character names to enable the KG for. |
| `CM_REBUILD_KG` | _empty_ | Comma-separated names to rebuild at startup, or `all`. |
| `CM_SYNC_INTERVAL` | `3.0` | Background cache-sync poll interval (seconds); `0` disables. |
| `CM_API_KEY` | _empty_ | Require this API key on every server endpoint (comma-separated list allowed); empty = no auth. |

A `.env` in the current working directory is auto-loaded, falling back to the
package's parent directory when absent. Values already in the real environment
win. The loader treats values literally, without variable interpolation or
shell execution. Single quotes preserve their contents; double quotes support
escaped backslashes and double quotes. Multiline values are unsupported.
`CM_HOST` and `CM_PORT` apply to the server console command, not direct uvicorn
launches.

---

## The dataclasses

All in `character_memory/config.py`, re-exported from the top level.

### `LLMConfig` — chat-completions client

| Field | Default | Notes |
|---|---|---|
| `base_url` | `OPENAI_BASE_URL` | Any OpenAI-compatible `/v1`. |
| `api_key` | `OPENAI_API_KEY` | |
| `model` | `OPENAI_MODEL` | |
| `temperature` | `0.7` | |
| `max_tokens` | `1024` | |
| `timeout` | `120.0` | seconds |

### `EmbeddingConfig` — embeddings client

| Field | Default | Notes |
|---|---|---|
| `base_url` | `OPENAI_EMBEDDINGS_BASE_URL` | |
| `api_key` | `OPENAI_EMBEDDINGS_API_KEY`, falling back to `OPENAI_API_KEY` | Empty override stays empty. |
| `model` | `OPENAI_EMBEDDINGS_MODEL` | |
| `dim` | `None` | inferred from the first request if `None` |
| `batch_size` | `64` | |
| `timeout` | `120.0` | |
| `retrieval_query_prefix` | `""` | prepended only when embedding retrieval queries |
| `retrieval_document_prefix` | `""` | prepended only when embedding indexed documents |
| `retrieval_min_similarity` | `None` | optional cosine floor for dense candidates |

Role prefixes are configuration, not model-name checks. Asymmetric retrieval
models can therefore use values such as `Query: ` / `Document: ` while
symmetric providers retain the empty defaults.

### `TemporalResolutionConfig` — time-aware recall

| Field | Default | Notes |
|---|---|---|
| `enabled` | `True` | master switch |
| `engine` | `"dateparser"` | local fast engine; `"llm"` is opt-in |
| `timezone` | `"UTC"` | IANA timezone used to turn dates into UTC ranges |
| `weight` | `1.0` | temporal relevance contribution (`0..1`) |
| `languages` | `[]` | broad fast defaults; `["*"]` loads every dateparser locale |
| `llm` | `TemporalLLMConfig()` | isolated endpoint/model/timeout/max-token settings |

The default engine uses no model and no embeddings. The LLM engine performs
exactly one chat call per recall turn. See [Automatic temporal
resolution](temporal_resolution.md) for the timestamp policy, scoring, direct
memory API, and custom-engine seam.

### `ChunkingConfig` — index-time chunking

| Field | Default | Notes |
|---|---|---|
| `info_chunker` | `"header"` | chunker name for `Information/*.md` |
| `dialogue_chunker` | `"dialogue"` | chunker name for `Dialogues/*.txt` |
| `header_max_tokens` | `512` | |
| `header_min_tokens` | `64` | fragments below this are merged |
| `dialogue_turns_per_chunk` | `6` | |
| `dialogue_context_width` | `3` | preceding turns carried as context |

### `MemoryConfig` — per-memory toggles + retrieval knobs

| Field | Default | Notes |
|---|---|---|
| `enabled_character_info` | `True` | and one `enabled_*` per memory |
| `enabled_knowledge_graph` | `False` | KG is opt-in |
| `enabled_calendar` | `False` | Dated events and weekly routines |
| `enabled_world` | `False` | Exact private-world state, routines, needs, and facts |
| `character_info_k` | `4` | and one `*_k` per memory (retrieval size) |
| `knowledge_graph_token_budget` | `1000` | graph-body token limit before global selection |
| `token_budget` | `None` | shared cap for rendered memory sections; unlimited by default |
| `sticky_threshold` | `0.95` | eligibility for a sticky retrieval slot; global selection may still exclude the item |
| `extract_interval` | `5` | run extraction every N user turns |
| `decay_half_life` | `259200` (3 days) | seconds; used by structured memories |
| `emotion_baseline` | `{neutral, joy, sadness, anxiety, anger, surprise}` | character resting state |
| `emotion_user_dims` | `{affection, valence, trust}` | per-user relationship dims |
| `retrieval_history_window` | `5` | how many recent user msgs drive retrieval |
| `retrieval_recency_decay` | `0.6` | per-step weight multiplier for older msgs |
| `dedup` | `DedupConfig()` | dedup behaviour (see below) |
| `knowledge_graph` | `KnowledgeGraphConfig()` | KG tunables (off by default) |
| `world` | `WorldConfig()` | seed file, auto-advance, extraction, model actions |
| `calendar` | `CalendarConfig()` | timezone, nearby recall window, world-routine import, extraction |

Helpers: `MemoryConfig.is_enabled(name)` and `MemoryConfig.k_for(name)`.

Set `memory.token_budget: 3000` in YAML or use
`MemoryConfig(token_budget=3000)` in Python. Per-call `budget` overrides apply
to recall, context building, and generation; omitted means inherit, `None`
means unlimited, and `0` omits memories. HTTP `/context` uses the same field,
with JSON `null` for unlimited. Existing per-memory limits still determine
the candidates. Rerankers and token counters are Python constructor/callable
injections, not YAML fields. See [Memory budget & reranking](memory_budget.md).

### `WorldConfig` and `world.yaml`

World time is resolved from the actor's current location, walking through its
parent locations before falling back to the actor timezone, world timezone,
and finally UTC. Routine occurrences keep absolute start/end timestamps once
started, even when the routine moves the actor into another timezone.

World facts may be `durable` (no expiry) or `temporary` (`valid_until` is
required). `WorldMemory.add_fact(..., valid_until=...)`, the world editor, and
the `upsert_world_fact` MCP tool use the same model. Expired temporary facts
remain in the ledger as history but are excluded from current-state prompts,
search, and knowledge-graph projection.

### `DedupConfig`

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | master switch |
| `exact` | `True` | case-insensitive string gate |
| `similarity_threshold` | `0.92` | `None` disables the similarity gate |
| `llm_judge` | `False` | LLM confirms similarity candidates |
| `consolidate` | `False` | merge duplicates into one entry instead of dropping |
| `per_user` | `True` | only compare entries sharing `user_id` |
| `candidate_pool` | `10` | per-item guard: how many RAG hits to re-rank |

### `ContradictionPolicy` (per-memory)

Returned by `StructuredMemory.contradiction_policy()`. `enabled` (default
`False`), `similarity_threshold` (`0.70`), `candidate_pool` (`20`),
`show_timestamps` (`True`). See [Extraction & dedup](extraction_and_dedup.md).

### `KnowledgeGraphConfig`

| Field | Default | Notes |
|---|---|---|
| `decay` | `0.5` | ACT-R BLL decay parameter (d) |
| `decay_half_life` | `604800` (1 week) | extra exponential recency factor on top of BLL |
| `gain` | `0.35` | spreading-activation gain |
| `hops` | `2` | pinned by design |
| `hop_decay` | `0.6` | per-hop attenuation |
| `base_weight` | `1.0` | BLL term weight in combined score |
| `spread_weight` | `1.2` | spreading term weight |
| `min_activation` | `0.0` | activation floor |
| `hebbian_threshold` | `0.15` | co-activation reinforcement threshold |
| `hebbian_lr` | `0.05` | co-recall strengthening amount |
| `self_seed` | `0.8` | SelfNode seed activation |
| `match_base` / `match_gain` | `4.0` / `3.0` | query-match activation seeding |
| `fact_batch_size` | `50` | maximum stored facts per KG structured-extraction request |
| `episode_batch_size` | `50` | maximum stored episodes per deterministic KG ingest batch (episodes do not use an LLM) |
| `wiki_batch_size` | `3` | maximum wiki chunks per KG structured-extraction request |
| `extraction_token_limit` | `10000` | maximum aggregate source-text tokens per batch; prompt/schema/output overhead is separate |
| `project_heartbeat` | `True` | derive KG facts/events from admitted heartbeat discoveries/actions |
| `heartbeat_min_importance` | `0.6` | minimum heartbeat importance admitted to the graph |
| `heartbeat_max_nodes` | `200` | maximum heartbeat rows projected, highest importance/newest first |
| `project_world` | `True` | derive KG identities, locations, durable facts, and events from `WorldMemory` |
| `world_event_max_nodes` | `500` | maximum visible durable world events projected |
| `world_include_simulation_events` | `False` | include simulator-generated event records; off to avoid routine/state noise |
| `world_location_seed` | `0.6` | transient activation seed for the observer's current location; not a stored fact |

World routines and mutable actor state (current activity, hunger, energy,
sleep, and current location) are deliberately not persisted into the graph.
Routines remain available through `WorldMemory` and, when enabled, the
calendar's live world-routine projection.

### `PromptConfig`

Every string the agent injects into the LLM prompt. Individually overridable;
see `prompts.py` for the full list. Highlights:

| Field | Purpose |
|---|---|
| `system` | Top-level system prompt template (`{character_name}`, `{base_instruction}`). |
| `emotion_note` | One-liner interpolating `{baseline}` / `{user_state}`. |
| `section_template` | Per-section template (default `"## {title}\n{body}"`). |
| `<name>_header` / `<name>_header_multi` | Section headers, singular + plural (group chat). |
| `section_order` | Order memory sections appear in the prompt. |
| `extraction_*` | Extraction LLM prompts (header, sentence rule, known-facts intro, footer, multi-note). |
| `dedup_judge` / `dedup_consolidate` / `dedup_contradict` | Deduplication LLM prompts. |

---

## The `config.yaml` format

A character's `config.yaml` is the **single source of truth** for everything
except global LLM credentials. It is written next to the character dir:

```
assets/Kurisu/
├── config.yaml       ← this file
├── Information/…
└── Dialogues/…
```

Every section is optional — delete a key to fall back to its library default.
Unknown keys are ignored, so the format is forward-compatible.

```yaml
name: Kurisu
aliases: ["Christina", "Makise Kurisu"]    # KG self-dedup; persona is scanned too
persona: |
  A neuroscience researcher. Goes by "Christina" (don't call her that).

llm:        { base_url, api_key, model, temperature, max_tokens, timeout }
embedding:  { base_url, api_key, model, dim, batch_size, timeout }
chunking:   { info_chunker, dialogue_chunker, header_max_tokens,
              header_min_tokens, dialogue_turns_per_chunk,
              dialogue_context_width }

temporal_resolution:
  enabled: true
  engine: dateparser
  timezone: Europe/Rome
  weight: 1.0
  languages: [it, en]
  llm: { base_url, api_key, model, timeout, max_tokens }

memory:
  enabled_character_info: true          # and every other enabled_* toggle
  character_info_k: 4                   # and every other *_k size
  token_budget: null                   # shared memory cap; e.g. 3000
  knowledge_graph_token_budget: 1000    # graph-only candidate body cap
  sticky_threshold: 0.95
  extract_interval: 5
  decay_half_life: 259200
  emotion_baseline:  { joy: 0.2, sadness: 0.1, … }
  emotion_user_dims: { affection: 0.0, valence: 0.0, trust: 0.0 }
  enabled_knowledge_graph: false
  retrieval_history_window: 5
  retrieval_recency_decay: 0.6
  dedup:          { enabled, exact, similarity_threshold, llm_judge,
                    consolidate, per_user, candidate_pool }
  knowledge_graph: { decay, decay_half_life, gain, hops, hop_decay,
                     base_weight, spread_weight, min_activation,
                     hebbian_threshold, hebbian_lr, self_seed,
                     match_base, match_gain, fact_batch_size,
                     episode_batch_size, wiki_batch_size,
                     extraction_token_limit }

prompts:
  system, emotion_note, section_template, *_header, *_header_multi,
  extraction_*, dedup_*, section_order
```

Load it with:

```python
agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config("assets/Kurisu/config.yaml")
agent.build()
```

The agent also reads a legacy `character.json` manifest (the format the
WebUI used to write) for one-way back-compat migration — persona, memory
toggles, retrieval sizes, section order and system prompt survive.

---

## Creating a character

The recommended workflow (from the project memory):

1. **Choose name and description**: create `assets/<Name>/` and write a
   `config.yaml` with `name:` and `persona:`. A quick way to get a fully
   populated starting file:

   ```python
   from character_memory.character_config import default_config_yaml
   print(default_config_yaml(name="Kurisu", persona="…", aliases=["Christina"]))
   ```

   Save the output as `assets/Kurisu/config.yaml`.

2. **Add wiki files** (optional) : drop Markdown into
   `assets/Kurisu/Information/`. These feed `character_info`.

3. **Add conversation files** (optional): drop example exchanges as `.txt`
   into `assets/Kurisu/Dialogues/`. These feed `dialogue_style`.

4. **Pick which memories to enable**: edit `memory.enabled_*` in the
   `config.yaml`. Drop the knowledge-graph marker file
   (`touch assets/Kurisu/.knowledge_graph`) or set
   `enabled_knowledge_graph: true` to opt into the KG.

5. **Build the indexes**: `agent.build()` (load-or-build, idempotent) or
   `charactermemory-server` on first start. Re-run `agent.rebuild()` (or
   `--rebuild-kg`) after editing source files.

---

## Programmatic configuration

```python
from character_memory import (
    CharacterAgent, CharacterMemoryConfig, LLMConfig, EmbeddingConfig,
    MemoryConfig, DedupConfig, ChunkingConfig, TemporalResolutionConfig,
)

cfg = CharacterMemoryConfig(
    llm=LLMConfig(model="my-model"),
    embedding=EmbeddingConfig(model="my-embeddings"),
    chunking=ChunkingConfig(header_max_tokens=256),
    temporal_resolution=TemporalResolutionConfig(
        engine="dateparser", timezone="Europe/Rome", languages=["it", "en"]
    ),
    memory=MemoryConfig(
        enabled_knowledge_graph=True,
        extract_interval=3,
        decay_half_life=7 * 24 * 3600,
        dedup=DedupConfig(enabled=True, consolidate=True),
    ),
)
agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config(cfg)
agent.build()
```

Constructor-level overrides always win:

- `CharacterAgent(prompt_config=PromptConfig())`: your prompts win over
  `config.yaml`'s.
- `CharacterAgent(persona="…")`: your persona wins.
- `load_from_config(…, llm=…, embedder=…)`: custom backends win over the
  auto-built OpenAI clients.

## Saving a character's config back to disk

```python
from character_memory.character_config import save_config

save_config(
    "assets/Kurisu",
    config=agent.config,
    prompts=agent.prompts,
    persona=agent.persona,
    name="Kurisu",
    aliases=["Christina"],
)
```

This is what the WebUI's Workshop tab does when you click "Save".
