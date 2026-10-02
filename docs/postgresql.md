# PostgreSQL and pgvector

CharacterMemory supports PostgreSQL for conversations and structured memories,
and pgvector plus PostgreSQL full-text search for hybrid retrieval. SQLite and
local BM25/FAISS remain the defaults. The production configuration targets one
application process on one machine; independent application workers sharing the
same character's in-memory graph/world state are not supported by this backend.

## Run locally

```bash
pip install 'charactermemory[postgres,server]'
export CM_POSTGRES_PASSWORD='choose-a-local-password'
docker compose -f examples/postgres/compose.yaml up -d
export CM_DATABASE_URL='postgresql://character_memory:choose-a-local-password@127.0.0.1:5432/character_memory'
export CM_STORAGE_BACKEND=postgres
export CM_RETRIEVAL_BACKEND=postgres
charactermemory-server
```

URL-encode special characters in credentials. The Compose example binds the
database to loopback and persists it in a named volume. Existing PostgreSQL
installations need pgvector >=0.8 and `CREATE EXTENSION vector` run by an
administrator. The library does not install extensions or provision the database.
The application role needs schema creation rights and ownership of its schemas.
PostgreSQL 17 is the example deployment; integration validation also ran on 18.

Merge [the example settings](../examples/postgres/config.yaml) into each character's
`config.yaml`. Explicit YAML/Python values take precedence over environment defaults.
Use a distinct, stable `storage.namespace` for each character. By default the
namespace derives from the absolute save directory; moving that directory changes
the default namespace. Schemas are named `cm_` plus the first 32 hex digits of the
namespace's SHA-256 hash. A global `CM_DATABASE_NAMESPACE` applies to every character,
so do not set it for a server hosting multiple characters unless each YAML overrides it.

Switching backends starts with fresh storage. There is no SQLite data migration.
Character source files, authored world configuration, and configuration YAML remain
on disk. A PostgreSQL retrieval backend requires `PostgresStore`; PostgreSQL storage
can alternatively use the existing local retrieval backend.

## Python and extension interfaces

```python
from character_memory import (
    CharacterAgent, CharacterMemoryConfig, StorageConfig, RetrievalConfig,
)

config = CharacterMemoryConfig(
    storage=StorageConfig(backend="postgres", namespace="kurisu-production"),
    retrieval=RetrievalConfig(backend="postgres"),
)
agent = CharacterAgent("assets/Kurisu").load_from_config(config).build()
try:
    chat = agent.create_chat("alice")
    chat.add_message("user", "Hello!", user_id="alice")
    answer = agent.generate_answer(chat)
finally:
    agent.close()
```

`StorageConfig.url` defaults to `CM_DATABASE_URL` and is omitted from its repr.
Keep URLs in environment variables when saving or sharing configuration files:
`save_config` omits empty URLs and URLs matching `CM_DATABASE_URL`; other explicitly configured URLs are serialized along with the settings.
The admin configuration response does not expose connection URLs.

`Store` is the storage ABC; `SQLiteStore` and `PostgresStore` implement it. Its
transaction escape hatch uses qmark SQL, portable result rows, `rowcount`, and
`lastrowid` for inserts into tables declared through `create_table`. `upsert` returns
the actual primary key, including string keys. SQL dialect adaptation lives in the
store, not in memory subclasses. Schema declarations retain the existing library
vocabulary, including `INTEGER PRIMARY KEY AUTOINCREMENT` and `REAL` (mapped to
PostgreSQL identity and double precision).

Callers may inject `CharacterAgent(..., store=my_store)`; injected stores override
storage configuration and remain caller-owned. Close them after closing their
agents. PostgreSQL pools are shared by connection URL and pool settings in one
process, with reference-counted closure. Defaults are 2–20 connections and a
30-second acquisition timeout; database connections are released before embedding
or LLM requests. Pool statistics are available through `store.pool.get_stats()`.

`PostgresHybridSearch(embedder, store, collection, ...)` implements `RAGSystem`,
including updates, deletion, counts, document enumeration, and weighted queries.
`exists(path)` and `refresh()` are backward-compatible lifecycle hooks; remote
indexes do not depend on `nodes.json` or FAISS files. Existing custom `RAGSystem`
subclasses inherit the local-file probe and a no-op refresh.

## Retrieval, recovery, and concurrency

The PostgreSQL backend fuses cosine and full-text candidates with the same weighted
RRF/confidence scoring as local retrieval. It preserves user filters, allowed IDs,
multiple search keys per application row, result grouping, and similarity thresholds.
Full-text ranking is **not BM25**: lexical order may differ. Query terms are OR-combined so full-sentence questions can match individual keywords. The default `simple`
text-search configuration avoids language-specific stemming; configure a different
PostgreSQL text-search configuration when appropriate. Hits expose `lexical_score`
and `lexical_backend`; `bm25_score` is retained as the compatibility field for the
lexical score.

GIN indexes support lexical/metadata filtering. HNSW supports cosine vectors up to
2,000 dimensions; higher dimensions use exact search. `hnsw: false` disables HNSW.
`ef_search` controls approximate search breadth. Selective queries that exhaust
HNSW candidates are completed with exact filtered search. Use
`search(..., exact=True)` to compare rankings against an exact baseline.

Collection text and embedding identity/dimension are durable. Rebuilds create a
separate table and publish it atomically after successful embedding/index creation.
A failed rebuild leaves the previous generation usable. Fingerprint or dimension
changes rebuild from raw text. Abrupt process termination during staging can leave
an unreferenced `cm_vectors_*` table; it is safe to remove after checking that no
`cm_collections.active_table` refers to it and no rebuild is running.

Structured-memory and graph source changes enqueue index work in the same SQL
transaction via triggers. Incremental repair atomically replaces affected search
keys and acknowledges only the processed revision. Failed embedding calls leave
source rows and pending work durable; later reads, persistence, refresh, or restart
retry the work. Row changes and embeddings are eventually consistent; a failed
repair is reported instead of silently returning an out-of-date structured index.
Do not retry an entire failed user turn blindly: a reply or source row may already
have committed before an indexing failure. Inspect persisted messages first.

Same-chat generation/extraction is serialized, including streaming consumption.
Always exhaust or close a stream. Different chats can generate concurrently. Graph,
world, and emotion mutations use in-process locks. Application code should submit
one turn at a time per chat (including the user-message write), and quiesce active
requests before reloading configuration, rebuilding characters, or closing agents.
The explicit admin character-delete endpoint also drops that character’s PostgreSQL schema. Deleting only local files does not remove remote data. Database backups must include all character schemas and vector tables; a copy of
`.cm_data` alone does not back up PostgreSQL state.

## Validation and measured workload

Run the real-backend contracts against a disposable database:

```bash
export CM_TEST_DATABASE_URL='postgresql://.../test_database'
CM_ASSETS_DIR=/tmp/empty-character-assets OPENAI_API_KEY=offline-test \
  python -m pytest -q tests/test_postgres_backends.py
python -m benchmarks.postgres_concurrency --output postgres-report.json
```

The benchmark creates unique schemas and removes those schemas afterward. It uses
100,000 facts, five characters, 50 simultaneous conversations, three turns per
conversation, and 512-dimensional deterministic embeddings. Each turn persists a
user/assistant pair, extracts a new fact, and checks filtered retrieval. It verifies
row counts, extraction completion, user isolation, and an empty repair queue.

The checked-in [sample report](benchmarks/postgres-100k.json) measured 14.9 turns/s,
3.24s median and 4.61s p95 per turn, and 100% filtered hybrid top-10 overlap against
exact search on 50 sample queries. Application peak RSS was 573 MiB including
indexing. PostgreSQL selected exact filtered plans for the five sampled automatic
plans. A separate forced-HNSW test confirmed index use on 50 queries and measured
100% dense recall@10 against exact search. The synthetic embeddings sum seeded,
dense Gaussian token vectors to avoid the many artificial distance ties produced
by sparse token hashing. These figures depend on hardware, corpus distribution, and memory toggles;
the benchmark enables fact extraction/retrieval and tests other memory systems
separately. Real LLM latency and embedding-server capacity are additional costs.


The integration run completed on PostgreSQL 18.6 with pgvector 0.8.6. The live
embedding smoke test reached the configured authenticated local server at
`127.0.0.1:9999`, but returned HTTP 500 because its Ollama upstream was unavailable.
The database and retrieval tests therefore use deterministic providers.
