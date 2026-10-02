# `character_info` — wiki / lore retrieval

| | |
|---|---|
| **Class** | `CharacterInfoMemory` (`memory/character_base.py`) |
| **Flavor** | RAG memory (pure `HybridSearch` index) |
| **Scope** | `CHARACTER` (recalled once, not per participant) |
| **Learns from chat?** | No — rebuilt from `<character>/Information/*.md` |
| **Config toggle** | `MemoryConfig.enabled_character_info` |
| **Retrieval size** | `MemoryConfig.character_info_k` (default `4`) |

## What it does

Hybrid (BM25 lexical + FAISS dense similarity, fused with reciprocal rank
fusion) search over the character's **wiki and lore**. Drop your background
material as Markdown files into `<character>/Information/` and the memory
chunks them (by header, by default) and surfaces the passages most relevant
to the current query.

This is the character's "what do I know about the world and myself" memory.

## How it's built

At `agent.build()` time the agent:

1. Chunk every `<character>/Information/*.md` file with the **header chunker**
   (`ChunkingConfig.info_chunker`, defaults to `"header"`,
   `header_max_tokens=512`, `header_min_tokens=64`).
2. Build a `HybridSearch` index over those chunks and persist it to
   `<save_directory>/info_index/`.

On subsequent builds the persisted index is loaded; call `agent.rebuild()`
to force a re-chunk + re-index (e.g. after editing the wiki).

## Recall

`recall(query, user_id, limit)` returns the top `limit` chunks for the query.
`user_id` is accepted for interface symmetry but **ignored** — this memory is
character-scoped, not per-user.

The default `format` joins chunks with a blank line between them.

## Section header

`"Character Information"` (overridable via `PromptConfig.character_info_header`).

## Standalone usage

```python
from character_memory import CharacterInfoMemory, HybridSearch, OpenAICompatibleEmbeddings
from character_memory.chunking.header_chunker import MarkdownByHeaderChunker

mem = CharacterInfoMemory(HybridSearch(OpenAICompatibleEmbeddings()))
chunks = MarkdownByHeaderChunker(256).chunk_directory("assets/Kurisu/Information")
mem.build(chunks)

for hit in mem.recall("What is the phonewave?", user_id="anyone", limit=2):
    print(f"- {hit.text}  (score={hit.score:.3f})")
```

## Editing

The WebUI renders wiki chunks as Markdown but they are **read-only**: the
files on disk are the source of truth. Edit `Information/*.md` and call
`agent.rebuild()` (or `charactermemory-server` will pick up the changes on
restart). The MCP `add_character_info` tool appends a chunk to the in-RAM
index for the session, but it is overwritten on the next rebuild.
