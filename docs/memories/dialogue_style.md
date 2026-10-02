# `dialogue_style` — few-shot example exchanges

| | |
|---|---|
| **Class** | `DialogueStyleMemory` (`memory/character_base.py`) |
| **Flavor** | RAG memory (pure `HybridSearch` index) |
| **Scope** | `CHARACTER` |
| **Learns from chat?** | No — rebuilt from `<character>/Dialogues/*.txt` |
| **Config toggle** | `MemoryConfig.enabled_dialogue_style` |
| **Retrieval size** | `MemoryConfig.dialogue_style_k` (default `4`) |

## What it does

Few-shot **style reference** retrieval. Drop example exchanges as `.txt`
files into `<character>/Dialogues/` and the memory surfaces the `k` most
similar past exchanges to the current user line, formatted as numbered
"Example 1 / Example 2 / …" blocks.

Where `character_info` grounds the character in *facts*, `dialogue_style`
grounds the model in *voice* — tone, cadence, verbal tics, in-character
phrasing.

## How it's built

1. Chunk every `<character>/Dialogues/*.txt` with the **dialogue chunker**
   (`ChunkingConfig.dialogue_chunker`, defaults to `"dialogue"`,
   `dialogue_turns_per_chunk=6`, `dialogue_context_width=3`).
2. Build a `HybridSearch` index and persist it to
   `<save_directory>/dialogue_index/`.

The dialogue chunker groups consecutive turns (with a little surrounding
context) into one chunk, so each indexed unit is a coherent exchange rather
than a single line.

## Recall & formatting

`recall` returns the top-`k` exchange chunks. The `format` override renders
them as:

```
[Example 1]
<exchange text>

[Example 2]
<exchange text>
```

## Section header

`"Example Exchanges (style reference)"` (overridable via
`PromptConfig.dialogue_style_header`).

## Standalone usage

```python
from character_memory import DialogueStyleMemory, HybridSearch, OpenAICompatibleEmbeddings
from character_memory.chunking.dialogue_chunker import DialogueChunker

mem = DialogueStyleMemory(HybridSearch(OpenAICompatibleEmbeddings()))
chunks = DialogueChunker().chunk_directory("assets/Kurisu/Dialogues")
mem.build(chunks)

for hit in mem.recall("What is the phonewave?", user_id="anyone", limit=3):
    print("---", hit.score); print(hit.text)
```

## Editing

Read-only in the WebUI (the files on disk are the source of truth). The MCP
`add_dialogue` tool appends a chunk to the in-RAM index for the session but
is overwritten on the next `agent.rebuild()` — persistent edits belong on
disk.
