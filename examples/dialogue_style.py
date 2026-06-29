"""Standalone example: DialogueStyleMemory (few-shot past-exchange retrieval).

Run from the project root:  python examples/dialogue_style.py
Requires: deps installed + embedding server at the configured endpoint.
"""

import os
import sys

# Make the package importable when running this file directly (must come
# before any `from character_memory...` import).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character_memory import DialogueStyleMemory, HybridSearch, OpenAICompatibleEmbeddings, OpenAICompatibleLLM
from character_memory.chunking.dialogue_chunker import DialogueChunker
from character_memory.config import CharacterMemoryConfig

# Character assets live at the project root, not under `examples/`.
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "Kurisu")


def main() -> None:
    # 1) Init LLM + embeddings.
    cfg = CharacterMemoryConfig.default()
    OpenAICompatibleLLM(cfg.llm)
    embedder = OpenAICompatibleEmbeddings(cfg.embedding)

    # 2) Init the memory.
    if not os.path.isdir(os.path.join(DATA, "Dialogues")):
        raise SystemExit(f"missing character dialogues at {DATA}/Dialogues — can't build the dialogue index.")
    mem = DialogueStyleMemory(HybridSearch(embedder))

    # 3) Build the index if absent, else load the persisted one.
    #    Note: do NOT start a path component with "/"; `os.path.join` treats
    #    absolute components as the new root and drops everything before.
    index_dir = os.path.join(DATA, ".cm_data", "dialogue_index")
    if os.path.exists(os.path.join(index_dir, "nodes.json")):
        mem.load(index_dir)
    else:
        chunks = DialogueChunker().chunk_directory(directory=os.path.join(DATA, "Dialogues"))
        mem.build(chunks)
        mem.persist(index_dir)

    # 4) Search for the exchange most similar to a new user line.
    for hit in mem.recall("What is the phonewave?", user_id="anyone", limit=3):
        print("---", hit.score)
        print(hit.text)


if __name__ == "__main__":
    main()
