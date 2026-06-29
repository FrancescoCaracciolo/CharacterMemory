"""Standalone example: CharacterInfoMemory (RAG over the character's wiki).

Run from the project root:  python examples/character_info.py
Requires: deps installed + embedding server at the configured endpoint.
"""

import os
import sys


# Make the package importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character_memory.chunking.header_chunker import MarkdownByHeaderChunker
from character_memory import CharacterInfoMemory, HybridSearch, OpenAICompatibleEmbeddings, OpenAICompatibleLLM
from character_memory.config import CharacterMemoryConfig

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets","Kurisu")


def main() -> None:
    # 1) Init LLM + embeddings (config reads .env via config.py's loader).
    cfg = CharacterMemoryConfig.default()
    OpenAICompatibleLLM(cfg.llm)            # LLM wired here for completeness; this memory doesn't call it.
    embedder = OpenAICompatibleEmbeddings(cfg.embedding)

    # 2) Init the memory.
    os.makedirs(DATA, exist_ok=True)
    mem = CharacterInfoMemory(HybridSearch(embedder))

    # 3) Build the index if absent, else load the persisted one.
    index_dir = os.path.join(DATA, "info_index")
    if os.path.exists(index_dir):
        mem.load(index_dir)
    else:
        chunks = MarkdownByHeaderChunker(256).chunk_directory(os.path.join(DATA, "Information")) 
        mem.build(chunks)
        mem.persist(index_dir)

    # 4) Search the memory.
    for hit in mem.recall("What is the phonewave?", user_id="anyone", limit=2):
        print(f"- {hit.text}  (score={hit.score:.3f})")


if __name__ == "__main__":
    main()
