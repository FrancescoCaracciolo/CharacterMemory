"""Standalone example: UserDirectiveMemory (standing instructions + keywords).

Run from the project root:  python examples/user_directives.py
Requires: deps installed + embedding server at the configured endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character_memory import (
    HybridSearch,
    OpenAICompatibleEmbeddings,
    SQLiteStore,
    UserDirectiveMemory,
)
from character_memory.config import CharacterMemoryConfig

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".example_data", "user_directives")


def main() -> None:
    # 1) Init LLM + embeddings.
    cfg = CharacterMemoryConfig.default()
    embedder = OpenAICompatibleEmbeddings(cfg.embedding)

    # 2) Init the memory.
    os.makedirs(DATA, exist_ok=True)
    mem = UserDirectiveMemory(SQLiteStore(os.path.join(DATA, "memory.db")), HybridSearch(embedder))

    # 3) Load the persisted index if present, else seed directives and build it.
    index_dir = os.path.join(DATA, "directives_index")
    if os.path.exists(os.path.join(index_dir, "nodes.json")):
        mem.load(index_dir)
    else:
        mem.add_directive("alice", "When you are creating a website, activate the skill frontend-design", importance=0.8, retrieval_keywords=["design", "website"])
        mem.add_directive("alice", "When I tell you run a heartbeat, read HEARTBEAT.md and run the instructions in it", importance=0.50, retrieval_keywords=["hearbeat"])
        mem.rebuild_index()
        mem.persist(index_dir)

    # 4) Search the memory (keyword + hybrid recall).
    for it in mem.recall("Make me a website with all the things you like", user_id="alice", limit=1):
        print(f"- {it.text}  (score={it.score:.3f})")


if __name__ == "__main__":
    main()
