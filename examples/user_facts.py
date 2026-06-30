"""Standalone example: UserFactMemory (per-user facts with confidence + decay).

Run from the project root:  python examples/user_facts.py
Requires: deps installed + embedding server at the configured endpoint.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character_memory.memory.extract import Extractor, build_extraction
from character_memory import (
    HybridSearch,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
    SQLiteStore,
    UserFactMemory,
)
from character_memory.config import CharacterMemoryConfig

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".example_data", "user_facts")


def main() -> None:
    # 1) Init LLM + embeddings.
    cfg = CharacterMemoryConfig.default()
    llm = OpenAICompatibleLLM(cfg.llm)
    embedder = OpenAICompatibleEmbeddings(cfg.embedding)

    # 2) Init the memory (SQLite rows + a RAG index over them).
    os.makedirs(DATA, exist_ok=True)
    mem = UserFactMemory(SQLiteStore(os.path.join(DATA, "memory.db")), HybridSearch(embedder))
    
    # 3) Create a mockup conversation
    conversation = [
        {"role": "user", "content": "Hi, my name is michael, nice to meet you."},
        {"role": "assistant", "content": "Ah, a new acquaintance? How... normal. No strange nicknames or eccentric introductions? I'm Makise Kurisu. Pleased to meet you, Michael. Though I must warn you, if this turns into another situation like with that mad scientist Okabe, I'm not responsible for my reactions."},
        {"role": "user", "content": "There is no problem, I'm an an engineer, not a mad scientist"},
        {"role": "assistant", "content": "An engineer? That's actually refreshing. Most people I meet these days are either completely lost when it comes to basic technology, or they're... well, like Okabe. So what field of engineering do you specialize in?"},
        {"role": "user", "content": "A MAD Engineer!"},
        {"role": "assistant", "content": "Oh no... Not another one. I swear, this universe is conspiring to surround me with theatrical lunatics. First Hououin Kyouma, now a \"MAD Engineer\"? (crosses arms) Please tell me you're not going to start striking dramatic poses and shouting about world domination every five minutes."},
        {"role": "user", "content": "Just joking don't worry Christina. I'm a nuclear engineer and I was called by Daru to check up the Phonewave"},
        {"role": "assistant", "content": "First of all, don't call me Christina. That's exclusively that idiot Okabe's habit, and I tolerate it only because I've run out of energy to correct him. Secondly... Daru actually called a nuclear engineer? That's interesting. The PhoneWave does have some concerning radiation readings that I've been meaning to investigate."}
    ]
    
    # 5) Extract Memory 
    # Read the memory extraction spec for that memory
    schema, instruction = build_extraction([mem.extraction_spec()])
    # Extract user facts using an LLM
    extractor = Extractor(llm)
    extracted = extractor.extract(conversation, schema, instruction)
    mem.apply_extraction(extracted["facts"], user_id="michael")


    # 4) Search the memory.
    for it in mem.recall("What is michael's job?", user_id="michael", limit=1):
        print(f"- {it.text}  (score={it.score:.3f})")


if __name__ == "__main__":
    main()
