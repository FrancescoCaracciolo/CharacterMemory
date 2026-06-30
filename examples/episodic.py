"""Standalone example: EpisodicMemory (events, decayed + weighted by emotion).

Run from the project root:  python examples/episodic.py
Requires: deps installed + embedding server at the configured endpoint.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from character_memory.memory.extract import Extractor, build_extraction

from character_memory import (
    EpisodicMemory,
    HybridSearch,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
    SQLiteStore,
)
from character_memory.config import CharacterMemoryConfig

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".example_data", "episodic")


def main() -> None:
    # 1) Init LLM + embeddings.
    cfg = CharacterMemoryConfig.default()
    llm = OpenAICompatibleLLM(cfg.llm)
    embedder = OpenAICompatibleEmbeddings(cfg.embedding)

    # 2) Init the memory.
    os.makedirs(DATA, exist_ok=True)
    mem = EpisodicMemory(SQLiteStore(os.path.join(DATA, "memory.db")), HybridSearch(embedder))

    # 3) Extract episodic memory from conversation
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

    # 4) Build the extractor
    schema, instructions = build_extraction([mem.extraction_spec()])
    extractor = Extractor(llm)
    extracted = extractor.extract(conversation, schema, instructions)
    print(extracted)
    mem.apply_extraction(extracted["episodes"], "michael")
    # 4) Search the memory.
    for it in mem.recall("Who called Michael?", user_id="michael", limit=1):
        print(f"- {it.text}  (score={it.score:.3f})")


if __name__ == "__main__":
    main()
