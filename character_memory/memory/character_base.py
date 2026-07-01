"""Memories backed purely by a RAG index.

Both wrap a `HybridSearch` class and simply return the top-k chunks for a
query; they differ only in how the hits are formatted into the prompt.
This is used for a inner memory type, not relevant to the user. 
For example wikis, conversations and base character information.
"""

from typing import Optional

from ..rag.hybrid import HybridSearch
from .base import Memory, MemoryItem
from ..chunking import Chunk

class RAGMemory(Memory):
    """Memory whose recall is just a hybrid search over pre-built chunks."""

    def __init__(self, hybrid: HybridSearch, *, enabled: bool = True, _title: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(enabled=enabled, name=name)
        self.hybrid = hybrid
        self._title = _title

    @property
    def title(self) -> str:
        return self._title or super().title

    def recall(self, query: str, user_id: str, limit: int) -> list[MemoryItem]:
        hits = self.hybrid.search(query, k=limit)
        return [
            MemoryItem(text=h.text, score=h.score, kind=self.name, metadata=h.metadata)
            for h in hits
        ]

    def format(self, items: list[MemoryItem]) -> str:
        return "\n\n".join(it.text for it in items)

    def build(self, info_chunks: list[Chunk]) -> None:
        return self.hybrid.build(info_chunks)

    def persist(self, path: str) -> None:
        return self.hybrid.persist(path)

    def load(self, path: str) -> None:
        return self.hybrid.load(path)

class CharacterInfoMemory(RAGMemory):
    """BM25 + similarity search over the character's wiki/story markdown."""

    name = "character_info"

    def __init__(self, hybrid: HybridSearch, *, enabled: bool = True, name: Optional[str] = None) -> None:
        super().__init__(hybrid, enabled=enabled, _title="Character Information", name=name)


class DialogueStyleMemory(RAGMemory):
    """Few-shot dialogue retrieval: the k most similar past exchanges."""

    name = "dialogue_style"

    def __init__(self, hybrid: HybridSearch, *, enabled: bool = True, name: Optional[str] = None) -> None:
        super().__init__(hybrid, enabled=enabled, _title="Example Exchanges (style reference)", name=name)

    def format(self, items: list[MemoryItem]) -> str:
        out = []
        for i, it in enumerate(items, 1):
            out.append(f"[Example {i}]\n{it.text}")
        return "\n\n".join(out)
