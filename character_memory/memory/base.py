"""Memory interfaces.

Every memory system subclasses `Memory`. The agent iterates the
*enabled* memories, asks each for a rendered prompt section, and concatenates
them. 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from ..chunking import Chunk

@dataclass
class MemoryItem:
    """A single recalled fact/snippet ready for the prompt."""

    text: str
    score: float = 0.0
    kind: str = ""
    metadata: dict = field(default_factory=dict)


class Memory(ABC):
    """Base class for all memory systems."""

    name: str = "memory"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    def recall(self, query: str, user_id: str, limit: int) -> list[MemoryItem]:
        """Return up to `limit` items relevant to `query` for `user_id`."""

    @abstractmethod
    def build(self, info_chunks:list[Chunk]) -> None:
        """Build the memory from the given `info_chunks`."""

    @abstractmethod 
    def persist(self, path: str) -> None:
        """Persist the memory to `path`."""

    @property
    def title(self) -> str:
        """Header used when this memory's section is rendered."""
        return self.name.replace("_", " ").title()

    def format(self, items: list[MemoryItem]) -> str:
        """Render recalled `items` into a prompt fragment (override me)."""
        return "\n".join(f"- {it.text}" for it in items)

    def build_section(self, query: str, user_id: str, limit: int) -> Optional[str]:
        """Recall (if enabled) and format; `None` when there is nothing to show."""
        items = self.recall(query, user_id, limit) if self.enabled else []
        if not items:
            return None
        return self.format(items)
