"""Memory interfaces.

Every memory system subclasses `Memory`. The agent iterates the
*enabled* memories, asks each for a rendered prompt section, and concatenates
them. 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from ..chunking import Chunk

@dataclass
class MemoryItem:
    """A single recalled fact/snippet ready for the prompt."""

    text: str
    score: float = 0.0
    kind: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionSpec:
    """How a memory is populated by LLM extraction.

    A memory that wants to learn from the conversation returns one of these
    from :meth:`Memory.extraction_spec`. The agent gathers the specs of every
    *enabled* memory, builds a single combined JSON schema + instruction out of
    them, runs extraction, then hands each memory the value it asked for via
    :meth:`Memory.apply_extraction`.

    - `field`: the key this memory owns in the extraction result
      (e.g. `"facts"`).
    - `schema`: the JSON-schema fragment for that field (the value you'd put
      under `properties.<field>`).
    - `instruction`: the human-readable bullet describing this field to the
      LLM (what to extract, value ranges, …).
    """

    field: str
    schema: dict[str, Any]
    instruction: str


class Memory(ABC):
    """Base class for all memory systems."""

    name: str = "memory"

    def __init__(self, *, enabled: bool = True, name: Optional[str] = None) -> None:
        self.enabled = enabled
        if name is not None:
            self.name = name

    @abstractmethod
    def recall(
        self, query: str, user_id: str, limit: int, state_changing: bool = True
    ) -> list[MemoryItem]:
        """Return up to `limit` items relevant to `query` for `user_id`.

        When `state_changing` is `False`, the recall is read-only: memories
        must not mutate any bookkeeping (e.g. recall-count bumps or
        last-recalled timestamps). Useful for inspection / context preview
        without skewing decay statistics.
        """

    @abstractmethod
    def build(self, info_chunks:list[Chunk]) -> None:
        """Build the memory from the given `info_chunks`."""

    @abstractmethod 
    def persist(self, path: str) -> None:
        """Persist the memory to `path`."""

    @abstractmethod 
    def load(self, path: str) -> None:
        """Load the memory from `path`."""

    @property
    def title(self) -> str:
        """Header used when this memory's section is rendered."""
        return self.name.replace("_", " ").title()

    def format(self, items: list[MemoryItem]) -> str:
        """Render recalled `items` into a prompt fragment (override me)."""
        return "\n".join(f"- {it.text}" for it in items)

    def build_section(
        self, query: str, user_id: str, limit: int, state_changing: bool = True
    ) -> Optional[str]:
        """Recall (if enabled) and format; `None` when there is nothing to show.

        `state_changing` is forwarded to :meth:`recall`.
        """
        items = (
            self.recall(query, user_id, limit, state_changing=state_changing)
            if self.enabled
            else []
        )
        if not items:
            return None
        return self.format(items)

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        """Return every memory stored in this memory's backend.

        `limit=0` (the default) returns everything; a positive `limit` caps
        the result. Memories without a backing store return an empty list.
        Override in subclasses that own a database or index.
        """
        return []

    # Extraction
    def extraction_spec(self) -> Optional[ExtractionSpec]:
        """What this memory wants extracted from the conversation, or `None`.

        Returning `None` (the default) opts the memory out of extraction
        entirely. The agent only consults *enabled* memories, so a disabled
        memory is never asked to extract
        """
        return None

    def apply_extraction(self, value: Any, user_id: str) -> list[MemoryItem]:
        """Consume this memory's portion of an extraction result.

        `value` is whatever the LLM produced under this memory's `field`
        (a list, a dict, … depending on the schema). Returns the items that
        were actually added to the memory, so callers (e.g. a deduplicator)
        can act on the freshly-written rows. No-op implementations return `[]`.
        """
        return []
