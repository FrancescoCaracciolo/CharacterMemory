"""Chunking interfaces.

A `Chunker` turn a raw document in a list of `Chunk`.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A unit of text destined for the index."""

    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class Chunker(ABC):
    """Base class for all chunking strategies."""

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split ``text`` into chunks, tagging each with ``source``."""
