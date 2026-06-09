"""Chunker factory + registry.

Register a new chunker with `register_chunker` (or just pass an instance
directly to the index builder).
"""

from __future__ import annotations

from typing import Callable

from .base import Chunker
from .dialogue_chunker import DialogueChunker
from .header_chunker import MarkdownByHeaderChunker

_FACTORIES: dict[str, Callable[..., Chunker]] = {
    "header": MarkdownByHeaderChunker,
    "dialogue": DialogueChunker,
}


def register_chunker(name: str, factory: Callable[..., Chunker]) -> None:
    _FACTORIES[name] = factory


def get_chunker(name: str, **kwargs) -> Chunker:
    """Instantiate the chunker registered under ``name``."""
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown chunker {name!r}; known: {sorted(_FACTORIES)}"
        ) from exc
    return factory(**kwargs)
