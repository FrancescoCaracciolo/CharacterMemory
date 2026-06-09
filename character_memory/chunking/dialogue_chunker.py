"""Chunk dialogue transcripts into turn-based windows with context.

Input format (one turn per block, blank-line separated):

    Speaker1: line one
    continuation of speaker1's turn

    Speaker2: reply

Each emitted `Chunk` covers `turns_per_chunk` consecutive turns and is
prefixed with up to `context_width` preceding turns so snippets stay coherent.
"""

import re
from dataclasses import dataclass
from .base import Chunk, Chunker

# Regex to get speakers
_SPEAKER_RE = re.compile(r"^([A-Za-z0-9_?+.\'\- ]{1,30}):\s*(.*)$")

@dataclass
class Turn:
    speaker: str
    text: str

    def render(self) -> str:
        body = self.text.strip()
        return f"{self.speaker}: {body}" if body else f"{self.speaker}:"


class DialogueChunker(Chunker):
    """Turn-window chunker for `Speaker: text` transcripts."""

    name = "dialogue"

    def __init__(
        self,
        turns_per_chunk: int = 6,
        context_width: int = 3,
        stride: int | None = None,
    ) -> None:
        self.turns_per_chunk = turns_per_chunk
        self.context_width = context_width
        # Default to a non-overlapping stride.
        self.stride = stride or turns_per_chunk

    def _parse_turns(self, text: str) -> list[Turn]:
        turns: list[Turn] = []
        for block in re.split(r"\n\s*\n", text):
            lines = [ln for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            m = _SPEAKER_RE.match(lines[0].strip())
            if not m:
                # Not a speaker line; attach to the previous turn if any.
                if turns:
                    turns[-1].text += "\n" + "\n".join(lines)
                continue
            speaker = m.group(1).strip()
            first = m.group(2).strip()
            rest = "\n".join(lines[1:]).strip()
            body = (first + ("\n" + rest if rest else "")).strip()
            turns.append(Turn(speaker=speaker, text=body))
        return turns

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        turns = self._parse_turns(text)
        chunks: list[Chunk] = []
        n = self.turns_per_chunk
        cw = self.context_width
        i = 0
        while i < len(turns):
            window = turns[i : i + n]
            if not window:
                break
            ctx = turns[max(0, i - cw) : i]
            parts = []
            if ctx:
                parts.append("[context]\n" + "\n\n".join(t.render() for t in ctx))
            parts.append("\n\n".join(t.render() for t in window))
            speakers = sorted({t.speaker for t in window})
            chunks.append(
                Chunk(
                    text="\n\n".join(parts),
                    source=source,
                    metadata={
                        "speakers": speakers,
                        "turns": len(window),
                        "context_turns": len(ctx),
                    },
                )
            )
            i += self.stride
        return chunks
