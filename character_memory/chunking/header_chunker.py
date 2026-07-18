"""Chunk markdown by headers, with a token-budget fallback for long sections."""

import re
from typing import Callable

from .base import Chunk, Chunker

# Regex to find markdown headers
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _token_counter() -> Callable[[str], int]:
    """Return a token counter; fall back to charcount/4 tiktoken
    is unavailable."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:  # pragma: no cover - defensive
        return lambda s: int(len(s)/4)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


class MarkdownByHeaderChunker(Chunker):
    """Split on markdown headers; overflow each section to `max_tokens`.

    The header line is kept as a prefix on every sub-chunk so context is
    preserved across the token-budget boundaries.
    """

    name = "header"

    def __init__(
        self, max_tokens: int = 512, header_level: int = 2, min_tokens: int = 64
    ) -> None:
        self.max_tokens = max_tokens
        self.header_level = header_level
        # Chunks below `min_tokens` are merged into a neighbour instead of being
        # emitted as near-empty fragments (e.g. a 540-token section leaving a
        # ~28-token tail after the 512-token budget split).
        self.min_tokens = max(0, int(min_tokens))
        self._count = _token_counter()

    def _sections(self, text: str) -> list[tuple[str, str]]:
        """Return ``(title, body)`` pairs, including a leading untitled part."""
        cuts = [
            (m.start(), len(m.group(1)), m.group(2).strip())
            for m in _HEADER_RE.finditer(text)
            if len(m.group(1)) <= self.header_level
        ]
        if not cuts:
            return [("", text.strip())]
        sections: list[tuple[str, str]] = []
        # Content before the first header.
        head = text[: cuts[0][0]].strip()
        if head:
            sections.append(("", head))
        for i, (start, _depth, title) in enumerate(cuts):  # noqa: B007
            header_end = _HEADER_RE.match(text[start:]).end()
            end = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
            body = text[start + header_end : end].strip()
            sections.append((title, body))
        return sections

    def _budget_split(self, title: str, body: str, source: str) -> list[Chunk]:
        prefix = f"# {title}\n\n" if title else ""
        # Whole section fits.
        if self._count(prefix + body) <= self.max_tokens:
            return [Chunk(text=prefix + body, source=source, metadata={"header": title})]

        chunks: list[Chunk] = []
        buf = prefix
        for para in _split_paragraphs(body):
            candidate = (buf + "\n\n" + para) if buf and buf != prefix else (
                buf + para if buf == prefix else para
            )
            if self._count(candidate) <= self.max_tokens:
                buf = candidate
            else:
                if buf and buf.strip():
                    chunks.append(Chunk(text=buf, source=source, metadata={"header": title}))
                # If a single paragraph still exceeds the budget, hard-split it.
                if self._count(prefix + para) > self.max_tokens:
                    chunks.extend(self._hard_split(prefix, para, source, title))
                    buf = prefix
                else:
                    buf = prefix + para
        if buf and buf.strip():
            chunks.append(Chunk(text=buf, source=source, metadata={"header": title}))
        return self._merge_tiny(chunks, source, title)

    def _merge_tiny(
        self, chunks: list[Chunk], source: str, title: str
    ) -> list[Chunk]:
        """Fold near-empty chunks into a neighbour so they are not indexed alone.

        Any chunk below `min_tokens` (tail, middle or leading) is merged into the
        previous chunk; a leading tiny chunk merges into its follower. Disabled when
        `min_tokens <= 0`. Merged text is separated by a blank line and keeps the
        section's `title` metadata.
        """
        if len(chunks) < 2 or self.min_tokens <= 0:
            return chunks

        def _join(a: Chunk, b: Chunk) -> Chunk:
            return Chunk(
                text=(a.text + "\n\n" + b.text).strip(),
                source=source,
                metadata={"header": title},
            )

        out: list[Chunk] = [chunks[0]]
        for c in chunks[1:]:
            if self._count(c.text) < self.min_tokens:
                out[-1] = _join(out[-1], c)
            else:
                out.append(c)
        if len(out) >= 2 and self._count(out[0].text) < self.min_tokens:
            out = [_join(out[0], out[1]), *out[2:]]
        return out

    def _hard_split(self, prefix: str, para: str, source: str, title: str) -> list[Chunk]:
        words = para.split()
        out: list[Chunk] = []
        cur = prefix
        for w in words:
            if self._count(cur + " " + w) > self.max_tokens and cur.strip():
                out.append(Chunk(text=cur, source=source, metadata={"header": title}))
                cur = prefix + w
            else:
                cur = (cur + " " + w) if cur and cur != prefix else prefix + w
        if cur.strip():
            out.append(Chunk(text=cur, source=source, metadata={"header": title}))
        return out

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        for title, body in self._sections(text):
            if not body:
                continue
            chunks.extend(self._budget_split(title, body, source))
        return chunks
