"""Read-side adapters that turn each `Memory` into paged, searchable records.

The GUI never talks to a memory subclass directly. Instead it goes through a
`MemoryAdapter`, which normalises the wildly different back-ends — SQLite rows
for the structured memories, in-RAM chunk nodes for the RAG memories, the
per-user JSON blob for emotion — into one `MemoryRecord` shape the frontend can
render.

Binding is by **base class**, not by name: any new `StructuredMemory` /
`RAGMemory` / `EmotionStatus` subclass is rendered by the matching adapter for
free (matching the library's "one new subclass, nothing else changes" rule).
A per-name override hook (`register`) lets a specific memory type swap in a
tailored adapter without touching the rest.

The two public entry points are :func:`overview` (sidebar: every memory with a
count) and :func:`read_memory` (one memory, one page, optional search + user
filter). Search prefers the memory's own hybrid (semantic) retrieval and falls
back to lexical matching when the embedding server is unreachable, so the GUI
stays useful offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from character_memory import EmotionStatus, Memory, StructuredMemory
from character_memory.memory.character_base import RAGMemory

# Hard cap on how many hits search ever ranks, so a query against a huge memory
# stays snappy. Pagination slices within this ranked window.
SEARCH_CAP = 200


# --------------------------------------------------------------------------- #
# Normalised shapes the API returns.
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    """One row/chunk/state, normalised for the frontend.

    `text` is a default display string; `fields` carries the raw, type-specific
    columns so a custom per-memory renderer can show whatever it wants. `score`
    is the search relevance (search mode) or effective-importance (browse mode);
    `meta` holds decay/recency bookkeeping for the structured memories.
    """

    id: Any
    user_id: Optional[str]
    text: str
    score: Optional[float]
    fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter base + builtins.
# --------------------------------------------------------------------------- #
class MemoryAdapter:
    """Normalise one memory backend into paged/searchable records."""

    kind = "generic"

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    @property
    def title(self) -> str:
        return self.memory.title

    def count(self, user_id: Optional[str] = None) -> int:
        return 0

    def users(self) -> list[str]:
        return []

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        raise NotImplementedError

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        return [], 0


class StructuredAdapter(MemoryAdapter):
    """SQLite rows + hybrid recall (facts / directives / episodes / heartbeat)."""

    kind = "structured"

    @property
    def m(self) -> StructuredMemory:
        return self.memory  # type: ignore[return-type]

    def _where(self, user_id: Optional[str]) -> Optional[dict[str, Any]]:
        return {"user_id": user_id} if user_id else None

    def count(self, user_id: Optional[str] = None) -> int:
        stmt = f"SELECT COUNT(*) AS c FROM {self.m.table}"
        params: list[Any] = []
        if user_id:
            stmt += " WHERE user_id = ?"
            params.append(user_id)
        rows = self.m.store.execute(stmt, params)
        return int(rows[0]["c"]) if rows else 0

    def users(self) -> list[str]:
        rows = self.m.store.execute(
            f"SELECT DISTINCT user_id FROM {self.m.table} ORDER BY user_id"
        )
        return [r["user_id"] for r in rows]

    def _record(self, row: dict[str, Any], score: Optional[float] = None) -> MemoryRecord:
        # Use the memory's own effective-importance (episodic weights in emotion).
        eff = self._safe_effective(row)
        return MemoryRecord(
            id=row.get("id"),
            user_id=row.get("user_id"),
            text=self.m.row_text(row),
            score=score if score is not None else eff,
            fields=dict(row),
            meta={
                "effective": eff,
                "importance": row.get("importance"),
                "recall_count": row.get("recall_count"),
                "created_at": row.get("created_at"),
                "last_recalled": row.get("last_recalled"),
            },
        )

    def _safe_effective(self, row: dict[str, Any]) -> Optional[float]:
        try:
            return float(self.m._effective(row))
        except Exception:
            return None

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        total = self.count(user_id)
        rows = self.m.store.select(
            self.m.table,
            where=self._where(user_id),
            order_by="id DESC",
            limit=size,
            offset=(page - 1) * size,
        )
        return [self._record(r) for r in rows], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ranked = self._ranked_rows(q, user_id)
        total = len(ranked)
        start = (page - 1) * size
        return [self._record(r, score=s) for r, s in ranked[start : start + size]], total

    def _ranked_rows(self, q: str, user_id: Optional[str]) -> list[tuple[dict, float]]:
        q = (q or "").strip()
        if not q:
            return []
        where = self._where(user_id)
        # Prefer semantic + lexical hybrid recall (needs the embedding server).
        try:
            hits = self.m.hybrid.search(q, k=SEARCH_CAP, where=where)
        except Exception:
            hits = []
        if hits:
            rows_by_id = {r["id"]: r for r in self.m.store.select(self.m.table, where=where)}
            out: list[tuple[dict, float]] = []
            for h in hits:
                rid = h.metadata.get("id")
                row = rows_by_id.get(rid)
                if row:
                    out.append((row, float(h.metadata.get("similarity") or h.score)))
            if out:
                return out
        # Lexical fallback (embedding server down / no hits): substring + count.
        return self._lexical(q, user_id)

    def _lexical(self, q: str, user_id: Optional[str]) -> list[tuple[dict, float]]:
        ql = q.lower()
        rows = self.m.store.select(self.m.table, where=self._where(user_id))
        scored: list[tuple[float, dict]] = []
        for r in rows:
            hay = (self.m.row_text(r) or "").lower()
            if ql in hay:
                scored.append((float(hay.count(ql)), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(r, s) for s, r in scored]


class RAGAdapter(MemoryAdapter):
    """In-RAM chunk nodes (character_info / dialogue_style)."""

    kind = "rag"

    @property
    def m(self) -> RAGMemory:
        return self.memory  # type: ignore[return-type]

    def _docs(self):
        try:
            return self.m.hybrid.documents
        except Exception:
            return []

    def count(self, user_id: Optional[str] = None) -> int:
        try:
            return int(self.m.hybrid.count)
        except Exception:
            return len(self._docs())

    def users(self) -> list[str]:
        return []

    def _record(self, doc, idx: int, score: Optional[float] = None) -> MemoryRecord:
        meta = dict(getattr(doc, "metadata", None) or {})
        text = getattr(doc, "text", "") or ""
        return MemoryRecord(
            id=meta.get("_pos", idx),
            user_id=meta.get("user_id"),
            text=text,
            score=score,
            fields={"source": meta.get("source", ""), "metadata": meta},
            meta={"source": meta.get("source", ""), "pos": meta.get("_pos", idx)},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        docs = self._docs()
        total = len(docs)
        start = (page - 1) * size
        window = docs[start : start + size]
        return [self._record(d, start + i) for i, d in enumerate(window)], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        q = (q or "").strip()
        if not q:
            return [], 0
        try:
            hits = self.m.hybrid.search(q, k=SEARCH_CAP)
            ranked = [
                (h.text, float(h.score), dict(h.metadata or {}))
                for h in hits
            ]
        except Exception:
            ranked = []
        if not ranked:  # lexical fallback over the in-RAM nodes
            ql = q.lower()
            for i, d in enumerate(self._docs()):
                text = getattr(d, "text", "") or ""
                if ql in text.lower():
                    meta = dict(getattr(d, "metadata", None) or {})
                    meta.setdefault("_pos", i)
                    ranked.append((text, float(text.lower().count(ql)), meta))
        total = len(ranked)
        start = (page - 1) * size
        recs = [
            MemoryRecord(
                id=m.get("_pos", 0),
                user_id=m.get("user_id"),
                text=text,
                score=score,
                fields={"source": m.get("source", ""), "metadata": m},
                meta={"source": m.get("source", "")},
            )
            for text, score, m in ranked[start : start + size]
        ]
        return recs, total


class EmotionAdapter(MemoryAdapter):
    """Per-user emotion state blobs (+ baseline exposed via page extra)."""

    kind = "emotion"

    @property
    def m(self) -> EmotionStatus:
        return self.memory  # type: ignore[return-type]

    def _rows(self, user_id: Optional[str] = None) -> list[dict]:
        return self.m.store.select(
            self.m.table,
            where={"user_id": user_id} if user_id else None,
            order_by="user_id",
        )

    def count(self, user_id: Optional[str] = None) -> int:
        return len(self._rows(user_id))

    def users(self) -> list[str]:
        return [r["user_id"] for r in self._rows()]

    def baseline(self) -> dict[str, float]:
        try:
            return {k: float(v) for k, v in self.m.baseline.items()}
        except Exception:
            return {}

    def _record(self, row: dict) -> MemoryRecord:
        try:
            blob = json.loads(row.get("state") or "{}")
        except (TypeError, ValueError):
            blob = {}
        # The blob holds numeric dims plus a string `comment` (relationship
        # descriptor). Split them so the dims render as bars and the comment
        # surfaces as text.
        comment = ""
        state: dict[str, float] = {}
        for k, v in blob.items():
            if k == "comment":
                comment = str(v or "")
            else:
                try:
                    state[k] = float(v)
                except (TypeError, ValueError):
                    continue
        text = ", ".join(f"{k}={v:.2f}" for k, v in state.items())
        if comment:
            text = (text + " | " if text else "") + f"relationship: {comment}"
        return MemoryRecord(
            id=row.get("user_id"),
            user_id=row.get("user_id"),
            text=text,
            score=None,
            fields={"state": state, "comment": comment},
            meta={"comment": comment} if comment else {},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        rows = self.m.store.select(
            self.m.table,
            where={"user_id": user_id} if user_id else None,
            order_by="user_id",
            limit=size,
            offset=(page - 1) * size,
        )
        return [self._record(r) for r in rows], self.count(user_id)

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ql = (q or "").lower().strip()
        rows = [r for r in self._rows(user_id) if not ql or ql in r["user_id"].lower()]
        total = len(rows)
        start = (page - 1) * size
        return [self._record(r) for r in rows[start : start + size]], total


class GenericAdapter(MemoryAdapter):
    """Fallback: anything exposing `get_memories()`."""

    kind = "generic"

    def _all(self) -> list:
        try:
            return list(self.memory.get_memories(0))
        except Exception:
            return []

    def count(self, user_id: Optional[str] = None) -> int:
        return len(self._all())

    def _record(self, item, idx: int, score: Optional[float] = None) -> MemoryRecord:
        meta = dict(getattr(item, "metadata", None) or {})
        return MemoryRecord(
            id=meta.get("id", idx),
            user_id=meta.get("user_id"),
            text=getattr(item, "text", "") or "",
            score=getattr(item, "score", None) if score is None else score,
            fields=meta,
            meta={"kind": getattr(item, "kind", "")},
        )

    def page(
        self, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        items = self._all()
        total = len(items)
        start = (page - 1) * size
        return [self._record(it, start + i) for i, it in enumerate(items[start : start + size])], total

    def search(
        self, q: str, page: int, size: int, user_id: Optional[str] = None
    ) -> tuple[list[MemoryRecord], int]:
        ql = (q or "").lower().strip()
        items = [it for it in self._all() if not ql or ql in (getattr(it, "text", "") or "").lower()]
        total = len(items)
        start = (page - 1) * size
        return [self._record(it, start + i) for i, it in enumerate(items[start : start + size])], total


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #
_BY_NAME: dict[str, type[MemoryAdapter]] = {}


def register(name: str, adapter_cls: type[MemoryAdapter]) -> None:
    """Override the adapter for one memory `name` (tailor a single type)."""
    _BY_NAME[name] = adapter_cls


def get_adapter(memory: Memory) -> MemoryAdapter:
    """Pick an adapter: explicit per-name override, else bind by base class."""
    cls = _BY_NAME.get(memory.name)
    if cls is not None:
        return cls(memory)
    if isinstance(memory, EmotionStatus):
        return EmotionAdapter(memory)
    if isinstance(memory, RAGMemory):
        return RAGAdapter(memory)
    if isinstance(memory, StructuredMemory):
        return StructuredAdapter(memory)
    return GenericAdapter(memory)


# --------------------------------------------------------------------------- #
# Public entry points used by the API.
# --------------------------------------------------------------------------- #
def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


def overview(agent) -> list[dict[str, Any]]:
    """One summary per memory (sidebar)."""
    out: list[dict[str, Any]] = []
    for name, mem in agent.memories.items():
        adapter = get_adapter(mem)
        out.append(
            {
                "name": name,
                "title": mem.title,
                "kind": adapter.kind,
                "enabled": bool(getattr(mem, "enabled", True)),
                "count": _safe(adapter.count, default=0) or 0,
                "users": _safe(adapter.users, default=[]) or [],
            }
        )
    return out


def read_memory(
    agent,
    name: str,
    *,
    page: int = 1,
    size: int = 25,
    user: Optional[str] = None,
    q: Optional[str] = None,
) -> dict[str, Any]:
    """One page of records for one memory, with optional search + user filter."""
    mem = agent.memories.get(name)
    if mem is None:
        raise KeyError(name)

    adapter = get_adapter(mem)
    page = max(1, int(page or 1))
    size = max(1, min(100, int(size or 25)))
    q = (q or "").strip()
    user = user or None

    if q:
        records, total = _safe(
            adapter.search, q, page, size, user, default=([], 0)
        ) or ([], 0)
        search = True
    else:
        records, total = _safe(
            adapter.page, page, size, user, default=([], 0)
        ) or ([], 0)
        search = False

    pages = max(1, math.ceil(total / size)) if size else 1
    extra: dict[str, Any] = {}
    if isinstance(adapter, EmotionAdapter):
        extra["baseline"] = adapter.baseline()

    return {
        "character": agent.character_name,
        "memory": name,
        "title": mem.title,
        "kind": adapter.kind,
        "enabled": bool(getattr(mem, "enabled", True)),
        "page": page,
        "size": size,
        "total": total,
        "pages": pages,
        "search": search,
        "query": q,
        "user": user,
        "users": _safe(adapter.users, default=[]) or [],
        "records": [asdict(r) for r in records],
        "extra": extra,
    }
