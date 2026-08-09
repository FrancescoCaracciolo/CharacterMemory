"""Base class for structured, SQLite-backed memories with decay + RAG recall.

Facts, directives, episodic events and heartbeat entries all follow the same
shape: rows in a SQLite table, a BM25+similarity index over their text,
decay-weighted importance.
Subclasses only declare their columns and how a row renders to text; all the
recall/persist/index plumbing lives here.
"""

import time
from typing import Any, Callable, Optional

from ..chunking.base import Chunk
from ..config import ContradictionPolicy
from ..rag.hybrid import HybridSearch
from .base import Memory, MemoryItem
from .decay import age_seconds, decay_score
from .store import SQLiteStore

# Columns every structured memory gets for free (decay + multi-user bookkeeping).
COMMON_COLUMNS: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "user_id": "TEXT NOT NULL",
    "importance": "REAL NOT NULL DEFAULT 0.5",
    "created_at": "REAL NOT NULL",
    "last_recalled": "REAL",
    "recall_count": "INTEGER NOT NULL DEFAULT 0",
}


class StructuredMemory(Memory):
    """Generic structured memory with hybrid recall and decay."""

    table: str = "structured"
    extra_columns: dict[str, str] = {}  # memory-specific columns
    text_column: str = "content"        # column holding the primary text for this memory
    # Most structured memories use hybrid search only to form a candidate set,
    # then rank by salience/decay. Source-event memory opts into retrieval order
    # because all of its immutable rows intentionally have equal importance.
    rank_by_relevance: bool = False

    def __init__(
        self,
        store: SQLiteStore,
        hybrid: HybridSearch,
        *,
        enabled: bool = True,
        half_life: float = 60 * 60 * 24 * 7,
        sticky_threshold: float = 0.8,
        clock: Optional[Callable[[], float]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(enabled=enabled, name=name)
        self.store = store
        self.hybrid = hybrid
        self.half_life = half_life
        self.sticky_threshold = sticky_threshold
        self._now = clock or time.time
        cols = {**COMMON_COLUMNS, **self.extra_columns}
        self.store.create_table(self.table, cols)

    def build(self, info_chunks: list[Chunk]) -> None:
        return self.hybrid.build(info_chunks)

    def persist(self, path: str) -> None:
        return self.hybrid.persist(path)

    def load(self, path: str) -> None:
        return self.hybrid.load(path)
    
    # MAPPING UTILITIES
    def row_text(self, row: dict[str, Any]) -> str:
        """Text used for embedding + BM25 (override me)."""
        return str(row.get("content", ""))

    def row_item(self, row: dict[str, Any], score: float) -> MemoryItem:
        """Render a row into a prompt item (override me)."""
        return MemoryItem(text=self.row_text(row), score=score, kind=self.name, metadata=dict(row))

    def _row_meta(self, row: dict[str, Any]) -> dict:
        return {"id": row["id"], "user_id": row["user_id"]}

    def index_chunks(self, row: dict[str, Any]) -> list[Chunk]:
        """Search keys contributed by one stored row.

        The default is the row's primary text. Subclasses may add secondary
        keys while keeping one SQLite row as the value returned to the prompt.
        """
        return [
            Chunk(
                text=self.row_text(row),
                source=self.table,
                metadata=self._row_meta(row),
            )
        ]

    # Extraction helpers (used by subclasses' apply_extraction).
    def _has_text(self, user_id: str, text: str, content_col: str = "content") -> bool:
        """True if a row for `user_id` already stores `text` (case-insensitive)."""
        rows = self.store.select(self.table, {"user_id": user_id})
        return any((r.get(content_col) or "").strip().lower() == text.strip().lower() for r in rows)

    @staticmethod
    def _clip(v: Any, lo: float = 0.0, hi: float = 1.0) -> float:
        """Coerce a possibly-bad LLM value into a clamped float in [lo, hi]."""
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = 0.5
        return max(lo, min(hi, x))

    # ADD ELEMENT
    def add(self, user_id: str, importance: float, **fields) -> int:
        now = self._now()
        row = {
            "user_id": user_id,
            "importance": float(importance),
            "created_at": now,
            "last_recalled": None,
            "recall_count": 0,
            **fields,
        }
        row_id = self.store.upsert(self.table, row)
        # Index the new row so BM25+similarity can find it.
        stored = self.store.select(self.table, {"id": row_id})[0]
        self.hybrid.add_documents(self.index_chunks(stored))
        return row_id

    def rebuild_index(self) -> None:
        rows = self.store.select(self.table)
        chunks = [chunk for row in rows for chunk in self.index_chunks(row)]
        self.hybrid.build(chunks)

    # RECALL functions
    def _effective(self, row: dict[str, Any]) -> float:
        # For emotional impact it must be overridden
        return decay_score(
            base_importance=float(row["importance"]),
            recall_count=int(row.get("recall_count", 0)),
            age_seconds=age_seconds(
                float(row["created_at"]), row.get("last_recalled"), self._now()
            ),
            half_life=self.half_life,
        )

    def recall(
        self,
        query: str,
        user_id: str,
        limit: int,
        sticky_limit: int = 10,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        rows_by_id = {r["id"]: r for r in self.store.select(self.table, {"user_id": user_id})}
        if not rows_by_id:
            return []

        # BM25 + similarity retrieval for this user.
        hits = self.hybrid.search(query, k=max(limit, self.hybrid.candidate_pool), where={"user_id": user_id})
        retrieval_ranks: dict[int, int] = {}
        for rank, hit in enumerate(hits):
            rid = hit.metadata.get("id")
            if rid in rows_by_id:
                retrieval_ranks.setdefault(int(rid), rank)
        candidate_ids: set[int] = set(retrieval_ranks)

        # High base-importance, injected regardless of query.
        i = 0
        for rid, row in rows_by_id.items():
            if float(row["importance"]) >= self.sticky_threshold:
                candidate_ids.add(rid)
            i+=1
            if i >= sticky_limit:
                break

        scored = []
        for rid in candidate_ids:
            row = rows_by_id[rid]
            scored.append((self._effective(row), row))
        if self.rank_by_relevance:
            scored.sort(
                key=lambda kv: (
                    retrieval_ranks.get(int(kv[1]["id"]), len(hits) + 1),
                    -kv[0],
                )
            )
        else:
            scored.sort(key=lambda kv: kv[0], reverse=True)
        chosen = scored[:limit]

        # Bump recall counters for what we surfaced (skipped when read-only).
        if state_changing:
            self._bump_recall([row["id"] for _, row in chosen])

        return [self.row_item(row, score) for score, row in chosen]

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        """All rows in this memory's table, rendered as items.

        `limit=0` returns every row; otherwise the top `limit` by id order.
        """
        rows = self.all_rows()
        if limit and limit > 0:
            rows = rows[:limit]
        return [self.row_item(r, self._effective(r)) for r in rows]

    def _bump_recall(self, ids: list[int]) -> None:
        if not ids:
            return
        now = self._now()
        for rid in ids:
            self.store.execute(
                f"UPDATE {self.table} SET recall_count = recall_count + 1, last_recalled = ? WHERE id = ?",
                [now, rid],
            )

    def all_rows(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.store.select(self.table, {"user_id": user_id} if user_id else None, order_by="id")

    def get_row(self, row_id: int) -> Optional[dict[str, Any]]:
        rows = self.store.select(self.table, {"id": row_id})
        return rows[0] if rows else None

    def update_row(self, row: dict[str, Any]) -> None:
        """Write a (possibly merged) row back via upsert. Does not touch the index."""
        self.store.upsert(self.table, row)

    def delete_row(self, row_id: int) -> None:
        """Delete a row by id. Does not touch the index."""
        self.store.delete(self.table, {"id": row_id})

    # Contradiction resolution policy — see character_memory.memory.dedup.
    def contradiction_policy(self) -> ContradictionPolicy:
        """How this memory wants contradicting facts resolved by the deduplicator.

        Default is disabled. Stable-fact memories override this to enable it
        and tune the similarity bar / candidate pool. The deduplicator calls
        this itself; callers normally don't.
        """
        return ContradictionPolicy()
