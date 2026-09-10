"""Base class for structured, SQLite-backed memories with decay + RAG recall.

Facts, directives, episodic events and heartbeat entries all follow the same
shape: rows in a SQLite table, a BM25+similarity index over their text,
decay-weighted importance.
Subclasses only declare their columns and how a row renders to text; all the
recall/persist/index plumbing lives here.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..chunking.base import Chunk
from ..config import ContradictionPolicy
from ..rag.base import Query
from ..rag.base import RAGSystem
from .base import Memory, MemoryItem
from .decay import (
    age_seconds,
    decay_score,
    intrinsic_score,
    relevance_repaired_score,
)
from .store_base import Store

if TYPE_CHECKING:
    from ..temporal import TemporalMatch, TemporalResolution, TemporalResolutionEngine

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
    # Retained for source compatibility with older subclasses. Relevance is
    # now part of every candidate's bounded score instead of an alternate sort.
    rank_by_relevance: bool = False
    supports_temporal_resolution: bool = True

    def __init__(
        self,
        store: Store,
        hybrid: RAGSystem,
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
        if getattr(hybrid, "remote", False) and getattr(store, "durable_indexes", False):
            self._repair_lock = hybrid._lock
            store.track_table(self.table, collection=hybrid.collection)
            hybrid._refresh_callback = self._repair_index


    def _repair_index(self) -> bool:
        """Replay transactional source changes; acknowledge only indexed revisions."""
        with self._repair_lock:
            pending = self.store.pending_index(self.hybrid.collection)
            if not pending:
                return False
            # Work is bounded per embedding batch. A newer concurrent revision
            # is retained by the conditional acknowledgement in the backend.
            for start in range(0, len(pending), self.hybrid.batch_size):
                batch = pending[start:start + self.hybrid.batch_size]
                ids = [int(item["row_key"]) for item in batch]
                chunks = []
                placeholders = ','.join('?' for _ in ids)
                rows = self.store.execute(f"SELECT * FROM {self.table} WHERE id IN ({placeholders})", ids)
                for row in rows:
                    chunks.extend(self.index_chunks(row))
                self.hybrid.replace_documents(ids, chunks, pending=batch)
            return True

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
        if getattr(self.hybrid, "remote", False):
            self.hybrid.refresh()
        else:
            self.hybrid.add_documents(self.index_chunks(stored))
        return row_id

    def add_many(
        self, entries: list[tuple[str, float, dict[str, Any]]]
    ) -> list[int]:
        """Store several rows and append their search chunks in one batch."""
        row_ids: list[int] = []
        for user_id, importance, fields in entries:
            row_ids.append(
                self.store.upsert(
                    self.table,
                    {
                        "user_id": user_id,
                        "importance": float(importance),
                        "created_at": self._now(),
                        "last_recalled": None,
                        "recall_count": 0,
                        **fields,
                    },
                )
            )
        if row_ids and getattr(self.hybrid, "remote", False):
            self.hybrid.refresh()
            return row_ids
        if row_ids:
            rows = [self.get_row(row_id) for row_id in row_ids]
            chunks = [
                chunk
                for row in rows
                if row is not None
                for chunk in self.index_chunks(row)
            ]
            self.hybrid.add_documents(chunks)
        return row_ids

    def apply_index_changes(
        self,
        *,
        removed_ids: Iterable[int] = (),
        updated_ids: Iterable[int] = (),
    ) -> None:
        """Incrementally mirror row removals/updates into the hybrid index.

        Updated rows are deleted first, then their current search chunks are
        appended in one embedding batch. Backends without deletion retain the
        legacy one-shot rebuild fallback.
        """
        if getattr(self.hybrid, "remote", False):
            self.hybrid.refresh()
            return
        removed = {int(row_id) for row_id in removed_ids}
        updated = {int(row_id) for row_id in updated_ids}
        affected = removed | updated
        if not affected:
            return
        try:
            self.hybrid.delete_documents(affected)
        except NotImplementedError:
            self.rebuild_index()
            return
        rows = [self.get_row(row_id) for row_id in sorted(updated)]
        chunks = [
            chunk
            for row in rows
            if row is not None
            for chunk in self.index_chunks(row)
        ]
        if chunks:
            self.hybrid.add_documents(chunks)

    def rebuild_index(self) -> None:
        if getattr(self.hybrid, "remote", False):
            with self._repair_lock:
                pending = self.store.pending_index(self.hybrid.collection)
                rows = self.store.select(self.table)
                chunks = [chunk for row in rows for chunk in self.index_chunks(row)]
                self.hybrid.build(chunks, pending=pending)
        else:
            rows = self.store.select(self.table)
            chunks = [chunk for row in rows for chunk in self.index_chunks(row)]
            self.hybrid.build(chunks)

    # RECALL functions
    def _effective(self, row: dict[str, Any]) -> float:
        """Query-independent familiar salience (override for custom models)."""
        return decay_score(
            base_importance=float(row["importance"]),
            recall_count=int(row.get("recall_count", 0)),
            age_seconds=age_seconds(
                self._age_anchor(row), row.get("last_recalled"), self._now()
            ),
            half_life=self.half_life,
        )

    def _age_anchor(self, row: dict[str, Any]) -> float:
        """Creation/event timestamp from which this memory actually ages."""
        semantic_time = row.get("occurred_at")
        return float(
            semantic_time
            if semantic_time is not None
            else row["created_at"]
        )

    def _intrinsic(self, row: dict[str, Any]) -> float:
        """Salience before decay/familiarity (override with `_effective`)."""
        return intrinsic_score(float(row["importance"]))

    def _rank_score(self, row: dict[str, Any], relevance: float) -> float:
        """Final candidate score after bounded relevance repair."""
        return relevance_repaired_score(
            self._intrinsic(row), self._effective(row), relevance
        )

    def _recall_where(self, user_id: str) -> dict[str, Any] | None:
        """Hybrid/SQLite filter for one recall; character memories override."""
        return {"user_id": user_id}

    def _sticky_rows(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rows eligible for the one-slot sticky reservation."""
        return [
            row
            for row in rows
            if float(row["importance"]) >= self.sticky_threshold
        ]

    def _additional_relevance(
        self,
        query: Query,
        rows_by_id: dict[int, dict[str, Any]],
    ) -> dict[int, float]:
        """Subclass hook for non-RAG triggers such as directive keywords."""
        return {}

    def temporal_interval(
        self, row: dict[str, Any]
    ) -> tuple[float, Optional[float]] | None:
        """Semantic event interval for temporal matching.

        The default intentionally returns ``None``: ``created_at`` records
        when a fact/profile was learned, not when that fact happened. Event
        memories override this with their real occurrence timestamp.
        """
        return None

    def _temporal_relevance(
        self,
        rows_by_id: dict[int, dict[str, Any]],
        resolution: Optional["TemporalResolution"],
    ) -> dict[int, "TemporalMatch"]:
        if resolution is None or resolution.is_empty:
            return {}
        matches: dict[int, "TemporalMatch"] = {}
        for row_id, row in rows_by_id.items():
            interval = self.temporal_interval(row)
            if interval is None:
                continue
            match = resolution.match(interval[0], interval[1])
            if match.score > 0.0:
                matches[row_id] = match
        return matches

    @staticmethod
    def _hit_relevance(hits: list[Any]) -> dict[int, float]:
        """Map row ids to normalized relevance, with a generic-RAG fallback."""
        positive = [max(0.0, float(hit.score)) for hit in hits]
        fallback_max = max(positive, default=0.0)
        relevance: dict[int, float] = {}
        for hit in hits:
            rid = hit.metadata.get("id")
            if rid is None:
                continue
            normalized = hit.metadata.get("normalized_relevance")
            if normalized is None:
                normalized = (
                    max(0.0, float(hit.score)) / fallback_max
                    if fallback_max > 0.0
                    else 0.0
                )
            try:
                row_id = int(rid)
            except (TypeError, ValueError):
                # Structured rows are integer SQLite ids. A malformed or
                # foreign backend hit is not a candidate for this memory.
                continue
            relevance[row_id] = max(
                relevance.get(row_id, 0.0),
                max(0.0, min(1.0, float(normalized))),
            )
        return relevance

    def _scored_item(
        self,
        row: dict[str, Any],
        *,
        score: float,
        semantic_relevance: float,
        temporal_relevance: float,
        combined_relevance: float,
        temporal_match: Optional["TemporalMatch"] = None,
    ) -> MemoryItem:
        item = self.row_item(row, score)
        item.metadata.update({
            "intrinsic_score": self._intrinsic(row),
            "familiar_score": self._effective(row),
            "retrieval_relevance": combined_relevance,
            "semantic_relevance": semantic_relevance,
            "temporal_relevance": temporal_relevance,
            "combined_relevance": combined_relevance,
            "ranking_score": score,
        })
        if temporal_match is not None and temporal_match.range is not None:
            item.metadata.update({
                "temporal_expression": temporal_match.range.expression,
                "temporal_range_start": temporal_match.range.start_timestamp,
                "temporal_range_end": temporal_match.range.end_timestamp,
                "temporal_grain": temporal_match.range.grain,
            })
        return item

    def recall(
        self,
        query: Query,
        user_id: str,
        limit: int,
        sticky_limit: int = 10,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: float = 1.0,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []

        where = self._recall_where(user_id)
        rows = self.store.select(self.table, where)
        rows_by_id = {int(r["id"]): r for r in rows}
        if not rows_by_id:
            return []

        # Search forms the query candidates. RAGSystem supplies an absolute
        # normalized score; third-party RAG backends fall back to normalization
        # against their best hit for this call.
        candidate_pool = int(getattr(self.hybrid, "candidate_pool", max(limit, 30)))
        hits = self.hybrid.search(
            query,
            k=max(limit, candidate_pool),
            where=where,
        )
        relevance_by_id = {
            rid: relevance
            for rid, relevance in self._hit_relevance(hits).items()
            if rid in rows_by_id
        }
        for rid, relevance in self._additional_relevance(query, rows_by_id).items():
            if rid in rows_by_id:
                relevance_by_id[rid] = max(
                    relevance_by_id.get(rid, 0.0),
                    max(0.0, min(1.0, float(relevance))),
                )

        resolution = self.resolve_temporal(
            query, temporal_resolution, temporal_resolution_engine
        )
        temporal_by_id = self._temporal_relevance(rows_by_id, resolution)
        weight = max(0.0, min(1.0, float(temporal_weight)))

        sticky = self._sticky_rows(rows)
        sticky_ids = {int(row["id"]) for row in sticky}
        reserve_sticky = limit > 1 and sticky_limit > 0 and bool(sticky_ids)
        candidate_ids = set(relevance_by_id)
        candidate_ids.update(temporal_by_id)
        if reserve_sticky:
            candidate_ids.update(sticky_ids)

        scored: list[
            tuple[float, int, dict[str, Any], float, float, float, Optional["TemporalMatch"]]
        ] = []
        for rid in candidate_ids:
            row = rows_by_id[rid]
            semantic = relevance_by_id.get(rid, 0.0)
            temporal_match = temporal_by_id.get(rid)
            temporal = temporal_match.score if temporal_match is not None else 0.0
            weighted_temporal = weight * temporal
            # Probabilistic OR: an exact temporal hit is as useful as a strong
            # semantic hit, while two partial signals reinforce one another.
            combined = 1.0 - (1.0 - semantic) * (1.0 - weighted_temporal)
            scored.append((
                self._rank_score(row, combined), rid, row, semantic,
                temporal, combined, temporal_match,
            ))
        scored.sort(key=lambda entry: (entry[0], entry[5], entry[4], entry[3], -entry[1]), reverse=True)
        if reserve_sticky:
            # A single partitioned slot prevents a collection of fresh,
            # high-importance but irrelevant sticky rows from consuming the
            # entire prompt. Query relevance still decides which sticky row
            # wins that slot.
            sticky_entries = [entry for entry in scored if entry[1] in sticky_ids]
            ordinary_entries = [
                entry for entry in scored if entry[1] not in sticky_ids
            ]
            chosen = [
                *ordinary_entries[: limit - 1],
                max(
                    sticky_entries,
                    key=lambda entry: (
                        entry[0],
                        entry[5],
                        entry[4],
                        entry[3],
                        self._effective(entry[2]),
                        -entry[1],
                    ),
                ),
            ]
            chosen.sort(
                key=lambda entry: (entry[0], entry[5], entry[4], entry[3], -entry[1]),
                reverse=True,
            )
        else:
            chosen = scored[:limit]

        # Bump recall counters for what we surfaced (skipped when read-only).
        if state_changing:
            self._bump_recall([entry[2]["id"] for entry in chosen])

        return [
            self._scored_item(
                row,
                score=score,
                semantic_relevance=semantic,
                temporal_relevance=temporal,
                combined_relevance=combined,
                temporal_match=temporal_match,
            )
            for score, _, row, semantic, temporal, combined, temporal_match in chosen
        ]

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
