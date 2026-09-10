"""Shared SQLite backend for structured memories.

One `SQLiteStore` holds a single connection (one `.db` per character)
shared by all structured memories. Each memory owns a table; the store handles
table creation, upserts, and parameterised selects so the memories never write
raw SQL.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .store_base import Store


class SQLiteStore(Store):
    """Thin, thread-safe wrapper around a sqlite3 connection."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Allow concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()

    def create_table(self, name: str, columns: dict[str, str], pk: str = "id") -> None:
        cols_sql = ", ".join(f"{c} {t}" for c, t in columns.items())
        with self._lock:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols_sql})")
            self._conn.commit()

    def execute(self, sql: str, params: Optional[list[Any]] = None) -> list[dict[str, Any]]:
        with self._lock:
            try:
                rows = self._conn.execute(sql, params or []).fetchall()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return [dict(r) for r in rows]

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield the connection inside one store-wide transaction.

        This is intentionally a small escape hatch for persistence code that
        must update several related rows atomically.  Callers execute SQL on
        the yielded connection directly; nesting regular ``SQLiteStore``
        methods inside the block would try to acquire the same non-reentrant
        lock and is therefore unsupported.

        ``immediate=True`` obtains SQLite's write reservation up front.  It is
        useful for snapshot-style writers such as the knowledge graph, where
        another process must not interleave a second graph snapshot between
        the first row and the last.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def columns(self, table: str) -> list[str]:
        """Column names of `table` (empty if the table does not exist)."""
        with self._lock:
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r["name"] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
