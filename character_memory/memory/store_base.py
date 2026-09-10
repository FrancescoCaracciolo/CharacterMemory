"""Portable synchronous storage contract used by chats and memories."""
from abc import ABC, abstractmethod
from typing import Any
import re
import sqlite3


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return '"' + value + '"'


class Store(ABC):
    """A store owns transactions, never embedding/model work.

    SQL escape hatches use qmark parameters and the library's SQLite-compatible
    DDL vocabulary. Transaction results support named/positional row access,
    ``fetchone``, ``fetchall``, ``rowcount`` and generated ``lastrowid``.
    """
    operational_errors = (sqlite3.OperationalError,)
    durable_indexes = False

    @abstractmethod
    def create_table(self, name: str, columns: dict[str, str], pk: str = "id") -> None: ...

    @abstractmethod
    def execute(self, sql: str, params=None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def transaction(self, *, immediate: bool = False): ...

    @abstractmethod
    def columns(self, table: str) -> list[str]: ...

    @abstractmethod
    def close(self) -> None: ...

    def upsert(self, table: str, row: dict[str, Any], pk: str = "id") -> Any:
        cols = list(row)
        names = ', '.join(map(identifier, cols))
        updates = ', '.join(f'{identifier(c)}=excluded.{identifier(c)}' for c in cols if c != pk)
        # A self-update gives callers the existing key even for key-only rows.
        updates = updates or f'{identifier(pk)}=excluded.{identifier(pk)}'
        rows = self.execute(
            f'INSERT INTO {identifier(table)} ({names}) VALUES ({", ".join("?" for _ in cols)}) '
            f'ON CONFLICT({identifier(pk)}) DO UPDATE SET {updates} RETURNING {identifier(pk)}',
            [row[c] for c in cols],
        )
        return rows[0][pk]

    def select(self, table, where=None, order_by=None, limit=None, offset=None):
        sql = f'SELECT * FROM {identifier(table)}'
        params = []
        if where:
            sql += ' WHERE ' + ' AND '.join(
                f'{identifier(k)} IS NULL' if v is None else f'{identifier(k)}=?'
                for k, v in where.items()
            )
            params.extend(v for v in where.values() if v is not None)
        if order_by:
            sql += f' ORDER BY {order_by}'  # trusted library SQL expression
        if limit is not None:
            sql += f' LIMIT {max(0, int(limit))}'
        elif offset is not None:
            sql += ' LIMIT -1'
        if offset is not None:
            sql += f' OFFSET {max(0, int(offset))}'
        return self.execute(sql, params)

    def delete(self, table, where):
        if not where:
            raise ValueError('delete requires a filter')
        sql = ' AND '.join(f'{identifier(k)}=?' for k in where)
        self.execute(f'DELETE FROM {identifier(table)} WHERE {sql}', list(where.values()))
