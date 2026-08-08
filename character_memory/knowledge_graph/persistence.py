"""Persistence for the knowledge graph.

Two stores, both per-character under `<save_directory>/kg_index/`:

- the **SQLite** tables `kg_nodes` and `kg_edges` in the shared `SQLiteStore`
  (one connection per character, same memory.db the other memories use);
- the **node-text hybrid index** (BM25 + dense similarity, fused with RRF)
  via the same `HybridSearch` reused everywhere in the library, persisted as
  `kg_index/nodes.json` + `kg_index/faiss.index`.

`load(path)` is the inverse of `save(path)` and preserves the rebuild-on-
embedding-dim-drift behaviour of `HybridSearch` (the dense index is rebuilt
from `nodes.json` when the stored dim no longer matches the live embedder).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Optional

try:  # POSIX (the library's supported server/CLI environments)
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from ..chunking.base import Chunk
from ..memory.store import SQLiteStore
from ..rag.hybrid import HybridSearch
from .edges import edge_from_dict
from .graph import KnowledgeGraph
from .nodes import node_from_dict

NODES_TABLE = "kg_nodes"
EDGES_TABLE = "kg_edges"

# Column definitions. The kind-specific fields ride inside the JSON `data`
# column so the schema is stable across node/edge kinds; the few columns we
# promote to first-class SQL are the ones the GUI / queries filter on.
_NODE_COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "kind": "TEXT NOT NULL",
    "user_id": "TEXT",
    "name": "TEXT",
    "text": "TEXT NOT NULL DEFAULT ''",
    "data": "TEXT NOT NULL DEFAULT '{}'",
    "created_at": "REAL",
    "last_recalled": "REAL",
    "recall_count": "INTEGER NOT NULL DEFAULT 0",
    "practice_times": "TEXT NOT NULL DEFAULT '[]'",
    "source": "TEXT NOT NULL DEFAULT ''",
}

_EDGE_COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "kind": "TEXT NOT NULL",
    "src": "TEXT NOT NULL",
    "dst": "TEXT NOT NULL",
    "weight": "REAL NOT NULL DEFAULT 0.5",
    "data": "TEXT NOT NULL DEFAULT '{}'",
}


def _ensure_tables(store: SQLiteStore) -> None:
    store.create_table(NODES_TABLE, _NODE_COLUMNS)
    store.create_table(EDGES_TABLE, _EDGE_COLUMNS)


def _user_id_of(node_dict: dict[str, Any]) -> Optional[str]:
    return node_dict.get("user_id") if isinstance(node_dict.get("user_id"), str) else None


def _name_of(node_dict: dict[str, Any]) -> Optional[str]:
    name = node_dict.get("name")
    return name if isinstance(name, str) and name else None


@contextmanager
def _graph_write_lock(path: str):
    """Serialize a complete SQLite + hybrid-index graph publication.

    ``HybridSearch.persist`` already locks its own two files, but that lock is
    too narrow for the graph: SQLite must be updated before ``nodes.json`` is
    published, and two processes must preserve that same ordering.  A
    graph-level lock prevents an older writer from publishing a stale hybrid
    snapshot after a newer SQLite merge has completed.
    """
    if not path:
        yield
        return
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ".graph.lock"), "a+") as lock_f:
        if fcntl is not None:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _upsert_sql(table: str, row: dict[str, Any]) -> tuple[str, list[Any]]:
    """Return a parameterised ``INSERT .. ON CONFLICT(id)`` statement."""
    columns = list(row)
    names = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column}=excluded.{column}" for column in columns if column != "id"
    )
    sql = (
        f"INSERT INTO {table} ({names}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}"
    )
    return sql, [row[column] for column in columns]


# --------------------------------------------------------------------- write
def save_graph(
    graph: KnowledgeGraph,
    store: SQLiteStore,
    hybrid: HybridSearch,
    path: str,
    *,
    replace: bool = False,
) -> KnowledgeGraph:
    """Merge ``graph`` into SQLite and publish a matching hybrid index.

    Routine extraction is deliberately additive.  The previous implementation
    deleted both KG tables before rewriting the caller's in-memory graph; a
    stale Discord/server worker could consequently erase every wiki row while
    persisting its newly learned user edge.  We now upsert the snapshot and
    apply only explicit graph tombstones.  A graph rebuilt from scratch uses
    ``replace=True`` to publish an authoritative replacement.

    The returned graph is reloaded from SQLite after the merge and is therefore
    the authoritative union when another process had added rows meanwhile.
    """
    # Materialise JSON rows before entering the cross-process lock.  This also
    # avoids iterating live dicts while an extraction thread adds a node/edge.
    nodes = list(graph.nodes.values())
    edges = list(graph.edges.values())
    removed_nodes = set(getattr(graph, "_removed_node_ids", set()))
    removed_edges = set(getattr(graph, "_removed_edge_ids", set()))
    node_rows: list[dict[str, Any]] = []
    for node in nodes:
        text = (node.text or "").strip()
        d = node.to_dict()
        node_rows.append(
            {
                "id": node.id,
                "kind": node.kind,
                "user_id": _user_id_of(d),
                "name": _name_of(d),
                "text": text,
                "data": json.dumps(d, ensure_ascii=False),
                "created_at": node.created_at,
                "last_recalled": node.last_recalled,
                "recall_count": int(node.recall_count),
                "practice_times": json.dumps(list(node.practice_times)),
                "source": node.source or "",
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for edge in edges:
        edge_rows.append(
            {
                "id": edge.id,
                "kind": edge.kind,
                "src": edge.src,
                "dst": edge.dst,
                "weight": float(edge.weight),
                "data": json.dumps(edge.to_dict(), ensure_ascii=False),
            }
        )

    with _graph_write_lock(path):
        _ensure_tables(store)
        with store.transaction(immediate=True) as conn:
            if replace:
                conn.execute(f"DELETE FROM {EDGES_TABLE}")
                conn.execute(f"DELETE FROM {NODES_TABLE}")
            else:
                # Node deletion also removes any durable edge touching it,
                # including an edge written by a process this snapshot had not
                # loaded yet.  This prevents dangling endpoints.
                for node_id in removed_nodes:
                    conn.execute(
                        f"DELETE FROM {EDGES_TABLE} WHERE src=? OR dst=?",
                        (node_id, node_id),
                    )
                    conn.execute(
                        f"DELETE FROM {NODES_TABLE} WHERE id=?", (node_id,)
                    )
                for edge_id in removed_edges:
                    conn.execute(
                        f"DELETE FROM {EDGES_TABLE} WHERE id=?", (edge_id,)
                    )
            for row in node_rows:
                sql, params = _upsert_sql(NODES_TABLE, row)
                conn.execute(sql, params)
            for row in edge_rows:
                sql, params = _upsert_sql(EDGES_TABLE, row)
                conn.execute(sql, params)

        # SQLite is now authoritative.  Re-read it while the graph publication
        # lock is held, then build/publish the exact same node set to the hybrid
        # index.  This keeps cross-process additions searchable immediately.
        durable = load_graph(store)
        chunks = [
            Chunk(
                text=(node.text or "").strip(),
                source=node.kind,
                metadata={"id": node.id, "kind": node.kind},
            )
            for node in durable.nodes.values()
            if (node.text or "").strip() and node.id != durable.SELF_ID
        ]
        hybrid.build(chunks)
        if path:
            hybrid.persist(path)
    return durable


# --------------------------------------------------------------------- read
def load_graph(store: SQLiteStore) -> KnowledgeGraph:
    """Reconstruct the graph from the `kg_nodes` / `kg_edges` tables."""
    _ensure_tables(store)
    data: dict[str, Any] = {"nodes": [], "edges": [], "counters": {}}
    for row in store.select(NODES_TABLE):
        try:
            node_data = json.loads(row.get("data") or "{}")
        except (TypeError, ValueError):
            continue
        # Defensive: keep the SQL columns authoritative for the common fields
        # that might have drifted from the JSON blob.
        node_data.setdefault("id", row["id"])
        node_data.setdefault("kind", row["kind"])
        node_data.setdefault("text", row.get("text", ""))
        node_data.setdefault("created_at", row.get("created_at") or 0.0)
        node_data.setdefault("last_recalled", row.get("last_recalled"))
        node_data.setdefault("recall_count", int(row.get("recall_count") or 0))
        try:
            node_data["practice_times"] = json.loads(row.get("practice_times") or "[]")
        except (TypeError, ValueError):
            node_data["practice_times"] = []
        node_data.setdefault("source", row.get("source") or "")
        data["nodes"].append(node_data)
    for row in store.select(EDGES_TABLE):
        try:
            edge_data = json.loads(row.get("data") or "{}")
        except (TypeError, ValueError):
            continue
        edge_data.setdefault("id", row["id"])
        edge_data.setdefault("kind", row["kind"])
        edge_data.setdefault("src", row["src"])
        edge_data.setdefault("dst", row["dst"])
        edge_data.setdefault("weight", float(row.get("weight") or 0.5))
        data["edges"].append(edge_data)
    # Recover counters from the highest assigned id per kind.
    for kind in ("fact", "episode", "entity"):
        max_n = 0
        for n in data["nodes"]:
            nid = n.get("id", "")
            if isinstance(nid, str) and nid.startswith(f"{kind}:"):
                try:
                    max_n = max(max_n, int(nid.split(":", 1)[1]))
                except (ValueError, IndexError):
                    pass
        data["counters"][kind] = max_n
    return KnowledgeGraph.from_dict(data)


def has_persisted(store: SQLiteStore, index_path: str) -> bool:
    """True when both the SQLite tables and the on-disk index exist."""
    cols = store.columns(NODES_TABLE)
    has_table = bool(cols)
    has_index = os.path.exists(os.path.join(index_path, "nodes.json"))
    return has_table and has_index


__all__ = [
    "NODES_TABLE",
    "EDGES_TABLE",
    "save_graph",
    "load_graph",
    "has_persisted",
]
