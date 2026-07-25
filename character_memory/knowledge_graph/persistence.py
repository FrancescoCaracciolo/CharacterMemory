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
from typing import Any, Optional

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


# --------------------------------------------------------------------- write
def save_graph(
    graph: KnowledgeGraph,
    store: SQLiteStore,
    hybrid: HybridSearch,
    path: str,
) -> None:
    """Persist `graph` to SQLite + the hybrid index directory `path`."""
    _ensure_tables(store)
    store.execute(f"DELETE FROM {NODES_TABLE}")
    store.execute(f"DELETE FROM {EDGES_TABLE}")

    # Persist every node (including SelfNode, whose baseline/current mood is
    # character state). Build hybrid-index chunks in the same pass, omitting
    # only nodes with no searchable text; SelfNode is always seeded at
    # retrieval time and contributes nothing to lexical/dense matching.
    chunks: list[Chunk] = []
    for node in graph.nodes.values():
        text = (node.text or "").strip()
        d = node.to_dict()
        store.upsert(
            NODES_TABLE,
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
            },
        )
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source=node.kind,
                    metadata={"id": node.id, "kind": node.kind},
                )
            )

    for edge in graph.edges.values():
        store.upsert(
            EDGES_TABLE,
            {
                "id": edge.id,
                "kind": edge.kind,
                "src": edge.src,
                "dst": edge.dst,
                "weight": float(edge.weight),
                "data": json.dumps(edge.to_dict(), ensure_ascii=False),
            },
        )

    # Rebuild the node-text index from scratch on every save. Node sets are
    # modest (hundreds, not millions) and a stale index is worse than a
    # cheap rebuild.
    hybrid.build(chunks)
    if path:
        os.makedirs(path, exist_ok=True)
        hybrid.persist(path)


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
