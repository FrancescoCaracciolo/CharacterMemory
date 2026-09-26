"""Extraction audit log: what each extraction pass changed.

Extraction writes into several independent memories at once (facts,
directives, episodes, the rolling user summary, emotion state, the knowledge
graph). Reading those memories afterwards shows *state*, not *change*: a
summary rewrite or a +0.1 trust nudge is invisible once applied. This module
records one row per extraction pass describing exactly what that pass did, so
callers (the GUI's Extractions panel) can show a readable changelog.

Two pieces:

* :class:`ExtractionRecorder` — snapshots the mutable state (user summaries,
  emotion, knowledge-graph node/edge ids) before a pass and diffs it against
  the state after extraction, dedup and graph ingestion have all run.
* :class:`ExtractionLog` — persists those reports in the shared store
  (``extraction_runs`` table), bounded to the most recent ``limit`` runs.

Both are duck-typed over the memories (``user_summary.get_summary``,
``emotion.get_current_mood``, ``knowledge_graph.retriever.graph``…) so a
custom memory set simply contributes whatever it supports. Recording never
raises into the extraction path; see ``CharacterAgent._extract_messages``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping, Optional

from ..knowledge_graph.edges import SYMMETRIC_KINDS
from .base import Memory, MemoryItem
from .store_base import Store

TABLE = "extraction_runs"
_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "chat_id": "TEXT",
    "user_id": "TEXT",
    "participants": "TEXT NOT NULL DEFAULT '[]'",
    "created_at": "REAL NOT NULL",
    "duration_ms": "REAL",
    "counts": "TEXT NOT NULL DEFAULT '{}'",
    "error": "TEXT",
    "preview": "TEXT",
    "report": "TEXT NOT NULL",
}

REPORT_VERSION = 1

# Memories whose change is reported as a delta rather than as a list of rows.
_DELTA_MEMORIES = {"user_summary", "emotion", "knowledge_graph"}

# Row bookkeeping that says nothing about *what* was learned.
_BOOKKEEPING_KEYS = {
    "id", "user_id", "created_at", "updated_at", "last_recalled", "recall_count",
    "chat_id", "source_message_ids", "embedding", "score", "raw_emotional_impact",
    "emotion_similarity", "impact", "similarity",
}

_MAX_MESSAGE_CHARS = 1200
_MAX_PREVIEW_CHARS = 160
_MAX_EXTRA_CHARS = 240
_MAX_NEIGHBORS = 24
_MAX_NEW_NODES = 200
_MAX_NEW_LINKS = 120
_EPS = 1e-9


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _incident(graph: Any, node_id: str) -> list[tuple[Any, Any]]:
    """Every ``(edge, other_node)`` touching ``node_id`` in either direction.

    ``KnowledgeGraph.neighbors`` walks directed edges from ``src`` only (the
    activation direction), which would hide e.g. the person a new fact is
    about. A changelog wants every link.
    """
    adj = getattr(graph, "_adj", None)
    if isinstance(adj, dict):
        edge_ids = list(dict.fromkeys(adj.get(node_id, [])))
    else:
        edge_ids = [e.id for e in graph.edges.values() if node_id in (e.src, e.dst)]
    out = []
    for eid in edge_ids:
        edge = graph.edges.get(eid)
        if edge is None:
            continue
        other = graph.nodes.get(edge.dst if edge.src == node_id else edge.src)
        if other is not None and other.id != node_id:
            out.append((edge, other))
    return out


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _preview(report: Mapping[str, Any]) -> Optional[str]:
    """The latest user turn of the pass, so a run list can say what it was about."""
    messages = report.get("messages") or []
    chosen = next(
        (m for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
        next((m for m in reversed(messages) if m.get("content")), None),
    )
    if chosen is None:
        return None
    text = " ".join(str(chosen["content"]).split())
    if len(text) > _MAX_PREVIEW_CHARS:
        text = text[: _MAX_PREVIEW_CHARS - 1].rstrip() + "…"
    return text


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class ExtractionLog:
    """Bounded, queryable history of extraction reports in the shared store."""

    def __init__(self, store: Store, *, limit: int = 200) -> None:
        self.store = store
        self.limit = max(0, int(limit))
        self.store.create_table(TABLE, _COLUMNS)
        if "preview" not in self.store.columns(TABLE):
            try:
                self.store.execute(f"ALTER TABLE {TABLE} ADD COLUMN preview TEXT")
            except self.store.operational_errors:
                if "preview" not in self.store.columns(TABLE):
                    raise

    def record(
        self,
        report: dict[str, Any],
        *,
        chat_id: Optional[str],
        user_id: Optional[str],
        participants: Iterable[str],
        created_at: Optional[float] = None,
        duration_ms: Optional[float] = None,
    ) -> int:
        counts = report_counts(report)
        rows = self.store.execute(
            f"INSERT INTO {TABLE} "
            "(chat_id, user_id, participants, created_at, duration_ms, counts, error, preview, report) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            [
                chat_id,
                user_id,
                _dumps(list(participants)),
                float(created_at if created_at is not None else time.time()),
                duration_ms,
                _dumps(counts),
                report.get("error"),
                _preview(report),
                _dumps(report),
            ],
        )
        run_id = int(rows[0]["id"])
        if self.limit:
            # Keep the newest ``limit`` runs; ids are monotonically increasing.
            self.store.execute(
                f"DELETE FROM {TABLE} WHERE id <= ?", [run_id - self.limit]
            )
        return run_id

    def _where(self, user_id: Optional[str]) -> tuple[str, list[Any]]:
        if not user_id:
            return "", []
        needle = _dumps(user_id)
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return (
            " WHERE (user_id = ? OR participants LIKE ? ESCAPE '\\')",
            [user_id, f"%{escaped}%"],
        )

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest-first run summaries (no full report) plus the total count."""
        where, params = self._where(user_id)
        total_rows = self.store.execute(
            f"SELECT COUNT(*) AS n FROM {TABLE}{where}", params
        )
        total = int(total_rows[0]["n"]) if total_rows else 0
        rows = self.store.execute(
            f"SELECT id, chat_id, user_id, participants, created_at, duration_ms, counts, error, preview "
            f"FROM {TABLE}{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, max(1, int(limit)), max(0, int(offset))],
        )
        return [self._summary(row) for row in rows], total

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        rows = self.store.execute(f"SELECT * FROM {TABLE} WHERE id = ?", [int(run_id)])
        if not rows:
            return None
        row = rows[0]
        out = self._summary(row)
        out["report"] = _loads(row.get("report"), {})
        return out

    def users(self) -> list[str]:
        """Every speaker that appears in a recorded run."""
        seen: dict[str, None] = {}
        for row in self.store.execute(f"SELECT user_id, participants FROM {TABLE}"):
            for uid in [row.get("user_id"), *_loads(row.get("participants"), [])]:
                if uid:
                    seen.setdefault(str(uid), None)
        return sorted(seen)

    @staticmethod
    def _summary(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "chat_id": row.get("chat_id"),
            "user_id": row.get("user_id"),
            "participants": _loads(row.get("participants"), []),
            "created_at": _float(row.get("created_at")),
            "duration_ms": _float(row.get("duration_ms")),
            "counts": _loads(row.get("counts"), {}),
            "error": row.get("error"),
            "preview": row.get("preview"),
        }


def report_counts(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-section tallies used by the run list."""
    memories = {}
    for name, block in (report.get("memories") or {}).items():
        items = block.get("items") or []
        memories[name] = {
            "new": sum(1 for it in items if it.get("status") == "new"),
            "folded": sum(1 for it in items if it.get("status") != "new"),
            "updated": len(block.get("updated") or []),
        }
    emotion = report.get("emotion") or {}
    kg = report.get("knowledge_graph") or {}
    counts = {
        "memories": memories,
        "new_items": sum(m["new"] for m in memories.values()),
        "summaries": len(report.get("summaries") or []),
        "emotion_users": len(emotion.get("users") or []),
        "mood_changed": bool(emotion.get("mood_changed")),
        "kg_nodes": len(kg.get("new_nodes") or []),
        "kg_links": int(kg.get("new_edge_count") or 0),
        "messages": len(report.get("messages") or []),
    }
    counts["total"] = (
        counts["new_items"]
        + sum(m["updated"] for m in memories.values())
        + counts["summaries"]
        + counts["emotion_users"]
        + (1 if counts["mood_changed"] else 0)
        + counts["kg_nodes"]
    )
    return counts


# --------------------------------------------------------------------------- #
# Before/after diff
# --------------------------------------------------------------------------- #
class ExtractionRecorder:
    """Capture state around one extraction pass and describe the change.

    Call :meth:`capture_before` before any memory is written, run the whole
    pass (extract → graph ingest → dedup → persist), then call
    :meth:`build_report` with the extractor's raw output, the ``added`` map
    and the dedup reports.
    """

    def __init__(
        self,
        memories: Mapping[str, Memory],
        *,
        user_id: str,
        participants: Optional[list[str]] = None,
        character_name: str = "",
    ) -> None:
        self.memories = memories
        self.user_id = user_id
        self.users = list(dict.fromkeys(participants or [user_id]))
        self.character_name = character_name
        self._summaries: dict[str, Optional[dict[str, Any]]] = {}
        self._mood: Optional[dict[str, float]] = None
        self._emotion: dict[str, dict[str, Any]] = {}
        self._kg_nodes: Optional[set[str]] = None
        self._kg_edges: Optional[set[str]] = None
        self.started = time.time()

    # ---- snapshot ---------------------------------------------------------
    def capture_before(self) -> "ExtractionRecorder":
        self.started = time.time()
        self._summaries = {uid: self._summary(uid) for uid in self.users}
        emotion = self._enabled("emotion")
        if emotion is not None and hasattr(emotion, "get_current_mood"):
            self._mood = dict(emotion.get_current_mood())
            self._emotion = {uid: self._emotion_state(emotion, uid) for uid in self.users}
        graph = self._graph()
        if graph is not None:
            self._kg_nodes = set(graph.nodes)
            self._kg_edges = set(graph.edges)
        return self

    def _enabled(self, name: str) -> Optional[Memory]:
        mem = self.memories.get(name)
        return mem if mem is not None and getattr(mem, "enabled", True) else None

    def _graph(self):
        kg = self._enabled("knowledge_graph")
        retriever = getattr(kg, "retriever", None)
        return getattr(retriever, "graph", None)

    def _summary(self, uid: str) -> Optional[dict[str, Any]]:
        mem = self._enabled("user_summary")
        if mem is None or not hasattr(mem, "get_summary"):
            return None
        row = mem.get_summary(uid)
        if row is None:
            return None
        parse = getattr(mem, "_parse_aliases", None)
        aliases = parse(row.get("aliases")) if callable(parse) else _loads(row.get("aliases"), [])
        return {
            "name": str(row.get("name") or uid),
            "aliases": list(aliases or []),
            "summary": str(row.get("summary") or ""),
        }

    @staticmethod
    def _emotion_state(emotion: Any, uid: str) -> dict[str, Any]:
        return {
            "dims": dict(emotion.get_user_state(uid)),
            "comment": str(emotion.get_user_comment(uid) or ""),
        }

    # ---- report -----------------------------------------------------------
    def build_report(
        self,
        *,
        rows: list[dict[str, Any]],
        raw: Optional[Mapping[str, Any]],
        added: Mapping[str, list[MemoryItem]],
        dedup_reports: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        dedup_reports = dedup_reports or {}
        report: dict[str, Any] = {
            "version": REPORT_VERSION,
            "character": self.character_name,
            "user_id": self.user_id,
            "participants": self.users,
            "messages": self._messages(rows),
            "memories": self._memories(added, dedup_reports),
            "summaries": self._summary_changes(),
            "emotion": self._emotion_changes(),
            "knowledge_graph": self._graph_changes(),
            "error": None,
        }
        if raw is not None and raw.get("error"):
            report["error"] = str(raw.get("error"))
        return report

    def _messages(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            content = str(r.get("content") or "")
            role = str(r.get("role") or "")
            speaker = (
                str(r.get("user_id") or self.user_id)
                if role == "user"
                else (self.character_name or role)
            )
            out.append({
                "id": r.get("id"),
                "role": role,
                "speaker": speaker,
                "content": content[:_MAX_MESSAGE_CHARS],
                "truncated": len(content) > _MAX_MESSAGE_CHARS,
                "occurred_at": _float(r.get("occurred_at")) or _float(r.get("created_at")),
            })
        return out

    def _memories(
        self,
        added: Mapping[str, list[MemoryItem]],
        dedup_reports: Mapping[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, items in added.items():
            if name in _DELTA_MEMORIES or not items:
                continue
            mem = self.memories.get(name)
            rep = dedup_reports.get(name)
            removed = {int(i) for i in (getattr(rep, "removed_ids", None) or [])}
            decisions = {
                int(d["incoming_id"]): d
                for d in (getattr(rep, "decisions", None) or [])
                if isinstance(d, dict) and d.get("incoming_id") is not None
            }
            entries = [self._item(mem, it, removed, decisions) for it in items]
            added_ids = {e["id"] for e in entries if e.get("id") is not None}
            updated = []
            for rid in getattr(rep, "updated_ids", None) or []:
                rid = int(rid)
                if rid in added_ids:
                    continue
                row = self._row(mem, rid)
                if row is not None:
                    updated.append(self._row_entry(mem, row, fallback_text=""))
            out[name] = {"items": entries, "updated": updated}
        return out

    @staticmethod
    def _row(mem: Any, row_id: Any) -> Optional[dict[str, Any]]:
        getter = getattr(mem, "get_row", None)
        if row_id is None or not callable(getter):
            return None
        try:
            return getter(int(row_id))
        except (TypeError, ValueError):
            return None

    def _item(
        self,
        mem: Any,
        item: MemoryItem,
        removed: set[int],
        decisions: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        meta = dict(item.metadata or {})
        row_id = meta.get("id")
        try:
            row_id = int(row_id) if row_id is not None else None
        except (TypeError, ValueError):
            row_id = None
        current = self._row(mem, row_id)
        entry = self._row_entry(mem, current or meta, fallback_text=item.text)
        entry["id"] = row_id
        decision = decisions.get(row_id) if row_id is not None else None
        if current is not None or (row_id is None and not decision):
            entry["status"] = "new"
        elif decision and decision.get("survivor_id") is not None:
            entry["status"] = "merged" if decision.get("action") in ("merge", "revise", "resolve") else "duplicate"
            entry["survivor_id"] = int(decision["survivor_id"])
            entry["reason"] = decision.get("reason")
            survivor = self._row(mem, decision["survivor_id"])
            if survivor is not None:
                entry["survivor_text"] = self._row_text(mem, survivor, "")
        else:
            entry["status"] = "duplicate" if row_id in removed else "removed"
        return entry

    @staticmethod
    def _row_text(mem: Any, row: Mapping[str, Any], fallback: str) -> str:
        col = getattr(mem, "text_column", None)
        if col and row.get(col):
            return str(row.get(col))
        return fallback or str(row.get("content") or row.get("summary") or row.get("text") or "")

    def _row_entry(
        self, mem: Any, row: Mapping[str, Any], *, fallback_text: str
    ) -> dict[str, Any]:
        text_col = getattr(mem, "text_column", None)
        extra: dict[str, Any] = {}
        for key, value in row.items():
            if key in _BOOKKEEPING_KEYS or key == text_col or key == "importance":
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                extra[key] = value
            elif isinstance(value, str):
                if not value or len(value) > _MAX_EXTRA_CHARS:
                    continue
                parsed = _loads(value, None) if value[:1] in "{[" else None
                if isinstance(parsed, dict):
                    if parsed and all(isinstance(v, (int, float)) for v in parsed.values()):
                        extra[key] = parsed
                elif isinstance(parsed, list):
                    if parsed:
                        extra[key] = parsed
                else:
                    extra[key] = value
            elif isinstance(value, dict) and value and all(
                isinstance(v, (int, float)) for v in value.values()
            ):
                extra[key] = dict(value)
        return {
            "id": row.get("id"),
            "user_id": row.get("user_id"),
            "text": self._row_text(mem, row, fallback_text),
            "importance": _float(row.get("importance")),
            "extra": extra,
        }

    def _summary_changes(self) -> list[dict[str, Any]]:
        changes = []
        for uid in self.users:
            before = self._summaries.get(uid)
            after = self._summary(uid)
            if after is None or after == before:
                continue
            aliases_before = set((before or {}).get("aliases") or [])
            changes.append({
                "user_id": uid,
                "before": before,
                "after": after,
                "aliases_added": [a for a in after["aliases"] if a not in aliases_before],
            })
        return changes

    def _emotion_changes(self) -> dict[str, Any]:
        emotion = self._enabled("emotion")
        if emotion is None or self._mood is None:
            return {"enabled": False, "mood": [], "mood_changed": False, "users": []}
        after_mood = dict(emotion.get_current_mood())
        axes = list(dict.fromkeys([*self._mood, *after_mood]))
        mood = []
        for axis in axes:
            b = float(self._mood.get(axis, 0.0))
            a = float(after_mood.get(axis, 0.0))
            mood.append({"axis": axis, "before": b, "after": a, "delta": a - b})
        users = []
        for uid in self.users:
            before = self._emotion.get(uid) or {"dims": {}, "comment": ""}
            after = self._emotion_state(emotion, uid)
            dims = []
            for dim in dict.fromkeys([*before["dims"], *after["dims"]]):
                b = float(before["dims"].get(dim, 0.0))
                a = float(after["dims"].get(dim, 0.0))
                dims.append({"dim": dim, "before": b, "after": a, "delta": a - b})
            comment_changed = before["comment"] != after["comment"]
            if comment_changed or any(abs(d["delta"]) > _EPS for d in dims):
                users.append({
                    "user_id": uid,
                    "dims": dims,
                    "comment_before": before["comment"],
                    "comment_after": after["comment"],
                    "comment_changed": comment_changed,
                })
        return {
            "enabled": True,
            "mood": mood,
            "mood_changed": any(abs(m["delta"]) > _EPS for m in mood),
            "users": users,
        }

    # ---- knowledge graph --------------------------------------------------
    def _node_label(self, node: Any) -> str:
        if node.kind == "self":
            return self.character_name or node.text or "Self"
        for attr in ("name", "content", "summary", "user_id"):
            value = getattr(node, attr, None)
            if value:
                return str(value)
        return str(node.text or node.id)

    def _node_view(self, node: Any) -> dict[str, Any]:
        view = {
            "id": node.id,
            "kind": node.kind,
            "label": self._node_label(node),
            "text": str(node.text or ""),
        }
        for attr in ("kind_label", "type", "user_id", "importance", "confidence"):
            value = getattr(node, attr, None)
            if value not in (None, ""):
                view[attr] = value
        return view

    def _graph_changes(self) -> dict[str, Any]:
        graph = self._graph()
        if graph is None or self._kg_nodes is None or self._kg_edges is None:
            return {"enabled": False, "new_nodes": [], "new_links": [], "new_edge_count": 0, "removed_nodes": 0}
        nodes = graph.nodes
        new_ids = [
            nid for nid in nodes
            if nid not in self._kg_nodes and not getattr(nodes[nid], "internal", False)
        ]
        new_set = set(new_ids)
        new_edges = [eid for eid in graph.edges if eid not in self._kg_edges]
        new_edge_set = set(new_edges)
        structural_first = lambda pair: (  # noqa: E731
            pair[0].kind in ("co_occurrence", "chat"),
            -abs(float(pair[0].weight or 0.0)),
        )
        new_nodes = []
        for nid in new_ids[:_MAX_NEW_NODES]:
            node = nodes[nid]
            view = self._node_view(node)
            pairs = [
                (edge, other) for edge, other in _incident(graph, nid)
                if not getattr(other, "internal", False)
            ]
            pairs.sort(key=structural_first)
            view["neighbors"] = [
                {
                    **self._node_view(other),
                    "edge_kind": edge.kind,
                    "edge_weight": float(edge.weight or 0.0),
                    "direction": (
                        "both" if edge.kind in SYMMETRIC_KINDS
                        else "out" if edge.src == nid else "in"
                    ),
                    "is_new": other.id in new_set,
                    "new_edge": edge.id in new_edge_set,
                }
                for edge, other in pairs[:_MAX_NEIGHBORS]
            ]
            view["neighbor_total"] = len(pairs)
            new_nodes.append(view)
        # Links added between nodes that already existed (e.g. a new relation
        # edge) are otherwise invisible from the node list.
        new_links = []
        for eid in new_edges:
            edge = graph.edges.get(eid)
            if edge is None or edge.src in new_set or edge.dst in new_set:
                continue
            src, dst = nodes.get(edge.src), nodes.get(edge.dst)
            if src is None or dst is None or getattr(src, "internal", False) or getattr(dst, "internal", False):
                continue
            new_links.append({
                "kind": edge.kind,
                "weight": float(edge.weight or 0.0),
                "src": self._node_view(src),
                "dst": self._node_view(dst),
            })
            if len(new_links) >= _MAX_NEW_LINKS:
                break
        return {
            "enabled": True,
            "new_nodes": new_nodes,
            "new_nodes_total": len(new_ids),
            "new_links": new_links,
            "new_edge_count": len(new_edge_set),
            "removed_nodes": len(self._kg_nodes - set(nodes)),
        }


__all__ = ["ExtractionLog", "ExtractionRecorder", "report_counts", "TABLE"]
