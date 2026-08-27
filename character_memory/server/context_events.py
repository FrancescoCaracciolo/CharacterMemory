"""Live `/context` observability events for the bundled web UI.

The broker is deliberately process-local.  The documented server deployment is
single-worker and the monitor is an in-memory observatory, not a durable chat
log or an inter-process event bus.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

from character_memory.character import ContextSnapshot
from character_memory.memory.base import MemoryItem

from .adapters import read_graph


def _json_safe(value: Any) -> Any:
    """Convert plugin metadata into a JSON-safe, bounded-data shape."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def item_payload(item: MemoryItem) -> dict[str, Any]:
    score = float(item.score or 0.0)
    return {
        "text": str(item.text or ""),
        "score": _json_safe(score),
        "kind": str(item.kind or ""),
        "metadata": _json_safe(item.metadata or {}),
    }


def _query_payload(query: Any) -> list[dict[str, Any]]:
    if isinstance(query, str):
        return [{"text": query, "weight": 1.0}]
    return [
        {"text": str(text), "weight": _json_safe(float(weight))}
        for text, weight in (query or [])
        if str(text).strip()
    ]


def _graph_for_diagnostic(
    agent: Any,
    diagnostic: dict[str, Any],
    *,
    user: str,
    query_text: str,
) -> Optional[dict[str, Any]]:
    activation = diagnostic.get("activation")
    if not isinstance(activation, dict):
        return None
    try:
        graph = read_graph(
            agent,
            q=query_text,
            user=user,
            limit=80,
            hops_subgraph=1,
            max_edges=400,
            precomputed_trace={str(k): float(v) for k, v in activation.items()},
            precomputed_retrieved={str(v) for v in diagnostic.get("retrieved_ids", [])},
            precomputed_breakdowns={
                str(k): v
                for k, v in (diagnostic.get("breakdowns") or {}).items()
                if isinstance(v, dict)
            },
        )
        return _json_safe(graph)
    except Exception:
        # A live diagnostic must never make a successful context request fail.
        return None


def build_context_event(
    agent: Any,
    *,
    snapshot: ContextSnapshot,
    character: str,
    user: str,
    message: str,
    chat_id: str,
) -> dict[str, Any]:
    """Turn a single-pass context snapshot into the browser event payload."""
    recalls: list[dict[str, Any]] = []
    for name, section in snapshot.sections.items():
        recall = snapshot.recalls.get(name)
        if recall is None:
            # User-authored prompt blocks have no retrieval diagnostics, but
            # keeping them in this ordered list lets the live monitor show the
            # exact /context sequence rather than silently hiding them.
            recalls.append(
                {
                    "name": name,
                    "title": "Intermediate prompt",
                    "scope": "prompt",
                    "body": section,
                    "section": section,
                    "items": [],
                }
            )
            continue
        recalls.append(
            {
                "name": recall.name,
                "title": recall.title,
                "scope": recall.scope,
                "body": recall.body,
                "section": recall.section,
                "items": [item_payload(item) for item in recall.items],
            }
        )

    query_items = _query_payload(snapshot.query)
    query_text = query_items[0]["text"] if query_items else ""
    graphs: dict[str, dict[str, Any]] = {}
    kg = snapshot.recalls.get("knowledge_graph")
    if kg is not None:
        diagnostics = kg.diagnostics or {}
        if isinstance(diagnostics.get("participants"), dict):
            for participant, diagnostic in diagnostics["participants"].items():
                if isinstance(diagnostic, dict):
                    graph = _graph_for_diagnostic(
                        agent, diagnostic, user=str(participant), query_text=query_text
                    )
                    if graph is not None:
                        graphs[str(participant)] = graph
        else:
            graph = _graph_for_diagnostic(
                agent, diagnostics, user=snapshot.user_id, query_text=query_text
            )
            if graph is not None:
                graphs[snapshot.user_id] = graph

    return {
        "created_at": time.time(),
        "character": character,
        "chat_id": chat_id,
        "user": user,
        "message": message,
        "participants": list(snapshot.participants),
        "query": query_items,
        "memories": recalls,
        # Keep the assembled response sections available as a small, stable
        # fallback for older clients; the structured ``memories`` list above
        # remains the source for exact item rows and diagnostics.
        "context": dict(snapshot.sections),
        "context_order": list(snapshot.sections),
        "context_text": "\n\n".join(snapshot.sections.values()),
        "memory_count": len(snapshot.recalls),
        "item_count": sum(len(r["items"]) for r in recalls),
        "graphs": graphs,
    }


class ContextEventBroker:
    """Thread-safe bounded event history plus SSE subscribers."""

    def __init__(self, max_events: int = 25, subscriber_queue_size: int = 64) -> None:
        self.max_events = max(1, int(max_events))
        self.subscriber_queue_size = max(4, int(subscriber_queue_size))
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_events)
        )
        self._sequences: dict[str, int] = defaultdict(int)
        # Keep the optional speaker filter with each queue so unrelated
        # context payloads never enter that subscriber's replay/live stream.
        self._subscribers: dict[
            str, dict[queue.Queue[dict[str, Any]], Optional[str]]
        ] = defaultdict(dict)
        self._lock = threading.RLock()

    def publish(self, character: str, event: dict[str, Any]) -> dict[str, Any]:
        """Assign a per-character sequence and fan out without blocking."""
        with self._lock:
            self._sequences[character] += 1
            enriched = dict(event)
            enriched["sequence"] = self._sequences[character]
            enriched["id"] = str(enriched["sequence"])
            self._history[character].append(enriched)
            for subscriber, user_filter in list(
                self._subscribers.get(character, {}).items()
            ):
                if user_filter is not None and str(enriched.get("user") or "") != user_filter:
                    continue
                try:
                    try:
                        subscriber.put_nowait(enriched)
                    except queue.Full:
                        # A browser that cannot keep up should see the newest
                        # request, not hold up the producer or other clients.
                        try:
                            subscriber.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            subscriber.put_nowait(enriched)
                        except queue.Full:
                            pass
                except Exception:
                    # A malformed/disconnected subscriber is not allowed to
                    # affect this publish or any other browser.
                    self._subscribers[character].pop(subscriber, None)
            return enriched

    def subscribe(
        self,
        character: str,
        last_event_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], queue.Queue[dict[str, Any]], Any]:
        """Return replayable history, a queue, and an unsubscribe callback.

        ``user_id`` filters by the event's current speaker. Sequence numbers
        remain global per character, so filtered clients can legitimately see
        gaps when other users generate requests.
        """
        try:
            after = int(last_event_id or 0)
        except (TypeError, ValueError):
            after = 0
        # Preserve the supplied value exactly: speaker matching is
        # case-sensitive and does not normalize whitespace.
        user_filter = str(user_id) if user_id not in (None, "") else None
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=self.subscriber_queue_size
        )
        with self._lock:
            replay = [
                event
                for event in self._history.get(character, ())
                if int(event.get("sequence", 0)) > after
                and (
                    user_filter is None
                    or str(event.get("user") or "") == user_filter
                )
            ]
            self._subscribers[character][subscriber] = user_filter

        def unsubscribe() -> None:
            with self._lock:
                subscribers = self._subscribers.get(character)
                if subscribers is None:
                    return
                subscribers.pop(subscriber, None)
                if not subscribers:
                    self._subscribers.pop(character, None)

        return replay, subscriber, unsubscribe

    def history(self, character: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history.get(character, ()))


def project_context_event(
    event: dict[str, Any], user_id: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Project one event for an exact-speaker iframe subscription.

    The graph snapshot was already privacy-scoped during context assembly.
    For a filtered stream, retain only the selected speaker's graph entry so a
    group event cannot expose alternate participant-labelled snapshots.
    """
    user_filter = str(user_id) if user_id not in (None, "") else None
    if user_filter is None:
        return event
    if str(event.get("user") or "") != user_filter:
        return None
    projected = dict(event)
    graphs = event.get("graphs")
    # A normal graph payload has list-valued ``nodes``/``edges``. The live
    # event shape is an outer user->graph mapping; checking the value type
    # keeps a perfectly valid speaker id such as ``"nodes"`` unambiguous.
    direct_graph = (
        isinstance(graphs, dict)
        and isinstance(graphs.get("nodes"), list)
        and isinstance(graphs.get("edges"), list)
    )
    if isinstance(graphs, dict) and not direct_graph:
        selected = graphs.get(user_filter)
        projected["graphs"] = {user_filter: selected} if selected is not None else {}
    return projected


def sse_event(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event['id']}\nevent: context\ndata: {payload}\n\n"


__all__ = [
    "ContextEventBroker",
    "build_context_event",
    "project_context_event",
    "item_payload",
    "sse_event",
]
