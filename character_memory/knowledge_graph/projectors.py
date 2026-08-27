"""Deterministic source-memory projections into the knowledge graph.

Projectors are intentionally no-LLM adapters.  They read an authoritative
memory, reconcile the graph nodes owned by that source, and never write back.
Third-party memories can register another projector without changing the
retriever or :class:`CharacterAgent`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..config import KnowledgeGraphConfig
from .edges import EpisodeEdge, FactEdge
from .graph import KnowledgeGraph, slugify
from .nodes import EntityNode, EpisodeNode, FactNode, Node, PersonNode


@dataclass
class ProjectionResult:
    """Mutation counts returned by one source projector."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def merge(self, other: "ProjectionResult") -> "ProjectionResult":
        self.added += other.added
        self.updated += other.updated
        self.removed += other.removed
        self.skipped += other.skipped
        return self

    def to_dict(self) -> dict[str, int]:
        return {
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "skipped": self.skipped,
        }


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clip(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _put_node(graph: KnowledgeGraph, desired: Node, result: ProjectionResult) -> Node:
    """Insert/update a projected node while retaining its recall telemetry."""
    current = graph.nodes.get(desired.id)
    if current is None:
        graph.add_node(desired)
        result.added += 1
        return desired
    if type(current) is not type(desired):
        graph.remove_node(current.id)
        graph.add_node(desired)
        result.removed += 1
        result.added += 1
        return desired

    before = current.to_dict()
    retained = {
        "last_recalled": current.last_recalled,
        "recall_count": current.recall_count,
        "practice_times": list(current.practice_times),
        "activation": current.activation,
        "external_refs": list(current.external_refs),
        "memory_owners": list(current.memory_owners),
        "character_scoped": bool(current.character_scoped),
        "privacy_scope_version": int(current.privacy_scope_version or 0),
    }
    for key, value in desired.__dict__.items():
        if key not in retained:
            setattr(current, key, value)
    current.last_recalled = retained["last_recalled"]
    current.recall_count = retained["recall_count"]
    current.practice_times = retained["practice_times"] or list(desired.practice_times)
    current.activation = retained["activation"]
    current.external_refs = list(dict.fromkeys([
        *retained["external_refs"], *desired.external_refs
    ]))
    current.memory_owners = list(dict.fromkeys([
        *retained["memory_owners"], *desired.memory_owners
    ]))
    current.character_scoped = bool(
        retained["character_scoped"] or desired.character_scoped
    )
    current.privacy_scope_version = max(
        retained["privacy_scope_version"], desired.privacy_scope_version
    )
    if current.to_dict() != before:
        result.updated += 1
    return current


def _clear_structural_edges(graph: KnowledgeGraph, node_id: str) -> None:
    """Refresh projector-owned attribution without erasing learned links."""
    for edge_id in list(graph._adj.get(node_id, [])):
        edge = graph.edges.get(edge_id)
        if edge is not None and edge.kind in {"fact", "episode"}:
            graph.remove_edge(edge_id)


def _remove_stale_owned(
    graph: KnowledgeGraph, source_prefixes: Iterable[str], desired_ids: set[str]
) -> int:
    prefixes = tuple(source_prefixes)
    removed = 0
    for node in list(graph.nodes.values()):
        if node.id not in desired_ids and node.source.startswith(prefixes):
            graph.remove_node(node.id)
            removed += 1
    return removed


def _label_matches(text: str, labels: Iterable[str]) -> bool:
    folded = text.casefold()
    for label in labels:
        value = str(label or "").strip()
        if len(value) < 3:
            continue
        if re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", folded):
            return True
    return False


def _mentioned_nodes(graph: KnowledgeGraph, text: str) -> list[Node]:
    matches: list[Node] = []
    for node in graph.nodes.values():
        if isinstance(node, PersonNode):
            labels = [node.name, node.user_id, *(node.aliases or [])]
        elif isinstance(node, EntityNode):
            labels = [node.name, *(node.aliases or [])]
        else:
            continue
        if _label_matches(text, labels):
            matches.append(node)
    return matches


class GraphSourceProjector(ABC):
    """Extension seam for projecting one memory into graph semantics."""

    name: str
    source_memory: str
    version: int = 1

    @abstractmethod
    def fingerprint(
        self, memory: Optional[Any], config: KnowledgeGraphConfig
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def reconcile(
        self,
        graph: KnowledgeGraph,
        memory: Optional[Any],
        config: KnowledgeGraphConfig,
        *,
        character: Optional[dict[str, Any]] = None,
    ) -> ProjectionResult:
        raise NotImplementedError


_PROJECTORS: dict[str, GraphSourceProjector] = {}


def register_graph_source_projector(
    projector: GraphSourceProjector, *, replace: bool = False
) -> GraphSourceProjector:
    """Register a projector by name and return it."""
    if not projector.name:
        raise ValueError("A graph source projector needs a name")
    if projector.name in _PROJECTORS and not replace:
        raise ValueError(f"Graph source projector {projector.name!r} is already registered")
    _PROJECTORS[projector.name] = projector
    return projector


def get_graph_source_projector(name: str) -> GraphSourceProjector:
    try:
        return _PROJECTORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown graph source projector {name!r}") from exc


def graph_source_projectors() -> list[GraphSourceProjector]:
    return list(_PROJECTORS.values())


class HeartbeatGraphProjector(GraphSourceProjector):
    name = "heartbeat"
    source_memory = "heartbeat"
    version = 1

    @staticmethod
    def _enabled(memory: Optional[Any], config: KnowledgeGraphConfig) -> bool:
        return bool(
            config.project_heartbeat
            and memory is not None
            and getattr(memory, "enabled", False)
        )

    def fingerprint(self, memory: Optional[Any], config: KnowledgeGraphConfig) -> str:
        rows = (
            [
                {
                    key: row.get(key)
                    for key in ("id", "importance", "created_at", "summary", "kind")
                }
                for row in memory.all_rows()
            ]
            if self._enabled(memory, config)
            else []
        )
        return _fingerprint({
            "enabled": self._enabled(memory, config),
            "min_importance": config.heartbeat_min_importance,
            "max_nodes": config.heartbeat_max_nodes,
            "rows": rows,
        })

    def reconcile(
        self,
        graph: KnowledgeGraph,
        memory: Optional[Any],
        config: KnowledgeGraphConfig,
        *,
        character: Optional[dict[str, Any]] = None,
    ) -> ProjectionResult:
        result = ProjectionResult()
        desired: set[str] = set()
        rows = memory.all_rows() if self._enabled(memory, config) else []
        admitted: list[dict[str, Any]] = []
        for row in rows:
            kind = str(row.get("kind") or "").strip().lower()
            if kind not in {"discovery", "action"}:
                result.skipped += 1
                continue
            if _clip(row.get("importance")) < float(config.heartbeat_min_importance):
                result.skipped += 1
                continue
            admitted.append(row)
        admitted.sort(
            key=lambda row: (
                float(row.get("importance") or 0.0),
                float(row.get("created_at") or 0.0),
                int(row.get("id") or 0),
            ),
            reverse=True,
        )
        cap = max(0, int(config.heartbeat_max_nodes))
        if len(admitted) > cap:
            result.skipped += len(admitted) - cap
            admitted = admitted[:cap]

        graph.ensure_self()
        for row in admitted:
            row_id = int(row["id"])
            kind = str(row["kind"]).lower()
            summary = str(row.get("summary") or "").strip()
            if not summary:
                result.skipped += 1
                continue
            created = float(row.get("created_at") or time.time())
            importance = _clip(row.get("importance"))
            source = f"heartbeat:{row_id}"
            if kind == "discovery":
                node_id = f"fact:heartbeat:{row_id}"
                node: Node = FactNode(
                    id=node_id,
                    text=f"Heartbeat finding: {summary}",
                    content=summary,
                    type="heartbeat",
                    confidence=0.5,
                    importance=importance,
                    created_at=created,
                    practice_times=[created],
                    source=source,
                )
            else:
                node_id = f"episode:heartbeat:{row_id}"
                node = EpisodeNode(
                    id=node_id,
                    text=f"Heartbeat action: {summary}",
                    summary=summary,
                    importance=importance,
                    timestamp=created,
                    participants=[graph.SELF_ID],
                    created_at=created,
                    practice_times=[created],
                    source=source,
                )
            desired.add(node_id)
            stored = _put_node(graph, node, result)
            graph.mark_character_scope(stored)
            _clear_structural_edges(graph, node_id)
            endpoints = [graph.nodes[graph.SELF_ID], *_mentioned_nodes(graph, summary)]
            seen: set[str] = set()
            for endpoint in endpoints:
                if endpoint.id in seen or endpoint.id == node_id:
                    continue
                seen.add(endpoint.id)
                graph.mark_character_scope(endpoint)
                if kind == "discovery":
                    graph.upsert_edge(FactEdge(
                        src=endpoint.id,
                        dst=stored.id,
                        weight=max(0.3, 0.3 + 0.5 * importance),
                        confidence=0.5,
                        importance=importance,
                        timestamp=created,
                    ))
                else:
                    graph.upsert_edge(EpisodeEdge(
                        src=endpoint.id,
                        dst=stored.id,
                        weight=max(0.3, 0.3 + 0.5 * importance),
                        timestamp=created,
                        importance=importance,
                    ))

        result.removed += _remove_stale_owned(graph, ("heartbeat:",), desired)
        return result


class WorldGraphProjector(GraphSourceProjector):
    name = "world"
    source_memory = "world"
    # v2 removes authored routines from the durable graph projection. Routines
    # remain available through WorldMemory (and CalendarMemory when enabled),
    # but are not knowledge in their own right.
    version = 2

    @staticmethod
    def _enabled(memory: Optional[Any], config: KnowledgeGraphConfig) -> bool:
        return bool(
            config.project_world
            and memory is not None
            and getattr(memory, "enabled", False)
        )

    def fingerprint(self, memory: Optional[Any], config: KnowledgeGraphConfig) -> str:
        enabled = self._enabled(memory, config)
        if not enabled:
            return _fingerprint({"enabled": False})
        seed = memory.seed_view()
        return _fingerprint({
            "enabled": True,
            "max_events": config.world_event_max_nodes,
            "include_simulation": config.world_include_simulation_events,
            # Only authored data that is actually projected belongs in this
            # fingerprint. In particular, routine edits must not cause a KG
            # reconciliation because routines are intentionally excluded.
            "seed": {
                "observer_id": seed.get("observer_id"),
                "locations": seed.get("locations", []),
                "actors": seed.get("actors", []),
                "facts": seed.get("facts", []),
            },
            # Recall counters are retrieval telemetry, not projection source
            # data; excluding them prevents every ordinary recall from
            # invalidating the graph fingerprint.
            "records": [
                {
                    key: row.get(key)
                    for key in (
                        "id", "importance", "created_at", "record_type",
                        "subject_id", "location_id", "content", "occurred_at",
                        "valid_until", "visibility", "witnesses", "source",
                        "dedupe_key", "metadata_json",
                    )
                }
                for row in memory.records.all_rows()
            ],
        })

    @staticmethod
    def _person_for_actor(
        graph: KnowledgeGraph,
        actor: dict[str, Any],
        observer_id: str,
        result: ProjectionResult,
    ) -> Node:
        actor_id = str(actor["id"])
        ref = f"world:actor:{actor_id}"
        found = graph.find_by_external_ref(ref)
        if found is not None:
            return found
        if actor_id == observer_id:
            node = graph.ensure_self()
        else:
            name = str(actor.get("name") or actor_id)
            node = graph._find_person_by_name(name, [])
            if node is None:
                node = PersonNode(
                    id=f"person:world:{slugify(actor_id)}",
                    user_id="",
                    name=name,
                    text=name,
                    created_at=time.time(),
                    practice_times=[time.time()],
                    source=f"world_identity:actor:{actor_id}",
                )
                graph.add_node(node)
                result.added += 1
        graph.bind_external_ref(node, ref)
        graph.mark_character_scope(node)
        return node

    @staticmethod
    def _entity_for_location(
        graph: KnowledgeGraph, location: dict[str, Any], result: ProjectionResult
    ) -> EntityNode:
        location_id = str(location["id"])
        ref = f"world:location:{location_id}"
        found = graph.find_by_external_ref(ref)
        if isinstance(found, EntityNode):
            return found
        name = str(location.get("name") or location_id)
        prior_ids = set(graph.nodes)
        node = graph.ensure_entity(name, kind_label="place")
        if node.id not in prior_ids:
            node.source = f"world_identity:location:{location_id}"
            result.added += 1
        graph.bind_external_ref(node, ref)
        graph.mark_character_scope(node)
        return node

    @staticmethod
    def _visible(row: dict[str, Any], observer_id: str) -> bool:
        visibility = str(row.get("visibility") or "known")
        witnesses = [str(value) for value in (_loads(row.get("witnesses"), []) or [])]
        return visibility in {"public", "known"} or observer_id in witnesses

    def reconcile(
        self,
        graph: KnowledgeGraph,
        memory: Optional[Any],
        config: KnowledgeGraphConfig,
        *,
        character: Optional[dict[str, Any]] = None,
    ) -> ProjectionResult:
        result = ProjectionResult()
        desired_nodes: set[str] = set()
        desired_refs: set[str] = set()
        enabled = self._enabled(memory, config)

        # Re-resolve all WorldMemory identities on each source change. Refs are
        # source-owned labels, not ownership of the canonical node itself.
        for node in graph.nodes.values():
            node.external_refs = [
                ref for ref in (node.external_refs or []) if not ref.startswith("world:")
            ]

        if enabled:
            graph.ensure_self()
            observer_id = str(memory.observer_id)
            locations = memory.state_store.locations()
            actors = memory.state_store.actors()
            location_nodes: dict[str, EntityNode] = {}
            actor_nodes: dict[str, Node] = {}

            for location in locations:
                location_id = str(location["id"])
                desired_refs.add(f"world:location:{location_id}")
                location_nodes[location_id] = self._entity_for_location(
                    graph, location, result
                )
            for actor in actors:
                actor_id = str(actor["id"])
                desired_refs.add(f"world:actor:{actor_id}")
                actor_nodes[actor_id] = self._person_for_actor(
                    graph, actor, observer_id, result
                )

            def put_fact(
                node_id: str,
                source: str,
                content: str,
                endpoints: Iterable[Optional[Node]],
                *,
                importance: float = 0.7,
                confidence: float = 0.8,
                created_at: Optional[float] = None,
            ) -> None:
                text = content.strip()
                if not text:
                    return
                existing = graph.nodes.get(node_id)
                created = float(
                    created_at
                    if created_at is not None
                    else getattr(existing, "created_at", 0.0) or time.time()
                )
                desired_nodes.add(node_id)
                stored = _put_node(graph, FactNode(
                    id=node_id,
                    text=f"World fact: {text}",
                    content=text,
                    type="world",
                    confidence=confidence,
                    importance=importance,
                    created_at=created,
                    practice_times=[created],
                    source=source,
                ), result)
                graph.mark_character_scope(stored)
                _clear_structural_edges(graph, stored.id)
                for endpoint in dict.fromkeys(
                    node.id for node in endpoints if node is not None
                ):
                    graph.mark_character_scope(endpoint)
                    graph.upsert_edge(FactEdge(
                        src=endpoint,
                        dst=stored.id,
                        weight=max(0.35, 0.3 + 0.6 * importance),
                        confidence=confidence,
                        importance=importance,
                        timestamp=created,
                    ))

            for location in locations:
                location_id = str(location["id"])
                description = str(location.get("description") or "").strip()
                if description:
                    put_fact(
                        f"fact:world:location:{slugify(location_id)}",
                        f"world_location:{location_id}",
                        f"{location.get('name') or location_id}: {description}",
                        [location_nodes[location_id]],
                        importance=0.8,
                        confidence=0.9,
                    )

            for actor in actors:
                actor_id = str(actor["id"])
                home = str(actor.get("home_location") or "")
                if home and home in location_nodes:
                    put_fact(
                        f"fact:world:actor:{slugify(actor_id)}",
                        f"world_actor:{actor_id}",
                        f"{actor.get('name') or actor_id}'s home is "
                        f"{location_nodes[home].name}.",
                        [actor_nodes[actor_id], location_nodes[home]],
                        importance=0.7,
                    )

            rows = memory.records.all_rows()
            fact_rows: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []
            for row in rows:
                if row.get("valid_until") is not None or not self._visible(row, observer_id):
                    result.skipped += 1
                    continue
                record_type = str(row.get("record_type") or "fact")
                metadata = _loads(row.get("metadata_json"), {})
                if (
                    record_type == "fact"
                    and row.get("source") == "extraction"
                    and metadata.get("temporal_kind") != "durable"
                ):
                    # Legacy learned rows predate explicit validity and may be
                    # stale present-tense observations. Keep them in history,
                    # but never promote them to durable graph truth.
                    result.skipped += 1
                    continue
                if record_type == "event":
                    if (
                        str(row.get("source") or "") == "simulation"
                        and not config.world_include_simulation_events
                    ):
                        result.skipped += 1
                        continue
                    event_rows.append(row)
                elif record_type == "fact":
                    fact_rows.append(row)
                else:
                    result.skipped += 1

            event_rows.sort(
                key=lambda row: (
                    float(row.get("importance") or 0.0),
                    float(row.get("occurred_at") or row.get("created_at") or 0.0),
                    int(row.get("id") or 0),
                ),
                reverse=True,
            )
            cap = max(0, int(config.world_event_max_nodes))
            if len(event_rows) > cap:
                result.skipped += len(event_rows) - cap
                event_rows = event_rows[:cap]

            for row in fact_rows:
                row_id = int(row["id"])
                subject = actor_nodes.get(str(row.get("subject_id") or ""))
                location = location_nodes.get(str(row.get("location_id") or ""))
                put_fact(
                    f"fact:world:record:{row_id}",
                    f"world_records:{row_id}",
                    str(row.get("content") or ""),
                    [subject or graph.nodes.get(graph.SELF_ID), location],
                    importance=_clip(row.get("importance"), 0.6),
                    confidence=0.8,
                    created_at=float(row.get("created_at") or time.time()),
                )

            for row in event_rows:
                row_id = int(row["id"])
                summary = str(row.get("content") or "").strip()
                if not summary:
                    result.skipped += 1
                    continue
                timestamp = float(
                    row.get("occurred_at") or row.get("created_at") or time.time()
                )
                importance = _clip(row.get("importance"))
                node_id = f"episode:world:record:{row_id}"
                source = f"world_records:{row_id}"
                actor = actor_nodes.get(str(row.get("subject_id") or ""))
                location = location_nodes.get(str(row.get("location_id") or ""))
                endpoints = [actor or graph.nodes.get(graph.SELF_ID), location]
                endpoint_ids = list(dict.fromkeys(
                    node.id for node in endpoints if node is not None
                ))
                desired_nodes.add(node_id)
                stored = _put_node(graph, EpisodeNode(
                    id=node_id,
                    text=f"World event: {summary}",
                    summary=summary,
                    importance=importance,
                    timestamp=timestamp,
                    participants=endpoint_ids,
                    created_at=timestamp,
                    practice_times=[timestamp],
                    source=source,
                ), result)
                graph.mark_character_scope(stored)
                _clear_structural_edges(graph, stored.id)
                for endpoint in endpoint_ids:
                    graph.mark_character_scope(endpoint)
                    graph.upsert_edge(EpisodeEdge(
                        src=endpoint,
                        dst=stored.id,
                        weight=max(0.3, 0.3 + 0.5 * importance),
                        timestamp=timestamp,
                        importance=importance,
                    ))

        result.removed += _remove_stale_owned(
            graph,
            (
                "world_location:", "world_actor:", "world_routine:",
                "world_records:",
            ),
            desired_nodes,
        )

        # Remove only source-created identity nodes that became true orphans.
        for node in list(graph.nodes.values()):
            if not node.source.startswith("world_identity:"):
                continue
            has_world_ref = any(ref in desired_refs for ref in node.external_refs)
            if not has_world_ref and not graph._adj.get(node.id):
                graph.remove_node(node.id)
                result.removed += 1
        return result


register_graph_source_projector(HeartbeatGraphProjector())
register_graph_source_projector(WorldGraphProjector())


__all__ = [
    "GraphSourceProjector",
    "ProjectionResult",
    "HeartbeatGraphProjector",
    "WorldGraphProjector",
    "register_graph_source_projector",
    "get_graph_source_projector",
    "graph_source_projectors",
]
