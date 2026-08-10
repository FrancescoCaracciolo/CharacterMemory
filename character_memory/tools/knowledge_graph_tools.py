"""Read-only, provider-neutral tools for knowledge-graph retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import Tool
from .event_tools import CalculateTimeDifference, ResolveTimeRange

if TYPE_CHECKING:
    from ..agent import CharacterAgent
    from ..knowledge_graph.nodes import Node
    from ..memory.knowledge_graph_memory import KnowledgeGraphMemory


def _kg_memory(agent: "CharacterAgent") -> "KnowledgeGraphMemory":
    from ..memory.knowledge_graph_memory import KnowledgeGraphMemory

    memory = (getattr(agent, "memories", {}) or {}).get("knowledge_graph")
    if not isinstance(memory, KnowledgeGraphMemory) or not memory.enabled:
        raise RuntimeError("knowledge_graph memory is not available")
    return memory


def _node_text(node: "Node") -> str:
    for field in ("content", "summary", "name", "text"):
        value = getattr(node, field, None)
        if value:
            return str(value)
    return node.id


def _node_record(
    node: "Node", *, activation: Optional[float] = None, max_chars: int = 3000
) -> dict[str, Any]:
    text = _node_text(node)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "…"
    record: dict[str, Any] = {
        "node_id": node.id,
        "kind": node.kind,
        "source": node.source,
        "text": text,
        "truncated": truncated,
    }
    if activation is not None:
        record["activation"] = float(activation)
    timestamp = getattr(node, "timestamp", None)
    if timestamp:
        record["timestamp"] = float(timestamp)
    return record


class SearchKnowledgeGraph(Tool):
    name = "search_knowledge_graph"
    description = (
        "Search the character's knowledge graph using hybrid matching and "
        "spreading activation. Use focused queries or sub-questions. Results "
        "include stable node IDs, node kinds, source provenance, activation, "
        "and compact node text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Focused graph query."},
            "user_id": {"type": "string"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": 8,
            },
            "max_chars_each": {
                "type": "integer",
                "minimum": 300,
                "maximum": 6000,
                "default": 2500,
            },
        },
        "required": ["query"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 8,
        max_chars_each: int = 2500,
    ) -> dict[str, Any]:
        memory = _kg_memory(self.agent)
        cap = max(1, min(30, int(limit)))
        max_chars = max(300, min(6000, int(max_chars_each)))
        uid = user_id or "default"
        activation = memory.retriever.test_activation(query, user_id=uid)
        ranked = sorted(activation.items(), key=lambda item: item[1], reverse=True)
        records = [
            _node_record(node, activation=score, max_chars=max_chars)
            for node_id, score in ranked
            if score >= memory.retriever.config.min_activation
            if (node := memory.retriever.graph.get_node(node_id)) is not None
        ][:cap]
        return {
            "query": query,
            "user_id": uid,
            "count": len(records),
            "nodes": records,
        }


class GetKnowledgeGraphNodes(Tool):
    name = "get_knowledge_graph_nodes"
    description = (
        "Fetch complete knowledge-graph nodes by stable node ID after a graph "
        "search identifies relevant candidates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
            },
            "max_chars_each": {
                "type": "integer",
                "minimum": 500,
                "maximum": 12000,
                "default": 6000,
            },
        },
        "required": ["node_ids"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(
        self, node_ids: list[str], max_chars_each: int = 6000
    ) -> dict[str, Any]:
        memory = _kg_memory(self.agent)
        ids = [str(node_id) for node_id in node_ids[:20]]
        max_chars = max(500, min(12000, int(max_chars_each)))
        nodes = [memory.retriever.graph.get_node(node_id) for node_id in ids]
        return {
            "requested_node_ids": ids,
            "count": sum(node is not None for node in nodes),
            "nodes": [
                _node_record(node, max_chars=max_chars)
                for node in nodes
                if node is not None
            ],
        }


class GetKnowledgeGraphNeighbors(Tool):
    name = "get_knowledge_graph_neighbors"
    description = (
        "Expand one knowledge-graph node to its directly connected nodes and "
        "typed edges. Use this for multi-hop questions, related facts, sequence "
        "reconstruction, or identifying other episodes connected to a person."
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 12,
            },
            "max_chars_each": {
                "type": "integer",
                "minimum": 300,
                "maximum": 5000,
                "default": 1800,
            },
        },
        "required": ["node_id"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(
        self, node_id: str, limit: int = 12, max_chars_each: int = 1800
    ) -> dict[str, Any]:
        memory = _kg_memory(self.agent)
        graph = memory.retriever.graph
        anchor = graph.get_node(str(node_id))
        if anchor is None:
            return {"node_id": str(node_id), "count": 0, "neighbors": []}
        cap = max(1, min(50, int(limit)))
        max_chars = max(300, min(5000, int(max_chars_each)))
        neighbors = []
        for edge in graph.edges.values():
            if edge.src == anchor.id:
                other_id = edge.dst
            elif edge.dst == anchor.id:
                other_id = edge.src
            else:
                continue
            other = graph.get_node(other_id)
            if other is None:
                continue
            neighbors.append(
                {
                    "edge": edge.to_dict(),
                    "node": _node_record(other, max_chars=max_chars),
                }
            )
            if len(neighbors) >= cap:
                break
        return {
            "node_id": anchor.id,
            "anchor": _node_record(anchor, max_chars=max_chars),
            "count": len(neighbors),
            "neighbors": neighbors,
        }


def knowledge_graph_tools(agent: "CharacterAgent") -> list[Tool]:
    """Return the standard read-only KG and temporal tools bound to an agent."""
    return [
        SearchKnowledgeGraph(agent),
        GetKnowledgeGraphNodes(agent),
        GetKnowledgeGraphNeighbors(agent),
        ResolveTimeRange(),
        CalculateTimeDifference(),
    ]


__all__ = [
    "SearchKnowledgeGraph",
    "GetKnowledgeGraphNodes",
    "GetKnowledgeGraphNeighbors",
    "knowledge_graph_tools",
]
