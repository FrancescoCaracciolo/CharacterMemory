"""Knowledge-graph retrieval for character memories.

An optional, additive retriever that ingests from the existing memory
systems (`user_facts`, `user_summary`, `emotion`, `episodic`) into a
per-character knowledge graph and retrieves via spreading activation with
ACT-R base-level learning and Hebbian co-occurrence reinforcement.

The source memories are the source of truth; the graph is a derived view
and never writes back to them.

See :doc:`docs/knowledge_graph` for the full design.
"""

from .activation import (
    base_level_activation,
    combined_activation,
    spread_activation,
)
from .edges import (
    CoOccurrenceEdge,
    Edge,
    EpisodeEdge,
    FactEdge,
    RelationEdge,
    TransitionEdge,
    edge_from_dict,
)
from .graph import KnowledgeGraph, slugify
from .ingest import (
    ingest_emotion,
    ingest_episodes,
    ingest_facts,
    ingest_summaries,
)
from .nodes import (
    EntityNode,
    EpisodeNode,
    FactNode,
    Node,
    PersonNode,
    SelfNode,
    node_from_dict,
)
from .persistence import has_persisted, load_graph, save_graph
from .retriever import (
    KnowledgeGraphConfig,
    KnowledgeGraphRetriever,
    KnowledgeGraphRetrivier,
)

__all__ = [
    # graph
    "KnowledgeGraph",
    "slugify",
    # nodes
    "Node",
    "SelfNode",
    "PersonNode",
    "FactNode",
    "EpisodeNode",
    "EntityNode",
    "node_from_dict",
    # edges
    "Edge",
    "RelationEdge",
    "FactEdge",
    "TransitionEdge",
    "EpisodeEdge",
    "CoOccurrenceEdge",
    "edge_from_dict",
    # activation
    "base_level_activation",
    "spread_activation",
    "combined_activation",
    # ingest
    "ingest_emotion",
    "ingest_summaries",
    "ingest_facts",
    "ingest_episodes",
    # persistence
    "save_graph",
    "load_graph",
    "has_persisted",
    # retriever
    "KnowledgeGraphRetriever",
    "KnowledgeGraphRetrivier",
    "KnowledgeGraphConfig",
]
