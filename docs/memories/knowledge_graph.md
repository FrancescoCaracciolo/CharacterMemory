# `knowledge_graph` — spreading-activation retrieval over a graph

| | |
|---|---|
| **Class** | `KnowledgeGraphMemory` (`memory/knowledge_graph_memory.py`) wrapping `KnowledgeGraphRetriever` |
| **Flavor** | Custom (its own persistence layout, `kg_index/`) |
| **Scope** | `PER_USER` (the queried user's `PersonNode` gets an activation bias) |
| **Learns from chat?** | **Indirectly** — reads the other memories as sources; never writes back |
| **Config toggle** | `MemoryConfig.enabled_knowledge_graph` (default **off**) |
| **Retrieval size** | `MemoryConfig.knowledge_graph_token_budget` (default `1000` tokens) |

> The full end-to-end design — node/edge types, ingestion, ACT-R decay,
> Hebbian co-occurrence, spreading activation — is in
> [`docs/knowledge_graph.md`](../knowledge_graph.md). This page covers how to
> *use* it from the agent / API.

## What it does

An **additive, opt-in** memory that sits alongside the others. The existing
memories remain the **source of truth**; the KG reads them, builds a graph
view, and retrieves through:

- **Spreading activation** (2 hops, per-hop attenuation),
- **ACT-R base-level learning** (decay over time, boosted by re-recall),
- **Hebbian co-occurrence reinforcement** (nodes recalled together
  strengthen their link).

The graph is **per-character, shared** across users, so Person↔Person
"transition" edges are meaningful. Modules stay independent: the KG never
writes back to a source memory.

## Enabling it

Three equivalent ways:

1. Ship an empty `.knowledge_graph` marker file in the character directory
   (this is how Kurisu ships with KG on).
2. Set the env var `CM_KG_CHARACTERS=Kurisu,Mayuri` (server-side).
3. Set `MemoryConfig.enabled_knowledge_graph = True` in code / `config.yaml`.

The KG is **off by default** because building it runs an LLM extraction pass
over the wiki.

## Lifecycle

| Method | Effect |
|---|---|
| `agent.build()` | Load `kg_index/` if present, then reconcile deterministic source projections when their source/version changed. |
| `agent.rebuild()` | Full re-build (everything, including the KG). |
| `agent.rebuild_knowledge_graph()` | KG-only rebuild — cheap, targeted. |
| `charactermemory-server --rebuild-kg [NAMES]` | Rebuild at server startup, then serve. |

After extraction, the agent calls `kg.retriever.update(added)` so the graph
ingests the freshly-added source rows **incrementally**, then mirrors any
dedup mutations via `apply_deduplication`.

Heartbeat and world data use deterministic source projectors instead of the
chat extractor. Heartbeat discoveries/actions are admitted by importance and
projected as facts/episodes. World locations are place entities, and visible
durable world facts/events link to their structured actors and locations.
World routines and mutable actor state are intentionally excluded; the
observer's current location contributes only a transient retrieval seed.

## Recall

`KnowledgeGraphMemory.recall(query, user_id, limit)` delegates to
`KnowledgeGraphRetriever.retrieve`, which:

1. Seeds activation on nodes whose text matches the query (hybrid search over
   node text), with the user's `PersonNode` and the `SelfNode` seeded too.
2. Spreads activation over the graph (gain × hop_decay per hop).
3. Combines the BLL term and the spreading term
   (`base_weight` / `spread_weight`).
4. Selects activation-ranked nodes within `limit` rendered-body tokens and,
   for state-changing recall, records exposure and a Hebbian step.
5. Returns those nodes as `MemoryItem`s, grouped by kind in the section
   (people → facts → episodes → entities).

`limit` is a token budget, not a node count. The agent supplies
`memory.knowledge_graph_token_budget` (default `1000`). If a
[global budget or reranker](../memory_budget.md) is active, graph candidates
are first retrieved read-only, then compete with other memories. Only graph
items selected for the final context receive exposure/Hebbian reinforcement.
The global cap also counts the graph section's header.

## Tuning

See `KnowledgeGraphConfig` in `config.py` — `decay`, `decay_half_life`,
`gain`, `hops` (pinned at 2), `hop_decay`, `base_weight`, `spread_weight`,
`min_activation`, `hebbian_threshold`, `hebbian_lr`, `self_seed`,
`match_base`, `match_gain`. Also overridable per-character via
`config.yaml`:

```yaml
memory:
  enabled_knowledge_graph: true
  knowledge_graph:
    decay: 0.5
    gain: 0.35
    hops: 2
    hop_decay: 0.6
    fact_batch_size: 50
    episode_batch_size: 50
    wiki_batch_size: 3
    extraction_token_limit: 10000
    project_heartbeat: true
    heartbeat_min_importance: 0.6
    heartbeat_max_nodes: 200
    project_world: true
    world_event_max_nodes: 500
    world_include_simulation_events: false
    world_location_seed: 0.6
```

The item limit and token limit are both enforced, so whichever is reached
first closes the batch. `extraction_token_limit` counts the source fact,
episode-summary, or wiki text; the system prompt, JSON schema, and generated
response add protocol overhead. Stored episodes are ingested deterministically
without an LLM request, but use the same bounded batching utility.

## Character self-dedup

The character itself is one node (`self`) under every name it goes by. The
retriever folds any `PersonNode` matching the character's name or any
declared alias into that singular `self` node, then merges `PersonNode`s that
share a name/alias. Aliases come from:

1. The `aliases:` top-level key in `config.yaml`,
2. A best-effort scan of the persona (quoted nicknames, parentheticals,
   "also known as" / "aka" clauses),
3. The canonical `name`.

Run `deduplicate_knowledge_graph` (MCP) or
`kg.retriever.deduplicate_persons()` for a one-time cleanup of a graph built
before self-dedup existed, or after editing `aliases`.

## Section header

`"Activated knowledge (graph)"`. Overridable via
`PromptConfig.knowledge_graph_header[_multi]`.

## Inspecting it

- **WebUI** — the `/gui` graph visualizer at
  `/api/graph/{character}?q=…&user=…&full=…` returns an activation-weighted
  subgraph for rendering (node radius / opacity / color encode activation).
- **MCP** — `graph_overview` (counts per kind + users) and
  `search_knowledge_graph` (activation search returning records + the
  activation subgraph), followed by `get_knowledge_graph_nodes` for complete
  node values or `get_knowledge_graph_neighbors` for typed edge expansion.
  Use `/mcp?character=<name>&tools=kg` for a graph-only MCP surface.
- **Python** — `kg = agent.memories["knowledge_graph"].retriever`;
  `kg.overview()`, `kg.test_activation(query, user_id)`,
  `kg.retrieve(query, user_id, limit)`.

## Standalone usage (against a built agent)

```python
from character_memory import CharacterAgent, LLMConfig, EmbeddingConfig, MemoryConfig

agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config(LLMConfig(), EmbeddingConfig(),
                       MemoryConfig(enabled_knowledge_graph=True))
agent.build()
kg = agent.memories["knowledge_graph"].retriever

print(kg.overview())
trace = kg.test_activation("what is the phonewave?", user_id="michael")
for nid, a in sorted(trace.items(), key=lambda kv: kv[1], reverse=True)[:10]:
    node = kg.graph.nodes.get(nid)
    print(f"  {a:7.3f}  {nid:24s}  kind={getattr(node, 'kind', '?'):8s}")
```

See `examples/knowledge_graph.py` for the full demo (seeds facts/summaries,
exercises `ingest` / `test_activation` / `retrieve` / `save`).
