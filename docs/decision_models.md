# Decision models and memory reconciliation

Decision models are a separate extension seam from chat generation. Implement
`DecisionClient.decide(DecisionRequest) -> DecisionResponse` to add a backend.
`decide_many` defaults to sequential calls. Questions can be boolean, categorical,
or ordered scores. HTTP adapters validate response IDs, types, allowed labels,
probability ranges/distributions, and score consistency. Provider confidence is
kept separate from label probability. LLM-generated probabilities are estimated;
a label without probabilities has origin `unavailable`.

## Configure Jev

```yaml
memory:
  dedup:
    enabled: true
    decision_provider: openrouter  # typesafe or llm are also supported
    decision_model: typesafe/jev-1.13
    decision_timeout: 30
    decision_candidate_pool: 20
    decision_max_request_bytes: 24000
    duplicate_probability: 0.98
    correction_probability: 0.99
    decision_margin: 0.20
    decision_llm_fallback: true
```

Set `OPENROUTER_API_KEY` for OpenRouter or `TYPESAFE_API_KEY` for TypeSafe.
TypeSafe defaults to `jev-latest`; omit `decision_model` when using that default.
OpenRouter uses `https://openrouter.ai/api/alpha/decisions`; TypeSafe uses
`https://api.typesafe.ai/v1/systemone`. These adapters do not use chat completions.
The resolved model is recorded in revision evidence. Pin the provider's model
version when measuring thresholds. Configuration contains no credentials.

Alternatively, inject a client into the agent or a standalone deduplicator:

```python
from character_memory import CharacterAgent, OpenRouterDecisionClient

agent = CharacterAgent(
    'assets/Kurisu',
    decision_client=OpenRouterDecisionClient(model='typesafe/jev-1.13'),
)
agent.load_from_config().build()
preview = agent.dedup('user_facts', user_id='alice', dry_run=True)
report = agent.dedup('user_facts', user_id='alice')
```

`memory.dedup.enabled` still controls automatic post-extraction passes. Explicit
`agent.dedup()` runs even when automatic passes are disabled. With no supplied
client or configured provider, existing deduplication behavior is retained.
`dry_run` also works on that legacy path. Preview can make model calls but does
not commit row changes, revisions, graph changes, or local index changes.

## Policy

- Equivalent assertions discard the incoming row without revising the survivor.
- Corrections keep the canonical row ID, archive before/after snapshots and
  evidence, and replace assertion-specific metadata and source message IDs.
  Old confidence is not carried into a new assertion. Recall telemetry and the
  canonical row's creation timestamp are retained.
- Current-state facts/directives can transition while preserving history.
  Dated episodes remain separate. A custom memory can set
  `supports_state_transitions = True`; its `contradiction_policy()` must also
  enable corrections.
- Ambiguous candidates, conflicting actions, or multiple correction targets
  retain both entries. One structured-LLM fallback checks the relationship and
  whether sufficient evidence exists. It never uses estimated confidence as a
  calibrated probability. Disable fallback to avoid the extra call.
- A newer insertion timestamp alone is not evidence of a correction. Source
  messages and revision history accompany candidate rows. Text is evidence,
  not instructions to the classifier.

Candidate retrieval is scoped by memory and user before comparison. Small
candidate groups use their existing rows directly; larger groups use bounded
hybrid retrieval plus a current-pass overlay. A single request asks about every
candidate in a batch. Oversized requests split at candidate boundaries; if one
candidate's complete evidence cannot fit, it remains unresolved rather than
being silently truncated. Retrieval is bounded, so a candidate missed by the
shortlist can remain undeduplicated. Sweeps use the same policy as extraction.

`DedupReport` includes proposed/applied decisions, unresolved count, model calls,
fallback calls, cache hits, and the existing removed/updated IDs. In decision
mode, `consolidate`, cosine deletion thresholds, and `per_user=False` do not
permit merges or cross-user deletion. Memory scope is authoritative.

## History and recovery

```python
history = agent.memories['user_facts'].revisions(row_id)
```

History lives in `memory_revisions`, outside recall indexes. Duplicate discards
create no history entry. Transactions recheck both source snapshots and commit
revision insertion, replacement, deletion, and synchronization work together.
SQLite and PostgreSQL use their existing portable transaction implementations.

`memory_reconciliation_pending` is a durable retry queue. `agent.build()` repairs
pending work after loading indexes, and `agent.persist_structured()` publishes
RAG and graph changes before acknowledging the captured queue entries. Failed
publication leaves work pending. In a standalone memory integration, call
`sync_reconciliation_index()` to replay RAG deltas and retain pending entries
until your graph/persistence coordinator has published them. Normal agent use
handles this automatically.

## Validation

Offline tests:

```bash
CM_ASSETS_DIR=/tmp/cm-empty-test-assets python3 -m pytest -q \
  tests/test_decisions.py tests/test_decision_reconciliation.py
```

Opt-in provider checks require `CM_TEST_DECISIONS_LIVE=1` and the corresponding
credential. PostgreSQL checks require `CM_TEST_DATABASE_URL` pointing to a test
database. Run `tests/test_decisions_live.py` for these contracts.

Evaluate thresholds on representative labeled pairs before enabling automatic
reconciliation. Run `python3 benchmarks/decision_reconciliation.py --help` for
the evaluation tool. Track false deletions, correction precision, abstentions,
model calls, and latency. Candidate recall requires labeled retrieval candidates
in the evaluation input. The defaults are provisional thresholds, not measured
accuracy guarantees. Historical versions cannot be reconstructed for corrections
that happened before revision recording was enabled.
