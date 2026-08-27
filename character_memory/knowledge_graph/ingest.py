"""Ingestion: turn source memories into graph nodes + edges.

The graph is built by reading the source memories through their public APIs
(`all_rows`, `get_user_state`, `baseline`, …) — **never** by writing back to
them. This keeps the modules independent: the source memories stay the source
of truth, the graph is a derived view.

Ingestion drivers, one per source memory family:

- :func:`ingest_emotion` — SelfNode baseline + one RelationEdge per known user.
- :func:`ingest_summaries` — one PersonNode per `user_summary` row.
- :func:`ingest_facts` — one FactNode per `user_facts` row plus entities
  extracted by bounded batched LLM calls, with FactEdges from each fact's
  subject (Person / Entity / Self) to the fact.
- :func:`ingest_episodes` — one EpisodeNode per `episodic` row plus
  EpisodeEdges to every participant.
- :func:`ingest_wiki` — flat, no-LLM fallback: header chunks -> FactNodes.
- :func:`ingest_wiki_llm` — semantic wiki ingest: sections -> internal
  FactNodes (provenance anchors), atomic visible FactNodes, scored
  PersonNodes/RelationEdges, EntityNodes (named things), and EpisodeNodes
  (story events).

Entity/people extraction is **context-aware and relevance-filtered**. Both the
fact-extraction and wiki-extraction prompts are told:

- *who the character is* (name + persona) — the lens for "relevant to the
  character";
- *what is already in the graph* (existing entity names, known people) — so the
  LLM reuses an existing name rather than minting a synonym (Phonewave /
  PhoneWave / PhoneWave (Original) collapse to one node);
- the **named-vs-generic** rule (a *Phonewave* is an entity, a generic
  *microwave* is not) and the **relevant-person** rule (a friend/rival gets a
  PersonNode; a famous person mentioned only in passing does not).

Co-occurrence: facts and episodes that share participants get a
`CoOccurrenceEdge(co_create=True)` after each batch.

The functions take already-loaded rows/items so the same code path serves the
initial full ingest (`KnowledgeGraphRetriever.ingest`) and the incremental
post-extraction update (`KnowledgeGraphRetriever.update`).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

from ..llm.base import LLMClient
from ..emotion_vectors import decode_emotion_vector, emotional_impact
from ..memory.emotion import EmotionStatus
from ..memory.episodic import EpisodicMemory
from ..memory.user_facts import UserFactMemory
from ..memory.user_summary import UserSummaryMemory
from .edges import (
    ChatEdge,
    CoOccurrenceEdge,
    EpisodeEdge,
    FactEdge,
    RelationEdge,
)
from .graph import KnowledgeGraph, slugify
from .nodes import EntityNode, EpisodeNode, FactNode, Node, PersonNode


# A character identity carried through extraction so the LLM can judge what is
# "relevant to the character". Either field may be empty.
CharacterContext = dict[str, str]

# Default batch limits. The retriever passes the configured values explicitly;
# these defaults keep the direct ingestion functions convenient and backwards
# compatible for callers that use them without a retriever.
_FACT_BATCH_SIZE = 50
_EPISODE_BATCH_SIZE = 50
_WIKI_BATCH_SIZE = 3
_EXTRACTION_TOKEN_LIMIT = 10_000


def _token_counter() -> Callable[[str], int]:
    """Return a token counter, with a dependency-free approximation fallback."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except Exception:  # pragma: no cover - defensive fallback
        return lambda text: (len(text) + 3) // 4


_count_tokens = _token_counter()


def _batch_by_limits(
    items: Iterable[Any],
    *,
    max_items: int,
    max_tokens: int,
    text_of: Callable[[Any], str],
) -> list[list[Any]]:
    """Split ``items`` by both item count and aggregate source-text tokens.

    An individual item larger than ``max_tokens`` is emitted alone: silently
    dropping or truncating a stored memory would make the graph incomplete.
    Invalid zero/negative limits are clamped to one so hand-edited config
    cannot create an infinite loop.
    """
    item_limit = max(1, int(max_items))
    token_limit = max(1, int(max_tokens))
    batches: list[list[Any]] = []
    current: list[Any] = []
    current_tokens = 0

    for item in items:
        item_tokens = _count_tokens(str(text_of(item) or ""))
        if current and (
            len(current) >= item_limit
            or current_tokens + item_tokens > token_limit
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens

    if current:
        batches.append(current)
    return batches


def _now(clock: Optional[Callable[[], float]] = None) -> float:
    return (clock or time.time)()


def _char_clause(character: Optional[CharacterContext]) -> str:
    """Render the "you are building a graph for <name>" clause.

    Empty/missing persona degrades to just the name; both missing degrades to
    the generic "a role-play character" (legacy behaviour).
    """
    if not character:
        return "You are building a knowledge graph for a role-play character."
    name = (character.get("name") or "").strip()
    persona = (character.get("persona") or "").strip()
    if name and persona:
        return (
            f"You are building a knowledge graph for the role-play character "
            f"**{name}**. Persona / background: {persona}"
        )
    if name:
        return (
            f"You are building a knowledge graph for the role-play character "
            f"**{name}**."
        )
    return "You are building a knowledge graph for a role-play character."


# ===========================================================================
# Fact extraction (user_facts)
# ===========================================================================
# JSON schema for a batched fact-extraction LLM call. The LLM decides, per fact,
# whose fact
# it is and which entities (places / objects / organizations / concepts) it
# mentions. `thing` is deliberately NOT a kind: the model must pick a real
# category or omit the entity — this kills the lazy "everything is a thing"
# fallback that produced generic-noun noise.
_FACT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "the fact's 0-based index in the input list"},
                    "subject": {
                        "type": "string",
                        "description": (
                            "who the fact is primarily about: one of the known "
                            "user_ids, a relevant person's name, or 'self' if it "
                            "is about the character themselves. Never a place or "
                            "object — those go in 'entities'."
                        ),
                    },
                    "subject_id": {
                        "type": "string",
                        "description": (
                            "Stable id of an existing person shown in the graph "
                            "catalog, or 'self'. Use an empty string only when "
                            "the subject is genuinely new."
                        ),
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "existing_id": {
                                    "type": "string",
                                    "description": (
                                        "Stable entity id from the existing graph "
                                        "catalog; empty only for a genuinely new entity."
                                    ),
                                },
                                "kind": {
                                    "type": "string",
                                    "description": "place | organization | object | concept",
                                },
                            },
                            "required": ["name", "kind", "existing_id"],
                        },
                    },
                },
                "required": ["index", "subject", "subject_id"],
            },
        }
    },
    "required": ["facts"],
}

_FACT_EXTRACTION_PROMPT = (
    "{char_clause}\n\n"
    "You will be given a numbered list of facts about various people and things. "
    "For EACH fact decide:\n"
    "1. WHO the fact is primarily about — one of the known user_ids, 'self' (if it "
    "is about the character themselves), or the NAME of a person who is personally "
    "relevant to the character (a friend, rival, family member, colleague, someone "
    "the user actually interacts with). Do NOT use a place/object as the subject.\n"
    "2. Which ENTITIES the fact mentions — but only **specific, named things** that "
    "matter to the character.\n\n"
    "ENTITY RULE — what counts as an entity:\n"
    "- An entity is a *named, distinctive* element of the character's world: e.g. "
    "the Phonewave, the IBN 5100, D-Mail, the Future Gadget Lab, SERN, Reading "
    "Steiner, the divergence meter.\n"
    "- Do NOT extract generic nouns or everyday objects — a microwave, a camera, a "
    "lab coat, a hotel, a database, a chicken tender, 'metadata', a byte count are "
    "NOT entities. The test: if it is just *a* thing of its kind, skip it; if it is "
    "*the* named thing, keep it. (A generic microwave is not an entity; the "
    "Phonewave is.)\n"
    "- Each entity's `kind` MUST be one of: place | organization | object | concept. "
    "Pick the best fit; do not invent other kinds.\n\n"
    "PERSON RULE:\n"
    "- Only surface a person as the `subject` if they are personally relevant to the "
    "character. Do NOT create person entries for famous/historical/public figures "
    "mentioned merely as references (e.g. Einstein, Mozart) unless they are part of "
    "the character's actual story.\n"
    "- Reuse a name that is already in the graph (see below) rather than inventing a "
    "synonym, so the same thing/person is not duplicated. When a matching person "
    "already exists, copy its stable id into `subject_id`.\n\n"
    "ID REUSE RULE:\n"
    "- The catalog below is authoritative. If an existing person/entity is the same "
    "real thing under a nickname, abbreviation, capitalization, or longer/shorter "
    "name, return its exact id.\n"
    "- For an existing entity set `existing_id` and keep its catalog name. Use an "
    "empty id only when no catalog node refers to it. Never invent an id.\n\n"
    "{state_clause}"
    "Return one entry per fact using its index. Respond ONLY with the JSON object "
    "described by the schema."
)


def _existing_state_clause(
    graph: KnowledgeGraph,
    known_users: list[str],
    character: Optional[CharacterContext] = None,
) -> str:
    """Render the 'already in the graph' clause so the LLM reuses existing names.

    Capped so a huge graph doesn't blow the prompt: up to 80 entity names and
    known people (names + aliases). Names are lowercased-compared at resolve
    time, so this is a hint, not a hard constraint.

    When ``character`` is wired, the character's own name + aliases are
    declared up front with the instruction to use ``'self'`` for them — this
    is the extraction-layer reinforcement of the deterministic self-dedup so
    the LLM does not mint a PersonNode for the character under another name.
    """
    people: list[str] = []
    for n in graph.nodes_of_kind("person"):
        label = getattr(n, "name", "") or getattr(n, "user_id", "") or n.id
        aliases = [a for a in (getattr(n, "aliases", []) or []) if a]
        rendered = label if not aliases else f"{label} (aka {', '.join(aliases[:4])})"
        people.append(f"{n.id} = {rendered}")
    entities: list[str] = []
    for n in graph.nodes_of_kind("entity"):
        label = getattr(n, "name", "") or n.text or n.id
        aliases = [a for a in (getattr(n, "aliases", []) or []) if a]
        rendered = label if not aliases else f"{label} (aka {', '.join(aliases[:4])})"
        entities.append(f"{n.id} = {rendered}")
    entities.sort()
    parts: list[str] = []
    if character:
        name = (character.get("name") or "").strip()
        aliases = [str(a).strip() for a in (character.get("aliases") or []) if str(a).strip()]
        if name:
            aka = f" (aka {', '.join(aliases[:4])})" if aliases else ""
            parts.append(
                f"The character themselves is {name}{aka}. For ANY fact about them — under "
                f"any of these names — use the subject 'self'. Do NOT create a person entry "
                f"for the character."
            )
    if known_users:
        parts.append(f"Known user_ids: {known_users} (use 'self' for facts about the character).")
    if people:
        parts.append("People already in the graph (return the id when matched): " + "; ".join(people[:40]))
    if entities:
        parts.append("Entities already in the graph (return the id when matched): " + "; ".join(entities[:80]))
    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


def _extract_fact_subjects(
    llm: Optional[LLMClient],
    facts: list[dict[str, Any]],
    known_users: list[str],
    *,
    graph: KnowledgeGraph,
    character: Optional[CharacterContext] = None,
    _on_llm_request_done: Optional[Callable[[], None]] = None,
) -> dict[int, dict[str, Any]]:
    """Run the batched fact-extraction call. Falls back to heuristic on failure.

    Returns `{fact_index: {subject, entities}}`. When no LLM is wired, every
    fact is attributed to the user that owned it (its `user_id` column) and
    no entities are extracted — the graph still builds, just sparser.
    """
    fallback = {i: {"subject": str(f.get("user_id") or "self"), "entities": []} for i, f in enumerate(facts)}
    if not facts or llm is None:
        return fallback
    numbered = [f"{i}. {f.get('content') or f.get('text') or ''}" for i, f in enumerate(facts)]
    prompt = _FACT_EXTRACTION_PROMPT.format(
        char_clause=_char_clause(character),
        state_clause=_existing_state_clause(graph, known_users, character),
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Facts:\n" + "\n".join(numbered)},
    ]
    try:
        result = llm.chat_structured(messages, _FACT_EXTRACTION_SCHEMA)
    except Exception:
        return fallback
    finally:
        if _on_llm_request_done is not None:
            _on_llm_request_done()
    out: dict[int, dict[str, Any]] = {}
    for entry in result.get("facts", []) or []:
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        subject = str(entry.get("subject") or "").strip()
        if not subject:
            subject = str(facts[idx].get("user_id") or "self") if 0 <= idx < len(facts) else "self"
        subject_id = str(entry.get("subject_id") or "").strip()
        subject_node = graph.nodes.get(subject_id)
        if subject_id != graph.SELF_ID and not isinstance(subject_node, PersonNode):
            subject_id = ""
        entities = entry.get("entities") or []
        clean_entities = []
        for e in entities:
            if isinstance(e, dict):
                name = str(e.get("name") or "").strip()
                if name:
                    kind = str(e.get("kind") or "").strip().lower()
                    # Reject the lazy 'thing' kind and any unknown kind: the
                    # model must pick a real category or drop the entity.
                    if kind not in {"place", "organization", "object", "concept"}:
                        continue
                    existing_id = str(e.get("existing_id") or "").strip()
                    if not isinstance(graph.nodes.get(existing_id), EntityNode):
                        existing_id = ""
                    clean_entities.append({
                        "name": name,
                        "kind": kind,
                        "existing_id": existing_id,
                    })
        out[idx] = {
            "subject": subject,
            "subject_id": subject_id,
            "entities": clean_entities,
        }
    # Fill any indices the LLM skipped with the row-owner heuristic.
    for i, f in enumerate(facts):
        out.setdefault(i, {"subject": str(f.get("user_id") or "self"), "entities": []})
    return out


# --------------------------------------------------------------------- emotion
def ingest_emotion(
    graph: KnowledgeGraph,
    emotion: EmotionStatus,
    *,
    known_users: Optional[Iterable[str]] = None,
    character: Optional[CharacterContext] = None,
) -> None:
    """SelfNode baseline + RelationEdge per known user.

    `known_users` should be the union of every participant the character
    knows (from `user_summary`); each gets their own RelationEdge carrying
    their per-user emotion dims and relationship comment. `character` (name
    + aliases) seeds the SelfNode's searchable text so the character is one
    node, queryable by any of their names.
    """
    self_node = graph.ensure_self(
        baseline=getattr(emotion, "baseline", {}) or {},
        current_mood=emotion.get_current_mood(),
    )
    graph.mark_character_scope(self_node)
    self_node.text = _self_node_text(character)
    for uid in known_users or []:
        if not uid:
            continue
        person = graph.ensure_person(uid)  # make sure the node exists even w/o a summary
        graph.mark_user_scope(person, uid)
        state = emotion.get_user_state(uid)
        comment = emotion.get_user_comment(uid)
        # Strength of the relationship = mean magnitude of the signed
        # per-user relationship dimensions.
        magnitude = (
            sum(abs(float(v)) for v in state.values()) / max(1, len(state))
            if state
            else 0.0
        )
        edge = RelationEdge(
            id="",
            kind="relation",
            src=self_node.id,
            dst=person.id,
            weight=max(0.2, min(1.0, 0.3 + 0.7 * magnitude)),
            valence=float(state.get("valence", 0.0)),
            trust=float(state.get("trust", 0.0)),
            affection=float(state.get("affection", 0.0)),
            comment=comment or "",
            provenance="emotion",
        )
        stored = graph.upsert_edge(edge)
        # Relationship state is mutable source data. Unlike learned edge
        # strengths, a refresh must also propagate decreases and comment
        # changes rather than merging by max.
        if isinstance(stored, RelationEdge):
            stored.weight = edge.weight
            stored.valence = edge.valence
            stored.trust = edge.trust
            stored.affection = edge.affection
            stored.comment = edge.comment
            stored.provenance = "emotion"


# -------------------------------------------------------------------- summaries
def ingest_summaries(graph: KnowledgeGraph, summary: UserSummaryMemory) -> list[str]:
    """One PersonNode per `user_summary` row. Returns the user_ids ingested."""
    rows = summary.store.select(summary.table)
    users: list[str] = []
    for r in rows:
        uid = str(r.get("user_id") or "")
        if not uid:
            continue
        try:
            aliases = summary._parse_aliases(r.get("aliases"))
        except Exception:
            aliases = []
        name = str(r.get("name") or uid)
        person = graph.ensure_person(uid, name=name, aliases=aliases)
        graph.mark_user_scope(person, uid)
        person.text = summary.row_text(r)
        person.source = f"user_summary:{r.get('id')}"
        person.created_at = float(r.get("created_at") or 0.0) or person.created_at
        users.append(uid)
    return users


# ------------------------------------------------------------------------ facts
def ingest_facts(
    graph: KnowledgeGraph,
    facts_mem: UserFactMemory,
    *,
    llm: Optional[LLMClient] = None,
    known_users: Optional[list[str]] = None,
    rows: Optional[list[dict[str, Any]]] = None,
    character: Optional[CharacterContext] = None,
    batch_size: int = _FACT_BATCH_SIZE,
    token_limit: int = _EXTRACTION_TOKEN_LIMIT,
    _on_llm_request_done: Optional[Callable[[], None]] = None,
) -> list[str]:
    """FactNodes + FactEdges + EntityNodes for every row.

    `rows` lets the caller pass a pre-filtered subset (used by the
    incremental `update` path). When omitted, the whole table is read.
    `character` carries the character identity so the extraction prompt can
    judge relevance. Returns the ids of the FactNodes created.
    """
    rows = rows if rows is not None else facts_mem.store.select(facts_mem.table)
    if not rows:
        return []
    created_fact_ids: list[str] = []
    # Track entities per fact so we can wire co-occurrence between them.
    fact_participants: dict[str, set[str]] = {}
    known_user_ids = list(known_users or [])
    batches = _batch_by_limits(
        rows,
        max_items=batch_size,
        max_tokens=token_limit,
        text_of=lambda row: str(row.get("content") or row.get("text") or ""),
    )
    for batch in batches:
        subjects = _extract_fact_subjects(
            llm,
            batch,
            known_user_ids,
            graph=graph,
            character=character,
            _on_llm_request_done=_on_llm_request_done,
        )

        for i, r in enumerate(batch):
            content = str(r.get("content") or "").strip()
            if not content:
                continue
            info = subjects.get(i) or {
                "subject": str(r.get("user_id") or "self"),
                "entities": [],
            }
            subject = info["subject"]
            entities = info.get("entities") or []

            fid = graph.next_id("fact")
            created_at = float(r.get("created_at") or 0.0) or _now()
            fact_node = FactNode(
                id=fid,
                kind="fact",
                text=content,
                content=content,
                type=str(r.get("type") or "general"),
                confidence=_clip(r.get("confidence", 0.5)),
                importance=_clip(r.get("importance", 0.5)),
                created_at=created_at,
                source=f"user_facts:{r.get('id')}",
                # Seed the ACT-R practice history with the creation event so a
                # fresh fact has a meaningful (positive) base-level activation.
                practice_times=[created_at],
                chat_id=r.get("chat_id"),
            )
            graph.add_node(fact_node)
            owner = str(r.get("user_id") or "").strip()
            if owner:
                graph.mark_user_scope(fact_node, owner)
            else:
                graph.mark_scope_resolved(fact_node)
            created_fact_ids.append(fid)
            ts = float(r.get("created_at") or 0.0) or fact_node.created_at
            confidence = _clip(r.get("confidence", 0.5))
            importance = _clip(r.get("importance", 0.5))

            # Resolve the subject endpoint and link it to the fact. A named
            # person subject becomes a PersonNode; places/objects stay in
            # `entities` and never reach here.
            subj_id = _resolve_subject(
                graph,
                subject,
                known_user_ids,
                character,
                existing_id=str(info.get("subject_id") or ""),
            )
            if owner:
                graph.mark_user_scope(subj_id, owner)
            graph.upsert_edge(
                FactEdge(
                    id="",
                    kind="fact",
                    src=subj_id,
                    dst=fid,
                    weight=max(0.3, min(1.0, 0.3 + 0.7 * importance)),
                    confidence=confidence,
                    importance=importance,
                    timestamp=ts,
                )
            )
            participants: set[str] = {subj_id}

            # Entities the fact mentions become EntityNodes linked to the fact.
            for e in entities:
                ent = graph.ensure_entity(
                    e["name"],
                    kind_label=e.get("kind", "thing"),
                    existing_id=e.get("existing_id", ""),
                )
                if owner:
                    graph.mark_user_scope(ent, owner)
                graph.upsert_edge(
                    FactEdge(
                        id="",
                        kind="fact",
                        src=ent.id,
                        dst=fid,
                        weight=0.4,
                        confidence=confidence,
                        importance=importance,
                        timestamp=ts,
                    )
                )
                participants.add(ent.id)
            # A fact semantically asserted about the character is character
            # knowledge even when a user originally supplied the source row.
            if subj_id == graph.SELF_ID:
                for participant_id in {fid, *participants}:
                    graph.mark_character_scope(participant_id)
            fact_participants[fid] = participants

    _wire_co_occurrence(graph, fact_participants, co_create=True)
    return created_fact_ids


def _resolve_subject(
    graph: KnowledgeGraph,
    subject: str,
    known_users: list[str],
    character: Optional[CharacterContext] = None,
    existing_id: str = "",
) -> str:
    """Map an extraction-time `subject` string to a node id.

    Resolution order: the character themselves (``'self'`` token, the
    character's name, or any declared alias) -> SelfNode; a known user_id ->
    its PersonNode; a known person name/alias -> that PersonNode; otherwise a
    fresh **PersonNode** keyed by the name (a relevant person the fact is
    about — the "someone the user talks about who is relevant" rule).
    Places/objects never reach here; they are handled as entities in
    :func:`ingest_facts`.
    """
    s = subject.strip()
    existing_id = (existing_id or "").strip()
    if existing_id == graph.SELF_ID:
        return graph.SELF_ID
    existing = graph.nodes.get(existing_id)
    if isinstance(existing, PersonNode):
        return existing.id
    if not s:
        return graph.SELF_ID
    # The character routes to the singular SelfNode even when the LLM used
    # the character's real name or a nickname instead of the 'self' token.
    if _is_self_name(s, [], character):
        return graph.SELF_ID
    if s in known_users or f"person:{s}" in graph.nodes:
        return graph.ensure_person(s).id
    # Maybe the LLM used a name/alias; match against known persons.
    for node in graph.nodes.values():
        if getattr(node, "user_id", None) == s or getattr(node, "name", None) == s:
            return node.id
        if s in getattr(node, "aliases", []) or []:
            return node.id
        if getattr(node, "name", None) and node.name.lower() == s.lower():
            return node.id
    # A named relevant person the graph hasn't seen: create a PersonNode so
    # they are represented uniformly (not as an EntityNode).
    return graph.ensure_person_by_key(slugify(s), name=s).id


# --------------------------------------------------------------------- episodes
def ingest_episodes(
    graph: KnowledgeGraph,
    episodic: EpisodicMemory,
    *,
    rows: Optional[list[dict[str, Any]]] = None,
    batch_size: int = _EPISODE_BATCH_SIZE,
    token_limit: int = _EXTRACTION_TOKEN_LIMIT,
) -> list[str]:
    """EpisodeNodes + EpisodeEdges per participant. Returns episode node ids.

    Episode ingestion is deterministic and makes no LLM request. The shared
    limits still bound each processing batch and keep the API consistent with
    facts and wiki sections.
    """
    rows = rows if rows is not None else episodic.store.select(episodic.table)
    created: list[str] = []
    participants_by_ep: dict[str, set[str]] = {}
    batches = _batch_by_limits(
        rows,
        max_items=batch_size,
        max_tokens=token_limit,
        text_of=lambda row: str(row.get("summary") or ""),
    )
    for batch in batches:
        for r in batch:
            summary = str(r.get("summary") or "").strip()
            if not summary:
                continue
            owner = str(r.get("user_id") or "")
            ts = float(r.get("created_at") or 0.0) or _now()
            eid = graph.next_id("episode")
            ep_node = EpisodeNode(
                id=eid,
                kind="episode",
                text=summary,
                summary=summary,
                emotional_shift=decode_emotion_vector(
                    r.get("emotional_shift", "{}"),
                    allowed_axes=episodic.emotion_baseline,
                ),
                importance=_clip(r.get("importance", 0.5)),
                timestamp=ts,
                created_at=ts,
                source=f"episodic:{r.get('id')}",
                practice_times=[ts],
                chat_id=r.get("chat_id"),
            )
            graph.add_node(ep_node)
            if owner:
                graph.mark_user_scope(ep_node, owner)
            else:
                graph.mark_scope_resolved(ep_node)
            created.append(eid)
            # The owner is always a participant; surface them as a PersonNode.
            participants: set[str] = set()
            if owner:
                owner_node = graph.ensure_person(owner)
                graph.mark_user_scope(owner_node, owner)
                participants.add(owner_node.id)
            else:
                owner_node = None
            graph.upsert_edge(
                EpisodeEdge(
                    id="",
                    kind="episode",
                    src=owner_node.id if owner_node is not None else graph.SELF_ID,
                    dst=eid,
                    weight=max(
                        0.3,
                        min(
                            1.0,
                            0.3
                            + 0.7 * emotional_impact(ep_node.emotional_shift)
                            + 0.3 * ep_node.importance,
                        ),
                    ),
                    timestamp=ts,
                    emotional_shift=ep_node.emotional_shift,
                    importance=ep_node.importance,
                    recall=False,
                )
            )
            participants_by_ep[eid] = participants
    _wire_co_occurrence(graph, participants_by_ep, co_create=True)
    return created


# ----------------------------------------------------------------- co-occurrence
def _wire_co_occurrence(
    graph: KnowledgeGraph,
    participants_by_node: dict[str, set[str]],
    *,
    co_create: bool,
) -> None:
    """Add CoOccurrenceEdges between nodes that share participants."""
    node_ids = list(participants_by_node.keys())
    for i, a in enumerate(node_ids):
        pa = participants_by_node[a]
        for b in node_ids[i + 1 :]:
            pb = participants_by_node[b]
            if pa & pb:
                graph.add_co_occurrence(a, b, co_create=co_create, weight=0.15)


# --------------------------------------------------------------------- chat edges
def wire_chat_edges(graph: KnowledgeGraph, *, weight: float = 0.1) -> None:
    """Add ChatEdges between the facts and episodes of the same conversation.

    Nodes whose ``chat_id`` is set (i.e. learned in a chat) are grouped by it
    and every pair within a group is linked with a low, fixed-weight
    ``ChatEdge``. Wiki-derived nodes carry no ``chat_id`` and are skipped, so
    the wiki subgraph is left untouched.

    Idempotent: ``upsert_edge`` merges by the canonical id, so re-running this
    over a graph that already has the edges is a no-op (weight is merged by
    ``max``, never duplicated).
    """
    # Group node ids by chat_id, keeping only facts and episodes.
    by_chat: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        if not isinstance(node, (FactNode, EpisodeNode)):
            continue
        cid = node.chat_id
        if cid:
            by_chat.setdefault(cid, []).append(node.id)
    for cid, node_ids in by_chat.items():
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1 :]:
                graph.upsert_edge(
                    ChatEdge(id="", kind="chat", src=a, dst=b, weight=weight)
                )


# ----------------------------------------------------------------------- wiki
def _drop_wiki_projection(graph: KnowledgeGraph) -> None:
    """Remove the refreshable wiki-derived projection from ``graph``.

    Wiki facts and episodes are immutable graph projections of the current
    character-info chunks, so their incident native edges are removed with
    the nodes.  Wiki-only people/entities and wiki-owned relationship edges
    are also retired; canonical nodes shared with another source survive.
    """
    for node in list(graph.nodes.values()):
        if (node.source or "").startswith("wiki:") and node.kind in (
            "fact",
            "episode",
        ):
            graph.remove_node(node.id)
    for edge in list(graph.edges.values()):
        if isinstance(edge, RelationEdge) and edge.provenance == "wiki":
            graph.remove_edge(edge.id)
    for node in list(graph.nodes.values()):
        if not isinstance(node, (PersonNode, EntityNode)):
            continue
        wiki_refs = [
            ref for ref in (node.external_refs or []) if ref.startswith("wiki:")
        ]
        if not wiki_refs:
            continue
        node.external_refs = [
            ref for ref in (node.external_refs or []) if not ref.startswith("wiki:")
        ]
        if node.memory_owners or node.source or node.external_refs:
            continue
        # A wiki-only canonical node has no remaining source after its wiki
        # references and incident wiki projection edges are removed.
        graph.remove_node(node.id)


def ingest_wiki(
    graph: KnowledgeGraph,
    sections: Iterable[dict[str, Any]],
) -> list[str]:
    """Structural wiki ingest: chunks -> FactNodes (type='wiki') on SelfNode.

    No LLM: every header-chunked wiki section becomes one FactNode carrying
    the full section text, wired to the SelfNode with a FactEdge so it
    activates when the character or her world is queried. Idempotent: callers
    drop existing ``wiki:*`` nodes first (the retriever does this on every
    run) so re-ingesting never duplicates.
    """
    _drop_wiki_projection(graph)
    created: list[str] = []
    now = _now()
    for sec in sections:
        text = str(sec.get("text") or "").strip()
        if not text:
            continue
        header = str(sec.get("header") or "").strip()
        source = str(sec.get("source") or "wiki")
        fid = graph.next_id("fact")
        fact = FactNode(
            id=fid,
            kind="fact",
            text=text,
            content=text,
            type="wiki",
            confidence=0.9,
            importance=_clip(sec.get("importance", 0.6)),
            created_at=now,
            source=f"wiki:{source}:{header}",
            practice_times=[now],
        )
        graph.add_node(fact)
        graph.mark_character_scope(fact)
        created.append(fid)
        graph.upsert_edge(
            FactEdge(
                id="",
                kind="fact",
                src=graph.SELF_ID,
                dst=fid,
                weight=max(0.3, min(1.0, 0.3 + 0.7 * _clip(sec.get("importance", 0.6)))),
                confidence=0.9,
                importance=_clip(sec.get("importance", 0.6)),
                timestamp=now,
            )
        )
    return created


# ------------------------------------------------------------------- wiki (LLM)
# Structured wiki extraction is deliberately richer than the old typed
# projection.  The source chunk remains an internal anchor, while the LLM
# emits short semantic facts and a scored event projection that can be wired
# with the graph's native FactEdge/EpisodeEdge/RelationEdge types.
_WIKI_EMOTION_DEFAULT_AXES = (
    "neutral", "joy", "sadness", "anxiety", "anger", "surprise"
)


def _wiki_extraction_schema(emotion_axes: Iterable[str]) -> dict[str, Any]:
    """Build the wiki schema with the character's configured emotion axes."""
    shift_schema = {
        "type": "object",
        "properties": {
            axis: {"type": "number", "minimum": 0, "maximum": 1}
            for axis in emotion_axes
        },
        "additionalProperties": False,
    }
    relation_schema = {
        "type": "object",
        "properties": {
            "valence": {"type": "number", "minimum": -1, "maximum": 1},
            "trust": {"type": "number", "minimum": -1, "maximum": 1},
            "affection": {"type": "number", "minimum": -1, "maximum": 1},
            "comment": {"type": "string"},
        },
        "required": ["valence", "trust", "affection", "comment"],
        "additionalProperties": False,
    }
    person_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "existing_id": {"type": "string"},
            "name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "relevance": {"type": "string"},
            "relation": relation_schema,
        },
        "required": [
            "key", "name", "aliases", "relevance", "existing_id", "relation"
        ],
        "additionalProperties": False,
    }
    entity_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "name": {"type": "string"},
            "existing_id": {"type": "string"},
            "kind": {"type": "string"},
        },
        "required": ["key", "name", "kind", "existing_id"],
        "additionalProperties": False,
    }
    fact_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "self_included": {"type": "boolean"},
            "person_keys": {"type": "array", "items": {"type": "string"}},
            "entity_keys": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "content", "importance", "self_included", "person_keys", "entity_keys"
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "self_present": {"type": "boolean"},
                        "section_importance": {
                            "type": "number", "minimum": 0, "maximum": 1
                        },
                        "persons": {"type": "array", "items": person_schema},
                        "entities": {"type": "array", "items": entity_schema},
                        "facts": {"type": "array", "items": fact_schema},
                        "is_event": {"type": "boolean"},
                        "event_summary": {"type": "string"},
                        "event_importance": {
                            "type": "number", "minimum": 0, "maximum": 1
                        },
                        "event_self_participates": {"type": "boolean"},
                        "event_participant_keys": {
                            "type": "array", "items": {"type": "string"}
                        },
                        "emotional_shift": shift_schema,
                    },
                    "required": [
                        "index", "self_present", "section_importance", "persons",
                        "entities", "facts", "is_event", "event_summary",
                        "event_importance", "event_self_participates",
                        "event_participant_keys", "emotional_shift",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sections"],
        "additionalProperties": False,
    }

# Only people at these relevance levels get a node. `background` and
# `mentioned` are dropped (covers famous people referenced in passing).
_WIKI_KEEP_RELEVANCE = {"protagonist", "close", "supporting"}

_WIKI_EXTRACTION_PROMPT = (
    "{char_clause}\n\n"
    "You will be given several numbered wiki sections about this character's world. "
    "For EACH section identify:\n"
    "1. PERSONS — the named characters/people who appear. For each person give a "
    "stable canonical `key` (lowercase, no spaces — e.g. 'okabe'), their most formal "
    "`name`, every other name/nickname in `aliases` (so 'Okabe', 'Rintaro Okabe', "
    "'Hououin Kyouma' collapse into ONE person), and a `relevance` level:\n"
    "   - protagonist: the character themselves or a co-lead;\n"
    "   - close: a main companion / love interest / family the character is close to;\n"
    "   - supporting: a recurring character the character actually interacts with;\n"
    "   - background: a named character who only appears in passing;\n"
    "   - mentioned: someone merely referenced (including famous real people — "
    "Einstein, Mozart, etc. — unless they are genuinely part of the story).\n"
    "   Reuse the SAME `key` across sections for the same person. If the person "
    "is in the catalog below, copy its exact id into `existing_id` instead of "
    "minting a variant. Also estimate the character's relationship to that person "
    "from this source only: signed valence/trust/affection in [-1,1] and a short "
    "comment; use zero and an empty comment when the source gives no signal.\n"
    "2. ENTITIES — *named, distinctive* things in the character's world only: "
    "places, organizations, objects, concepts. Same rule as facts: the Phonewave / "
    "IBN 5100 / D-Mail / Future Gadget Lab are entities; a generic microwave / "
    "camera / lab coat / hotel / database is NOT. Each entity `kind` MUST be one of: "
    "place | organization | object | concept. If a catalog entity is the same "
    "thing under an abbreviation, nickname, capitalization, or longer/shorter "
    "name, return its exact `existing_id`; never invent an id.\n"
    "3. FACTS — split the section into short, self-contained semantic facts. "
    "For each, calculate importance in [0,1], set `self_included` only when the "
    "fact is about or directly involves the character, and reference people/entities "
    "using their exact keys. Do not copy the full source chunk into a fact.\n"
    "4. is_event — true if the section describes a story event/episode; if so, give "
    "a non-empty one-sentence event_summary, importance, participant keys, and the "
    "character's sparse emotional_shift on the configured axes. Set the shift to "
    "{{}} when the character does not participate. `self_present` controls only the "
    "section anchor; `event_self_participates` controls the Self→EpisodeEdge.\n\n"
    "{state_clause}"
    "Keep only specifically named elements. Respond ONLY with the JSON object "
    "described by the schema."
)


def _extract_wiki_batch(
    llm: Optional[LLMClient],
    batch: list[tuple[int, str]],
    *,
    graph: KnowledgeGraph,
    character: Optional[CharacterContext] = None,
    emotion_axes: Iterable[str] = _WIKI_EMOTION_DEFAULT_AXES,
    _on_llm_request_done: Optional[Callable[[], None]] = None,
) -> dict[int, dict[str, Any]]:
    """LLM-extract semantic wiki facts, people, entities and one event."""
    axes = tuple(str(axis) for axis in emotion_axes if str(axis).strip())
    empty = {
        i: {
            "self_present": False,
            "section_importance": 0.6,
            "is_event": False,
            "event_summary": "",
            "event_importance": 0.6,
            "event_self_participates": False,
            "event_participant_keys": [],
            "emotional_shift": {},
            "persons": [],
            "entities": [],
            "facts": [],
        }
        for i, _ in batch
    }
    if not batch or llm is None:
        return empty
    numbered = [f"{i}. {t}" for i, t in batch]
    prompt = _WIKI_EXTRACTION_PROMPT.format(
        char_clause=_char_clause(character),
        state_clause=_existing_state_clause(graph, [], character),
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Sections:\n" + "\n".join(numbered)},
    ]
    try:
        result = llm.chat_structured(messages, _wiki_extraction_schema(axes))
    except Exception:
        return empty
    finally:
        if _on_llm_request_done is not None:
            _on_llm_request_done()
    if not isinstance(result, dict):
        return empty
    out: dict[int, dict[str, Any]] = {}
    for entry in result.get("sections", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        persons_raw = entry.get("persons") or []
        persons: list[dict[str, Any]] = []
        for p in persons_raw:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            key = str(p.get("key") or "").strip().lower()
            relevance = str(p.get("relevance") or "").strip().lower()
            if not name or not key:
                continue
            aliases = [str(a).strip() for a in (p.get("aliases") or []) if str(a).strip() and str(a).strip() != name]
            existing_id = str(p.get("existing_id") or "").strip()
            if not isinstance(graph.nodes.get(existing_id), PersonNode):
                existing_id = ""
            relation = p.get("relation") if isinstance(p.get("relation"), dict) else {}
            if not relation:
                relation = p
            persons.append({
                "key": key,
                "name": name,
                "aliases": aliases,
                "relevance": relevance,
                "existing_id": existing_id,
                "relation": {
                    "valence": _clip(relation.get("valence", 0.0), -1.0, 1.0, 0.0),
                    "trust": _clip(relation.get("trust", 0.0), -1.0, 1.0, 0.0),
                    "affection": _clip(relation.get("affection", 0.0), -1.0, 1.0, 0.0),
                    "comment": str(relation.get("comment") or "").strip(),
                },
            })
        entities: list[dict[str, str]] = []
        for e in entry.get("entities") or []:
            if isinstance(e, dict):
                name = str(e.get("name") or "").strip()
                if name:
                    kind = str(e.get("kind") or "").strip().lower()
                    if kind not in {"place", "organization", "object", "concept"}:
                        continue
                    existing_id = str(e.get("existing_id") or "").strip()
                    if not isinstance(graph.nodes.get(existing_id), EntityNode):
                        existing_id = ""
                    key = str(e.get("key") or slugify(name)).strip().lower()
                    entities.append({
                        "key": key,
                        "name": name,
                        "kind": kind,
                        "existing_id": existing_id,
                    })
        facts: list[dict[str, Any]] = []
        for f in entry.get("facts") or []:
            if not isinstance(f, dict):
                continue
            content = str(
                f.get("content") or f.get("text") or f.get("summary") or ""
            ).strip()
            if not content:
                continue
            raw_people = f.get("person_keys") or f.get("persons") or f.get("participants") or []
            raw_entities = f.get("entity_keys") or f.get("entities") or []
            person_keys: list[str] = []
            for ref in raw_people:
                if isinstance(ref, dict):
                    ref = ref.get("key") or ref.get("id") or ref.get("name")
                if str(ref).strip():
                    person_keys.append(str(ref).strip().lower())
            entity_keys: list[str] = []
            for ref in raw_entities:
                if isinstance(ref, dict):
                    ref = ref.get("key") or ref.get("id") or ref.get("name")
                if str(ref).strip():
                    entity_keys.append(str(ref).strip().lower())
            subject = str(f.get("subject") or "").strip()
            self_included = bool(
                f.get("self_included", f.get("involves_self", False))
            )
            if subject and _is_self_name(subject, [], character):
                self_included = True
            elif subject:
                person_keys.append(subject.lower())
            facts.append({
                "content": content,
                "importance": _clip(f.get("importance", 0.6)),
                "self_included": self_included,
                "person_keys": list(dict.fromkeys(
                    k for k in person_keys if k
                )),
                "entity_keys": list(dict.fromkeys(
                    k for k in entity_keys if k
                )),
            })
        try:
            emotional_shift = decode_emotion_vector(
                entry.get("emotional_shift")
                or entry.get("event_emotional_shift")
                or {},
                allowed_axes=axes,
            )
        except (TypeError, ValueError):
            emotional_shift = {}
        event_self_participates = bool(
            entry.get(
                "event_self_participates",
                entry.get("self_participates", entry.get("self_included", False)),
            )
        )
        if not event_self_participates:
            emotional_shift = {}
        out[idx] = {
            "self_present": bool(
                entry.get("self_present", entry.get("character_present", False))
            ),
            "section_importance": _clip(
                entry.get("section_importance", entry.get("importance", 0.6))
            ),
            "is_event": bool(entry.get("is_event")),
            "event_summary": str(
                entry.get("event_summary") or entry.get("summary") or ""
            ).strip(),
            "event_importance": _clip(
                entry.get("event_importance", entry.get("importance", 0.6))
            ),
            "event_self_participates": event_self_participates,
            "event_participant_keys": list(dict.fromkeys(
                str(
                    k.get("key") or k.get("id") or k.get("name")
                    if isinstance(k, dict)
                    else k
                ).strip().lower()
                for k in (
                    entry.get("event_participant_keys")
                    or entry.get("participants")
                    or []
                )
                if str(
                    k.get("key") or k.get("id") or k.get("name")
                    if isinstance(k, dict)
                    else k
                ).strip()
            )),
            "emotional_shift": emotional_shift,
            "persons": persons,
            "entities": entities,
            "facts": facts,
        }
    out.update({i: empty[i] for i, _ in batch if i not in out})
    return out


def ingest_wiki_llm(
    graph: KnowledgeGraph,
    llm: LLMClient,
    sections: Iterable[dict[str, Any]],
    *,
    batch_size: int = _WIKI_BATCH_SIZE,
    token_limit: int = _EXTRACTION_TOKEN_LIMIT,
    character: Optional[CharacterContext] = None,
    _on_llm_request_done: Optional[Callable[[], None]] = None,
) -> list[str]:
    """Build a native semantic graph from LLM-extracted wiki sections.

    Each section keeps one hidden full-text FactNode as a provenance anchor.
    The LLM additionally emits atomic visible facts, relevant people/entities,
    and an optional scored episode.  All topology uses the native relation,
    fact, and episode edge types; no source-specific association edge exists.
    """
    _drop_wiki_projection(graph)
    items = list(enumerate(sections))
    if not items:
        return []
    self_node = graph.ensure_self()
    emotion_axes = tuple(getattr(self_node, "baseline", {}) or _WIKI_EMOTION_DEFAULT_AXES)
    extractions: dict[int, dict[str, Any]] = {}
    batches = _batch_by_limits(
        items,
        max_items=batch_size,
        max_tokens=token_limit,
        text_of=lambda item: str(item[1].get("text") or ""),
    )
    for sub in batches:
        batch_extractions = _extract_wiki_batch(
            llm,
            [(i, sec.get("text", "")) for i, sec in sub],
            graph=graph,
            character=character,
            emotion_axes=emotion_axes,
            _on_llm_request_done=_on_llm_request_done,
        )
        extractions.update(batch_extractions)
        # Seed the canonical catalog immediately so the next LLM batch sees
        # and reuses nodes found in this one.  Previously all batches were
        # extracted before any node was created, making cross-batch reuse
        # entirely dependent on the model remembering an unseen key.
        for ext in batch_extractions.values():
            for person_data in ext.get("persons", []):
                if person_data.get("relevance") not in _WIKI_KEEP_RELEVANCE:
                    continue
                if _looks_like_self(
                    graph,
                    person_data["name"],
                    person_data.get("aliases") or [],
                    character,
                ):
                    continue
                person = graph.ensure_person_by_key(
                    person_data["key"],
                    name=person_data["name"],
                    aliases=person_data.get("aliases") or [],
                    existing_id=person_data.get("existing_id", ""),
                )
                graph.mark_character_scope(person)
                graph.bind_external_ref(person, f"wiki:person:{person_data['key']}")
            for entity_data in ext.get("entities", []):
                entity = graph.ensure_entity(
                    entity_data["name"],
                    kind_label=entity_data.get("kind", "thing"),
                    existing_id=entity_data.get("existing_id", ""),
                )
                graph.mark_character_scope(entity)
                graph.bind_external_ref(
                    entity, f"wiki:entity:{entity_data.get('key') or slugify(entity_data['name'])}"
                )

    created_fact_ids: list[str] = []
    now = _now()
    for i, sec in items:
        text = str(sec.get("text") or "").strip()
        if not text:
            continue
        header = str(sec.get("header") or "").strip()
        source = str(sec.get("source") or "wiki")
        section_source = f"wiki:{source}:{header}"
        section_importance = _clip(ext.get("section_importance", 0.6))
        fid = graph.next_id("fact")
        fact = FactNode(
            id=fid,
            kind="fact",
            text=text,
            content=text,
            type="wiki",
            confidence=0.9,
            importance=section_importance,
            created_at=now,
            source=section_source,
            practice_times=[now],
            internal=True,
        )
        graph.add_node(fact)
        graph.mark_character_scope(fact)
        created_fact_ids.append(fid)
        ext = extractions.get(i, {})
        person_ids: list[str] = []
        person_by_key: dict[str, str] = {}
        relation_data: dict[str, dict[str, Any]] = {}
        # Relevant people -> PersonNode (canonical key, aliases merged). The
        # SelfNode's own character is represented by `self`, not a person node.
        for p in ext.get("persons", []):
            person_by_key[p["key"]] = graph.SELF_ID if _looks_like_self(
                graph, p["name"], p.get("aliases") or [], character
            ) else ""
            if p.get("relevance") not in _WIKI_KEEP_RELEVANCE:
                continue
            key = p["key"]
            name = p["name"]
            aliases = p.get("aliases") or []
            # Avoid minting a person node for the character themselves: they
            # are already the SelfNode.
            if _looks_like_self(graph, name, aliases, character):
                person_by_key[key] = graph.SELF_ID
                continue
            person = graph.ensure_person_by_key(
                key,
                name=name,
                aliases=aliases,
                existing_id=p.get("existing_id", ""),
            )
            graph.mark_character_scope(person)
            person_by_key[key] = person.id
            person_ids.append(person.id)
            relation_data[person.id] = p.get("relation") or {}
            graph.bind_external_ref(person, f"wiki:person:{key}")
        # Establish or refresh a native relationship to every relevant person.
        for pid, rel in relation_data.items():
            existing = graph.get_edge_between("relation", graph.SELF_ID, pid)
            if isinstance(existing, RelationEdge) and existing.provenance != "wiki":
                # Emotion memory is authoritative for user-owned relations;
                # it already supplies the direct Self -> Person link.
                continue
            valence = _clip(rel.get("valence", 0.0), -1.0, 1.0, 0.0)
            trust = _clip(rel.get("trust", 0.0), -1.0, 1.0, 0.0)
            affection = _clip(rel.get("affection", 0.0), -1.0, 1.0, 0.0)
            magnitude = (abs(valence) + abs(trust) + abs(affection)) / 3.0
            relation = RelationEdge(
                id="",
                kind="relation",
                src=graph.SELF_ID,
                dst=pid,
                weight=max(0.2, min(1.0, 0.3 + 0.7 * magnitude)),
                valence=valence,
                trust=trust,
                affection=affection,
                comment=str(rel.get("comment") or "").strip(),
                provenance="wiki",
            )
            stored = graph.upsert_edge(relation)
            if isinstance(stored, RelationEdge):
                stored.weight = relation.weight
                stored.valence = relation.valence
                stored.trust = relation.trust
                stored.affection = relation.affection
                stored.comment = relation.comment
                stored.provenance = "wiki"
        entity_by_key: dict[str, str] = {}
        # Named entities -> EntityNode.
        for e in ext.get("entities", []):
            ent = graph.ensure_entity(
                e["name"],
                kind_label=e.get("kind", "thing"),
                existing_id=e.get("existing_id", ""),
            )
            graph.mark_character_scope(ent)
            ekey = str(e.get("key") or slugify(e["name"])).strip().lower()
            entity_by_key[ekey] = ent.id
            graph.bind_external_ref(ent, f"wiki:entity:{ekey}")

        def person_id_for_key(key: str) -> str:
            """Resolve an extracted person reference, including Self aliases."""
            normalized = str(key or "").strip().lower()
            if normalized == "self" or _is_self_name(normalized, [], character):
                return graph.SELF_ID
            return person_by_key.get(normalized, "")

        # The hidden anchor is wired only to endpoints actually present in its
        # section.  These native FactEdges preserve activation/provenance while
        # keeping the full chunk out of normal results.
        anchor_endpoints: list[str] = []
        if ext.get("self_present"):
            anchor_endpoints.append(graph.SELF_ID)
        anchor_endpoints.extend(person_ids)
        anchor_endpoints.extend(entity_by_key.values())
        for endpoint in dict.fromkeys(anchor_endpoints):
            graph.upsert_edge(
                FactEdge(
                    id="",
                    kind="fact",
                    src=endpoint,
                    dst=fid,
                    weight=max(0.3, min(1.0, 0.3 + 0.7 * section_importance)),
                    confidence=0.9,
                    importance=section_importance,
                    timestamp=now,
                )
            )

        # Atomic semantic facts are visible and carry their own calculated
        # importance.  Only explicitly referenced endpoints receive a FactEdge.
        for fact_data in ext.get("facts", []):
            content = str(fact_data.get("content") or "").strip()
            if not content:
                continue
            importance = _clip(fact_data.get("importance", 0.6))
            fact_id = graph.next_id("fact")
            semantic_fact = FactNode(
                id=fact_id,
                kind="fact",
                text=content,
                content=content,
                type="wiki",
                confidence=0.9,
                importance=importance,
                created_at=now,
                source=f"{section_source}:fact",
                practice_times=[now],
            )
            graph.add_node(semantic_fact)
            graph.mark_character_scope(semantic_fact)
            created_fact_ids.append(fact_id)
            endpoints: list[str] = []
            if fact_data.get("self_included"):
                endpoints.append(graph.SELF_ID)
            endpoints.extend(
                person_id_for_key(key)
                for key in fact_data.get("person_keys", [])
                if person_id_for_key(key)
            )
            endpoints.extend(
                entity_by_key[key]
                for key in fact_data.get("entity_keys", [])
                if entity_by_key.get(key)
            )
            for endpoint in dict.fromkeys(endpoints):
                graph.upsert_edge(
                    FactEdge(
                        id="",
                        kind="fact",
                        src=endpoint,
                        dst=fact_id,
                        weight=max(0.3, min(1.0, 0.3 + 0.7 * importance)),
                        confidence=0.9,
                        importance=importance,
                        timestamp=now,
                    )
                )

        if ext.get("is_event") and ext.get("event_summary"):
            summary = str(ext["event_summary"]).strip()
            participant_ids: list[str] = []
            if ext.get("event_self_participates"):
                participant_ids.append(graph.SELF_ID)
            participant_ids.extend(
                person_id_for_key(key)
                for key in ext.get("event_participant_keys", [])
                if person_id_for_key(key)
            )
            participant_ids = list(dict.fromkeys(participant_ids))
            if not participant_ids:
                continue
            event_importance = _clip(ext.get("event_importance", 0.6))
            event_shift = dict(ext.get("emotional_shift") or {}) if ext.get(
                "event_self_participates"
            ) else {}
            eid = graph.next_id("episode")
            ep = EpisodeNode(
                id=eid,
                kind="episode",
                text=summary,
                summary=summary,
                emotional_shift=event_shift,
                importance=event_importance,
                timestamp=now,
                created_at=now,
                source=f"{section_source}:episode",
                practice_times=[now],
                participants=participant_ids,
            )
            graph.add_node(ep)
            graph.mark_character_scope(ep)
            for pid in participant_ids:
                graph.upsert_edge(
                    EpisodeEdge(
                        id="",
                        kind="episode",
                        src=pid,
                        dst=eid,
                        weight=max(
                            0.3,
                            min(
                                1.0,
                                0.3
                                + (0.7 * emotional_impact(event_shift) if pid == graph.SELF_ID else 0.0)
                                + 0.3 * event_importance,
                            ),
                        ),
                        timestamp=now,
                        emotional_shift=event_shift if pid == graph.SELF_ID else {},
                        importance=event_importance,
                        recall=False,
                    )
                )
    return created_fact_ids


def _looks_like_self(
    graph: KnowledgeGraph,
    name: str,
    aliases: list[str],
    character: Optional[CharacterContext],
) -> bool:
    """True if `name`/`aliases` refer to the character themselves.

    The SelfNode already represents the character; minting a PersonNode for
    them too would split their identity across two nodes. We check the
    character context name and the SelfNode text.
    """
    return _is_self_name(name, aliases, character)


def _self_labels(character: Optional[CharacterContext]) -> set[str]:
    """Lowercased set of all names by which the character is known.

    Combines ``character['name']`` and ``character['aliases']`` (the unified
    list wired from the config — hand-authored ∪ persona-scanned). The
    literal token ``'self'`` is always included so the fact-extraction
    subject token maps cleanly to the SelfNode.
    """
    if not character:
        return {"self"}
    out: set[str] = {"self"}
    cn = (character.get("name") or "").strip().lower()
    if cn:
        out.add(cn)
    for a in character.get("aliases") or []:
        a = str(a or "").strip().lower()
        if a:
            out.add(a)
    return out


def _is_self_name(
    name: str,
    aliases: Iterable[str],
    character: Optional[CharacterContext],
) -> bool:
    """True if `name`/`aliases` denote the character themselves.

    Central self-detection used by BOTH ingestion paths (fact subjects and
    wiki persons) so the character can never be split into ``self`` plus one
    or more ``person:`` nodes. A candidate matches when any of its labels
    (lowercased) equals the character's name, any declared alias, or the
    literal ``'self'`` token.
    """
    candidates = {str(c).strip().lower() for c in [name, *aliases] if str(c or "").strip()}
    if not candidates:
        return False
    return bool(candidates & _self_labels(character))


def _self_node_text(character: Optional[CharacterContext]) -> str:
    """Searchable text for the SelfNode: the character's name + aliases.

    Replaces the old generic literal ``'the character'`` so the SelfNode is
    actually surfaced by name/alias queries and so substring-based
    self-detection has a real signal. Falls back to ``'the character'`` when
    no identity is wired.
    """
    if not character:
        return "the character"
    parts = [c for c in [character.get("name") or "", *(character.get("aliases") or [])] if c]
    parts = list(dict.fromkeys([p.strip() for p in parts if p and p.strip()]))
    return ". ".join(parts) or "the character"


# --------------------------------------------------------------------- helpers
def _clip(v: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


__all__ = [
    "ingest_emotion",
    "ingest_summaries",
    "ingest_facts",
    "ingest_episodes",
    "ingest_wiki",
    "ingest_wiki_llm",
]
