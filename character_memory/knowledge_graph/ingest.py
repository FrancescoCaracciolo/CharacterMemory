"""Ingestion: turn source memories into graph nodes + edges.

The graph is built by reading the source memories through their public APIs
(`all_rows`, `get_user_state`, `baseline`, …) — **never** by writing back to
them. This keeps the modules independent: the source memories stay the source
of truth, the graph is a derived view.

Ingestion drivers, one per source memory family:

- :func:`ingest_emotion` — SelfNode baseline + one RelationEdge per known user.
- :func:`ingest_summaries` — one PersonNode per `user_summary` row.
- :func:`ingest_facts` — one FactNode per `user_facts` row plus entities
  extracted by a single batched LLM call, with FactEdges from each fact's
  subject (Person / Entity / Self) to the fact.
- :func:`ingest_episodes` — one EpisodeNode per `episodic` row plus
  EpisodeEdges to every participant.
- :func:`ingest_wiki` — flat, no-LLM fallback: header chunks -> FactNodes.
- :func:`ingest_wiki_llm` — typed wiki ingest: sections -> FactNodes +
  PersonNodes (relevant characters) + EntityNodes (named things) +
  EpisodeNodes (story events).

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
from .nodes import EpisodeNode, FactNode, Node


# A character identity carried through extraction so the LLM can judge what is
# "relevant to the character". Either field may be empty.
CharacterContext = dict[str, str]


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
# JSON schema for the batched fact-extraction LLM call. One call over all
# facts at once keeps ingestion cheap; the LLM decides, per fact, whose fact
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
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "description": "place | organization | object | concept",
                                },
                            },
                            "required": ["name", "kind"],
                        },
                    },
                },
                "required": ["index", "subject"],
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
    "synonym, so the same thing/person is not duplicated.\n\n"
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

    Capped so a huge graph doesn't blow the prompt: ~60 entity names and all
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
        people.append(label if not aliases else f"{label} (aka {', '.join(aliases[:3])})")
    entities = sorted(
        {getattr(n, "name", "") or n.text for n in graph.nodes_of_kind("entity")}
    )
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
        parts.append("People already in the graph (reuse these names, do not duplicate): " + "; ".join(people[:40]))
    if entities:
        parts.append("Entities already in the graph (reuse these names, do not duplicate): " + "; ".join(entities[:60]))
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
    out: dict[int, dict[str, Any]] = {}
    for entry in result.get("facts", []) or []:
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        subject = str(entry.get("subject") or "").strip()
        if not subject:
            subject = str(facts[idx].get("user_id") or "self") if 0 <= idx < len(facts) else "self"
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
                    clean_entities.append({"name": name, "kind": kind})
        out[idx] = {"subject": subject, "entities": clean_entities}
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
    self_node = graph.ensure_self(baseline=getattr(emotion, "baseline", {}) or {})
    self_node.text = _self_node_text(character)
    for uid in known_users or []:
        if not uid:
            continue
        graph.ensure_person(uid)  # make sure the node exists even w/o a summary
        state = emotion.get_user_state(uid)
        comment = emotion.get_user_comment(uid)
        # Strength of the relationship = mean magnitude of the signed dims.
        magnitude = (
            sum(abs(float(v)) for v in state.values()) / max(1, len(state))
            if state
            else 0.0
        )
        edge = RelationEdge(
            id="",
            kind="relation",
            src=self_node.id,
            dst=f"person:{uid}",
            weight=max(0.2, min(1.0, 0.3 + 0.7 * magnitude)),
            valence=float(state.get("valence", 0.0)),
            trust=float(state.get("trust", 0.0)),
            affection=float(state.get("affection", 0.0)),
            comment=comment or "",
        )
        graph.upsert_edge(edge)


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
    # Resolve subjects + entities in one LLM call over the whole batch.
    subjects = _extract_fact_subjects(
        llm, rows, list(known_users or []), graph=graph, character=character
    )

    created_fact_ids: list[str] = []
    # Track entities per fact so we can wire co-occurrence between them.
    fact_participants: dict[str, set[str]] = {}

    for i, r in enumerate(rows):
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        info = subjects.get(i) or {"subject": str(r.get("user_id") or "self"), "entities": []}
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
        created_fact_ids.append(fid)
        ts = float(r.get("created_at") or 0.0) or fact_node.created_at
        confidence = _clip(r.get("confidence", 0.5))
        importance = _clip(r.get("importance", 0.5))

        # Resolve the subject endpoint and link it to the fact. A named person
        # subject becomes a PersonNode (the "someone else relevant" rule); a
        # place/object stays in `entities` and never reaches here.
        subj_id = _resolve_subject(graph, subject, known_users or [], character)
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
            ent = graph.ensure_entity(e["name"], kind_label=e.get("kind", "thing"))
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
        fact_participants[fid] = participants

    _wire_co_occurrence(graph, fact_participants, co_create=True)
    return created_fact_ids


def _resolve_subject(
    graph: KnowledgeGraph,
    subject: str,
    known_users: list[str],
    character: Optional[CharacterContext] = None,
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
    if not s:
        return graph.SELF_ID
    # The character routes to the singular SelfNode even when the LLM used
    # the character's real name or a nickname instead of the 'self' token.
    if _is_self_name(s, [], character):
        return graph.SELF_ID
    if s in known_users or f"person:{s}" in graph.nodes:
        return f"person:{s}"
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
) -> list[str]:
    """EpisodeNodes + EpisodeEdges per participant. Returns episode node ids."""
    rows = rows if rows is not None else episodic.store.select(episodic.table)
    created: list[str] = []
    participants_by_ep: dict[str, set[str]] = {}
    for r in rows:
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
            emotional_shift=_clip(r.get("emotional_shift", 0.0), -1.0, 1.0, 0.0),
            importance=_clip(r.get("importance", 0.5)),
            timestamp=ts,
            created_at=ts,
            source=f"episodic:{r.get('id')}",
            practice_times=[ts],
            chat_id=r.get("chat_id"),
        )
        graph.add_node(ep_node)
        created.append(eid)
        # The owner is always a participant; surface them as a PersonNode.
        participants: set[str] = set()
        if owner:
            graph.ensure_person(owner)
            participants.add(f"person:{owner}")
        graph.upsert_edge(
            EpisodeEdge(
                id="",
                kind="episode",
                src=f"person:{owner}" if owner else graph.SELF_ID,
                dst=eid,
                weight=max(0.3, min(1.0, 0.3 + 0.7 * abs(ep_node.emotional_shift) + 0.3 * ep_node.importance)),
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
        created.append(fid)
        graph.upsert_edge(
            FactEdge(
                id="",
                kind="fact",
                src=graph.SELF_ID,
                dst=fid,
                weight=0.5,
                confidence=0.9,
                importance=0.6,
                timestamp=now,
            )
        )
    return created


# ------------------------------------------------------------------- wiki (LLM)
# Structured person extraction: each relevant character becomes ONE node
# keyed by a stable canonical `key`, with all its names folded into
# `aliases`. `relevance` drives the keep/drop decision (background /
# mentioned-only people are dropped — this excludes famous people referenced
# in passing unless they are genuinely part of the story).
_WIKI_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "is_event": {
                        "type": "boolean",
                        "description": "true if this section describes a story event / episode / chapter",
                    },
                    "event_summary": {
                        "type": "string",
                        "description": "one-sentence summary of the event, if is_event",
                    },
                    "persons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": (
                                        "a stable canonical identifier for this person, "
                                        "lowercase, no spaces (e.g. 'okabe'). Reuse the same "
                                        "key across sections so the same person becomes one node."
                                    ),
                                },
                                "name": {"type": "string", "description": "the most formal / complete name"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "every other name/nickname used for this person",
                                },
                                "relevance": {
                                    "type": "string",
                                    "description": "protagonist | close | supporting | background | mentioned",
                                },
                            },
                            "required": ["key", "name", "relevance"],
                        },
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "description": "place | organization | object | concept",
                                },
                            },
                            "required": ["name", "kind"],
                        },
                    },
                },
                "required": ["index", "is_event", "persons", "entities"],
            },
        }
    },
    "required": ["sections"],
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
    "   Reuse the SAME `key` across sections for the same person.\n"
    "2. ENTITIES — *named, distinctive* things in the character's world only: "
    "places, organizations, objects, concepts. Same rule as facts: the Phonewave / "
    "IBN 5100 / D-Mail / Future Gadget Lab are entities; a generic microwave / "
    "camera / lab coat / hotel / database is NOT. Each entity `kind` MUST be one of: "
    "place | organization | object | concept.\n"
    "3. is_event — true if the section describes a story event/episode; if so, a "
    "one-sentence event_summary.\n\n"
    "Keep only specifically named elements. Respond ONLY with the JSON object "
    "described by the schema."
)


def _extract_wiki_batch(
    llm: Optional[LLMClient],
    batch: list[tuple[int, str]],
    *,
    graph: KnowledgeGraph,
    character: Optional[CharacterContext] = None,
) -> dict[int, dict[str, Any]]:
    """LLM-extract persons/entities/events for one batch of (index, text)."""
    empty = {i: {"is_event": False, "persons": [], "entities": []} for i, _ in batch}
    if not batch or llm is None:
        return empty
    numbered = [f"{i}. {t}" for i, t in batch]
    prompt = _WIKI_EXTRACTION_PROMPT.format(char_clause=_char_clause(character))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Sections:\n" + "\n".join(numbered)},
    ]
    try:
        result = llm.chat_structured(messages, _WIKI_EXTRACTION_SCHEMA)
    except Exception:
        return empty
    out: dict[int, dict[str, Any]] = {}
    for entry in result.get("sections", []) or []:
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
            persons.append({"key": key, "name": name, "aliases": aliases, "relevance": relevance})
        entities: list[dict[str, str]] = []
        for e in entry.get("entities") or []:
            if isinstance(e, dict):
                name = str(e.get("name") or "").strip()
                if name:
                    kind = str(e.get("kind") or "").strip().lower()
                    if kind not in {"place", "organization", "object", "concept"}:
                        continue
                    entities.append({"name": name, "kind": kind})
        out[idx] = {
            "is_event": bool(entry.get("is_event")),
            "event_summary": str(entry.get("event_summary") or "").strip(),
            "persons": persons,
            "entities": entities,
        }
    out.update({i: empty[i] for i, _ in batch if i not in out})
    return out


def ingest_wiki_llm(
    graph: KnowledgeGraph,
    llm: LLMClient,
    sections: Iterable[dict[str, Any]],
    *,
    batch_size: int = 6,
    character: Optional[CharacterContext] = None,
) -> list[str]:
    """LLM wiki ingest: sections -> FactNodes + PersonNodes + EntityNodes + EpisodeNodes.

    Each header-chunked wiki section becomes a ``FactNode(type='wiki')`` linked
    to the SelfNode (so it activates on character/world queries). An LLM then
    extracts, per section: relevant **people** (``PersonNode``, keyed by a
    canonical key with aliases merged, only protagonist/close/supporting kept),
    named **entities** (``EntityNode`` — place/org/object/concept; generic
    nouns dropped), and story **events** (``EpisodeNode``). All extracted nodes
    are wired to the section fact and to each other via co-occurrence, so the
    wiki becomes a typed subgraph rather than flat fact chunks. Idempotent: the
    retriever drops prior ``wiki:*`` fact/episode nodes first; person/entity
    nodes dedupe by key/name and are preserved.
    """
    items = list(enumerate(sections))
    if not items:
        return []
    extractions: dict[int, dict[str, Any]] = {}
    for b in range(0, len(items), max(1, batch_size)):
        sub = items[b : b + batch_size]
        extractions.update(
            _extract_wiki_batch(
                llm, [(i, sec.get("text", "")) for i, sec in sub],
                graph=graph, character=character,
            )
        )

    created_fact_ids: list[str] = []
    section_participants: dict[str, set[str]] = {}
    now = _now()
    for i, sec in items:
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
            importance=0.6,
            created_at=now,
            source=f"wiki:{source}:{header}",
            practice_times=[now],
        )
        graph.add_node(fact)
        created_fact_ids.append(fid)
        graph.upsert_edge(
            FactEdge(
                id="",
                kind="fact",
                src=graph.SELF_ID,
                dst=fid,
                weight=0.5,
                confidence=0.9,
                importance=0.6,
                timestamp=now,
            )
        )
        ext = extractions.get(i, {})
        participants: set[str] = {graph.SELF_ID}
        # Relevant people -> PersonNode (canonical key, aliases merged). The
        # SelfNode's own character is represented by `self`, not a person node.
        for p in ext.get("persons", []):
            if p.get("relevance") not in _WIKI_KEEP_RELEVANCE:
                continue
            key = p["key"]
            name = p["name"]
            aliases = p.get("aliases") or []
            # Avoid minting a person node for the character themselves: they
            # are already the SelfNode.
            if _looks_like_self(graph, name, aliases, character):
                continue
            person = graph.ensure_person_by_key(key, name=name, aliases=aliases)
            graph.upsert_edge(
                FactEdge(
                    id="",
                    kind="fact",
                    src=person.id,
                    dst=fid,
                    weight=0.5,
                    confidence=0.9,
                    importance=0.6,
                    timestamp=now,
                )
            )
            participants.add(person.id)
        # Named entities -> EntityNode.
        for e in ext.get("entities", []):
            ent = graph.ensure_entity(e["name"], kind_label=e.get("kind", "thing"))
            graph.upsert_edge(
                FactEdge(
                    id="",
                    kind="fact",
                    src=ent.id,
                    dst=fid,
                    weight=0.45,
                    confidence=0.9,
                    importance=0.5,
                    timestamp=now,
                )
            )
            participants.add(ent.id)
        if ext.get("is_event"):
            summary = ext.get("event_summary") or text
            eid = graph.next_id("episode")
            ep = EpisodeNode(
                id=eid,
                kind="episode",
                text=summary,
                summary=summary,
                emotional_shift=0.0,
                importance=0.6,
                timestamp=now,
                created_at=now,
                source=f"wiki:{source}:{header}",
                practice_times=[now],
            )
            graph.add_node(ep)
            for pid in participants:
                if pid == graph.SELF_ID:
                    continue
                graph.upsert_edge(
                    EpisodeEdge(
                        id="",
                        kind="episode",
                        src=pid,
                        dst=eid,
                        weight=0.5,
                        timestamp=now,
                        emotional_shift=0.0,
                        importance=0.6,
                        recall=False,
                    )
                )
            participants.add(eid)
        section_participants[fid] = participants
    _wire_co_occurrence(graph, section_participants, co_create=True)
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
