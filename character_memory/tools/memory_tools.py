"""Built-in tools that let the character read its own memory mid-conversation.

All tools here are **read-only**. The character writing its own memories during
a chat would bypass the extractor + deduplicator (which run out-of-band on the
extraction interval), so writes stay on the extractor / MCP-server path. If you
need a character-driven write, register your own tool — but think twice.

The factory :func:`memory_tools` takes a :class:`~character_memory.agent.CharacterAgent`
and returns a list of :class:`Tool` instances closing over it, ready to drop
into ``generate_answer(tools=...)``. They reuse the memory objects' existing
``recall`` / ``get_memories`` / ``all_rows`` methods, so retrieval quality is
exactly what the prompt already gets — no second code path to keep in sync.

Recall inside a tool is always issued with ``state_changing=False`` so a
character querying its own memory does not bump recall-count / last-recalled
bookkeeping (that would skew decay statistics just for asking).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import Tool

if TYPE_CHECKING:  # avoid an import-time dependency on the agent package
    from ..agent import CharacterAgent


def _agent_memories(agent: "CharacterAgent") -> dict:
    """The agent's enabled memories, keyed by name (empty if not loaded)."""
    return getattr(agent, "memories", {}) or {}


def memory_tools(agent: "CharacterAgent") -> list[Tool]:
    """Build the read-only memory self-tools bound to ``agent``.

    Returns a fresh list every call; pass it to ``generate_answer(tools=...)``
    or merge it with your own tools::

        tools = memory_tools(agent) + [GetWeather()]
        agent.generate_answer(chat, tools=tools)
    """
    return [
        _SearchMemory(agent),
        _GetUserFacts(agent),
        _GetUserSummary(agent),
        _GetUserEmotion(agent),
        _ListKnownUsers(agent),
    ]


# --------------------------------------------------------------------------- #
# search_memory
# --------------------------------------------------------------------------- #
class _SearchMemory(Tool):
    name = "search_memory"
    description = (
        "Search the character's own memories for information relevant to a "
        "query. Searches across the enabled memories (character info, "
        "dialogue style, user facts, directives, episodes, …) using the same "
        "hybrid retrieval the prompt uses. Use this to look something up "
        "before answering when you are unsure."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "memory": {
                "type": "string",
                "description": "Restrict the search to one memory by name. "
                "Omit to search every enabled memory.",
            },
            "user_id": {
                "type": "string",
                "description": "For per-user memories, whose data to search. "
                "Omit for character-scoped memories.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, query: str, memory: Optional[str] = None,
            user_id: Optional[str] = None, limit: int = 5) -> str:
        all_mems = _agent_memories(self.agent)
        if memory:
            if memory not in all_mems:
                return f"No memory named {memory!r}. Known: {sorted(all_mems)}."
            mems = {memory: all_mems[memory]}
        else:
            mems = all_mems
        if not mems:
            return "No memories are available."
        limit = max(1, min(20, int(limit)))
        uid = user_id or "default"
        lines: list[str] = []
        for name, mem in mems.items():
            if not getattr(mem, "enabled", True):
                continue
            try:
                # ``limit`` remains the tool's output-item cap. KG prompt
                # retrieval itself is token-budgeted, so give it the same
                # budget as normal prompt construction and slice afterward.
                recall_bound = (
                    mem.token_budget
                    if name == "knowledge_graph" and hasattr(mem, "token_budget")
                    else limit
                )
                items = mem.recall(
                    query, uid, recall_bound, state_changing=False
                )[:limit]
            except Exception as e:  # noqa: BLE001 - one bad memory shouldn't fail the tool
                lines.append(f"[{name}] error: {e!r}")
                continue
            if not items:
                continue
            lines.append(f"[{name}]")
            for it in items:
                lines.append(f"- {it.text}")
        return "\n".join(lines) if lines else "No matching memories."


# --------------------------------------------------------------------------- #
# get_user_facts
# --------------------------------------------------------------------------- #
class _GetUserFacts(Tool):
    name = "get_user_facts"
    description = (
        "List the top facts the character remembers about a specific user, "
        "ranked by importance. Use this to recall details about someone before "
        "responding to them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The user to look up."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
        },
        "required": ["user_id"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, user_id: str, limit: int = 8) -> str:
        from ..memory.structured import StructuredMemory  # local import avoids cycles

        mem = _agent_memories(self.agent).get("user_facts")
        if not isinstance(mem, StructuredMemory):
            return "user_facts memory is not available."
        limit = max(1, min(20, int(limit)))
        rows = mem.all_rows(user_id=user_id)
        rows.sort(key=lambda r: float(r.get("importance", 0.0)), reverse=True)
        rows = rows[:limit]
        if not rows:
            return f"No facts stored about {user_id!r}."
        return "\n".join(f"- {mem.row_text(r).strip()}" for r in rows if mem.row_text(r).strip())


# --------------------------------------------------------------------------- #
# get_user_summary
# --------------------------------------------------------------------------- #
class _GetUserSummary(Tool):
    name = "get_user_summary"
    description = (
        "Return the character's stored profile summary for a user (name, "
        "aliases, free-text summary), if one exists."
    )
    parameters = {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, user_id: str) -> str:
        from ..memory.user_summary import UserSummaryMemory

        mem = _agent_memories(self.agent).get("user_summary")
        if not isinstance(mem, UserSummaryMemory):
            return "user_summary memory is not available."
        row = mem.get_summary(user_id)
        if not row:
            return f"No summary stored about {user_id!r}."
        return mem.row_text(row).strip()


# --------------------------------------------------------------------------- #
# get_user_emotion
# --------------------------------------------------------------------------- #
class _GetUserEmotion(Tool):
    name = "get_user_emotion"
    description = (
        "Return how the character currently feels toward a user: the numeric "
        "emotion dimensions (e.g. affection, valence, trust) and the "
        "relationship descriptor, plus the baseline and current mood."
    )
    parameters = {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, user_id: str) -> str:
        from ..memory.emotion import EmotionStatus

        mem = _agent_memories(self.agent).get("emotion")
        if not isinstance(mem, EmotionStatus):
            return "emotion memory is not available."
        baseline = ", ".join(f"{k}={v:.2f}" for k, v in mem.baseline.items())
        current = ", ".join(
            f"{k}={v:.2f}" for k, v in mem.get_current_mood().items()
        )
        state = ", ".join(f"{k}={v:.2f}" for k, v in mem.get_user_state(user_id).items())
        comment = mem.get_user_comment(user_id)
        parts = [
            f"baseline: {baseline}",
            f"current mood: {current}",
            f"toward {user_id}: {state}",
        ]
        if comment:
            parts.append(f"relationship: {comment}")
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# list_known_users
# --------------------------------------------------------------------------- #
class _ListKnownUsers(Tool):
    name = "list_known_users"
    description = (
        "List the distinct users the character has memories about (across "
        "facts, directives, episodes, summaries). Use this to discover who the "
        "character knows before looking someone up."
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self) -> str:
        from ..memory.structured import StructuredMemory

        users: set[str] = set()
        for mem in _agent_memories(self.agent).values():
            if isinstance(mem, StructuredMemory):
                for r in mem.all_rows():
                    uid = r.get("user_id")
                    if uid:
                        users.add(str(uid))
        if not users:
            return "The character does not remember any specific users yet."
        return "Known users:\n" + "\n".join(f"- {u}" for u in sorted(users))
