"""Memory interfaces.

Every memory system subclasses `Memory`. The agent iterates the
*enabled* memories, asks each for a rendered prompt section, and concatenates
them. 
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from ..chunking import Chunk
from ..rag.base import Query

if TYPE_CHECKING:  # avoid a circular import at runtime (extract.py imports base)
    from .extract import ExtractionContext
    from ..temporal import TemporalResolution, TemporalResolutionEngine


def _relative_age(seconds: float) -> str:
    """Humanized age for `seconds` elapsed ("just now", "3 days ago", …)."""
    if seconds < 60:
        return "just now"
    steps = (
        (60, 60, "minute"),
        (3600, 24, "hour"),
        (86400, 7, "day"),
        (604800, 5, "week"),
        (2592000, 12, "month"),
    )
    for threshold, per_step, unit in steps:
        if seconds < threshold * per_step:
            n = max(1, int(seconds // threshold))
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    years = max(1, int(seconds // 31536000))
    return f"{years} year{'s' if years != 1 else ''} ago"


def format_item_timestamp(
    metadata: dict, style: str = "both", now: Optional[float] = None
) -> str:
    """Render the timestamp label for a recalled item's metadata, or "".

    Prefers ``occurred_at`` (when the thing happened — conversation events,
    world records) over ``created_at`` (when the memory was learned). The
    label is returned bare (no brackets); callers wrap it as they see fit.
    Returns "" when no timestamp is available or `style` is "none"/unknown,
    so memories without timestamps render exactly as before.

    `style`: "absolute" (short local date+time, like dedup's judge prompts),
    "relative" (humanized age), "both" (absolute with the age in parentheses).
    """
    ts = metadata.get("occurred_at") or metadata.get("created_at")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return ""
    parts: list[str] = []
    if style in ("absolute", "both"):
        parts.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)))
    if style in ("relative", "both"):
        parts.append(_relative_age(max(0.0, (now if now is not None else time.time()) - ts)))
    if not parts:
        return ""
    if len(parts) == 2:
        return f"{parts[0]} ({parts[1]})"
    return parts[0]


def item_bullet(it: "MemoryItem", style: str) -> str:
    """One `- {text}` bullet, stamped with `[ {timestamp} ]` when available.

    Shared by :meth:`Memory.format` / :meth:`Memory.format_grouped` and the
    knowledge-graph renderer so every prompt section stamps alike.
    """
    label = format_item_timestamp(it.metadata, style)
    return f"- {it.text} [{label}]" if label else f"- {it.text}"


class MemoryScope(str, Enum):
    """How a memory relates to the participants of a conversation.

    Memories stay single-user at their core (:meth:`Memory.recall` always takes
    one ``user_id``). The *scope* is a declaration each memory makes about
    itself; the orchestrator uses it to decide how to fan recall out across a
    multi-participant (group) chat:

    * ``PER_USER`` (default): recall once per participant and group the results
      by speaker. Used by facts / directives / episodic / emotion — anything
      that stores information about a specific person.
    * ``CHARACTER``: recall once and ignore participants entirely. Used by
      memories that are about the character, not the user (the wiki, example
      dialogues, the heartbeat journal).

    Keeping this as a per-memory declaration (rather than hard-coding it in the
    agent) lets each module describe its own behaviour, so new memories opt in
    or out of multi-user handling without touching the orchestrator.
    """

    PER_USER = "per_user"
    CHARACTER = "character"

@dataclass
class MemoryItem:
    """A single recalled fact/snippet ready for the prompt."""

    text: str
    score: float = 0.0
    kind: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RecallResult:
    """The result of one memory recall and its prompt rendering.

    ``diagnostics`` is intentionally backend-defined.  It lets observability
    clients inspect a retrieval without asking the memory to recall a second
    time; normal callers can ignore it.
    """

    items: list[MemoryItem] = field(default_factory=list)
    body: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionSpec:
    """How a memory is populated by LLM extraction.

    A memory that wants to learn from the conversation returns one of these
    from :meth:`Memory.extraction_spec`. The agent gathers the specs of every
    *enabled* memory, builds a single combined JSON schema + instruction out of
    them, runs extraction, then hands each memory the value it asked for via
    :meth:`Memory.apply_extraction`.

    - `field`: the key this memory owns in the extraction result
      (e.g. `"facts"`).
    - `schema`: the JSON-schema fragment for that field (the value you'd put
      under `properties.<field>`).
    - `instruction`: the human-readable bullet describing this field to the
      LLM (what to extract, value ranges, …).
    - `per_user`: when True, the memory's items are attributed to specific
      participants in a group chat. In multi-user mode the extractor augments
      the item schema with a ``user_id`` enum (the participants) and the LLM is
      asked to stamp each item with the participant it is about; in single-user
      mode this flag is a no-op (the chat's one user is used). Per-user
      memories set it True; character-scoped memories leave it False.
    - `snapshot`: when True, the field is a complete replacement snapshot
      rather than a delta of newly discovered items. The extraction prompt
      adds a stronger merge/preservation rule after the generic delta footer.
    """

    field: str
    schema: dict[str, Any]
    instruction: str
    per_user: bool = False
    snapshot: bool = False


class Memory(ABC):
    """Base class for all memory systems."""

    name: str = "memory"

    #: How this memory relates to conversation participants. See
    #: :class:`MemoryScope`. Per-user memories keep the default (``PER_USER``);
    #: character-scoped memories (wiki, example dialogues, heartbeat) override
    #: it to ``CHARACTER``. The orchestrator reads this to decide how to fan
    #: recall out across a multi-participant chat.
    scope: MemoryScope = MemoryScope.PER_USER

    #: How recalled items are timestamped in the prompt. One of "none",
    #: "absolute" (short local date+time), "relative" (humanized age) or
    #: "both"; see :func:`format_item_timestamp`. Items whose metadata carries
    #: no timestamp are rendered unchanged. The agent sets this from
    #: :attr:`MemoryConfig.timestamp_style`; direct ``Character(memories=...)``
    #: callers keep this default or set it themselves.
    timestamp_style: str = "both"
    # The orchestrator only forwards temporal kwargs to opt-in memories. This
    # preserves source compatibility with third-party Memory subclasses.
    supports_temporal_resolution: bool = False

    def __init__(self, *, enabled: bool = True, name: Optional[str] = None) -> None:
        self.enabled = enabled
        self.temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None
        self.temporal_resolution_timezone: str = "UTC"
        self.temporal_resolution_weight: float = 1.0
        if name is not None:
            self.name = name

    def resolve_temporal(
        self,
        query: Query,
        temporal_resolution: bool | "TemporalResolution" | None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
    ) -> Optional["TemporalResolution"]:
        """Normalize recall's public temporal arguments, failing open."""
        if not temporal_resolution:
            return None
        from ..temporal import TemporalResolution

        if isinstance(temporal_resolution, TemporalResolution):
            return temporal_resolution
        engine = temporal_resolution_engine or self.temporal_resolution_engine
        if engine is None:
            return None
        try:
            return engine.resolve(query, timezone=self.temporal_resolution_timezone)
        except Exception:
            # Temporal matching is ranking enrichment: parser/backend failures
            # must retain the legacy semantic results.
            return None

    def _recall_with_temporal(
        self,
        query: Query,
        user_id: str,
        limit: int,
        *,
        state_changing: bool,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> list[MemoryItem]:
        if self.supports_temporal_resolution:
            return self.recall(
                query,
                user_id,
                limit,
                state_changing=state_changing,
                temporal_resolution=temporal_resolution,
                temporal_resolution_engine=temporal_resolution_engine,
                temporal_weight=(
                    self.temporal_resolution_weight
                    if temporal_weight is None
                    else temporal_weight
                ),
            )
        return self.recall(query, user_id, limit, state_changing=state_changing)

    @abstractmethod
    def recall(
        self, query: Query, user_id: str, limit: int, state_changing: bool = True
    ) -> list[MemoryItem]:
        """Return results within ``limit`` for ``query`` and ``user_id``.

        For most memories ``limit`` is a top-k item count. ``0`` disables
        automatic prompt retrieval for this memory; MCP and tools still
        search with their own limits. Memories whose entries vary
        substantially in size may define another unit; the knowledge graph
        treats it as a rendered prompt-token budget.

        `query` is normally the last user message, but may be a list of
        ``(text, weight)`` pairs (e.g. one per recent chat message, with older
        ones weighted less). Backends that search fuse the weighted queries;
        backends that ignore the query (e.g. emotion) accept it unchanged.

        When `state_changing` is `False`, the recall is read-only: memories
        must not mutate any bookkeeping (e.g. recall-count bumps or
        last-recalled timestamps). Useful for inspection / context preview
        without skewing decay statistics.
        """

    @abstractmethod
    def build(self, info_chunks:list[Chunk]) -> None:
        """Build the memory from the given `info_chunks`."""

    @abstractmethod 
    def persist(self, path: str) -> None:
        """Persist the memory to `path`."""

    @abstractmethod 
    def load(self, path: str) -> None:
        """Load the memory from `path`."""

    @property
    def title(self) -> str:
        """Header used when this memory's section is rendered."""
        return self.name.replace("_", " ").title()

    def format(self, items: list[MemoryItem]) -> str:
        """Render recalled `items` into a prompt fragment (override me)."""
        return "\n".join(item_bullet(it, self.timestamp_style) for it in items)

    def build_section(
        self,
        query: Query,
        user_id: str,
        limit: int,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> Optional[str]:
        """Recall (if enabled) and format; `None` when there is nothing to show.

        `state_changing` is forwarded to :meth:`recall`.
        """
        return self.build_section_result(
            query,
            user_id,
            limit,
            state_changing=state_changing,
            temporal_resolution=temporal_resolution,
            temporal_resolution_engine=temporal_resolution_engine,
            temporal_weight=temporal_weight,
        ).body

    def build_section_result(
        self,
        query: Query,
        user_id: str,
        limit: int,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> RecallResult:
        """Recall and render once, retaining the actual items for inspection."""
        items = (
            self._recall_with_temporal(
                query,
                user_id,
                limit,
                state_changing=state_changing,
                temporal_resolution=temporal_resolution,
                temporal_resolution_engine=temporal_resolution_engine,
                temporal_weight=temporal_weight,
            )
            if self.enabled
            else []
        )
        return RecallResult(items=items, body=self.format(items) if items else None)

    # Multi-participant (group chat) handling --------------------------------
    # These are orchestration conveniences layered on top of the single-user
    # :meth:`recall`. A memory's :attr:`scope` decides how they fan recall out:
    # PER_USER recalls once per participant, CHARACTER recalls once. Each method
    # falls back to the single-user path when there is just one participant, so
    # a 1:1 chat is bit-for-bit identical to the legacy rendering.
    def recall_participants(
        self,
        query: Query,
        participants: list[str],
        limit: int,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> list[MemoryItem]:
        """Recall items for the conversation's participants.

        * ``PER_USER`` scope: recall once per participant (each gets its own
          ``limit`` bound); the speaker is carried in each item's metadata.
        * ``CHARACTER`` scope: recall once (the memory is not about any one
          participant); the first participant is passed to :meth:`recall` as a
          dummy ``user_id`` and ignored by the implementation.

        The single participant is the fast path: a plain ``recall`` call.
        """
        if not participants:
            return []
        if len(participants) == 1:
            return self._recall_with_temporal(
                query,
                participants[0],
                limit,
                state_changing=state_changing,
                temporal_resolution=temporal_resolution,
                temporal_resolution_engine=temporal_resolution_engine,
                temporal_weight=temporal_weight,
            )
        if self.scope is MemoryScope.CHARACTER:
            return self._recall_with_temporal(
                query,
                participants[0],
                limit,
                state_changing=state_changing,
                temporal_resolution=temporal_resolution,
                temporal_resolution_engine=temporal_resolution_engine,
                temporal_weight=temporal_weight,
            )
        items: list[MemoryItem] = []
        for uid in participants:
            items.extend(
                self._recall_with_temporal(
                    query,
                    uid,
                    limit,
                    state_changing=state_changing,
                    temporal_resolution=temporal_resolution,
                    temporal_resolution_engine=temporal_resolution_engine,
                    temporal_weight=temporal_weight,
                )
            )
        return items

    def format_grouped(
        self, items: list[MemoryItem], participants: list[str]
    ) -> str:
        """Render recalled `items` grouped by participant.

        Used by PER_USER memories in a group chat. Each item carries the
        speaker it belongs to under ``metadata['user_id']``. Items with no
        speaker (or an unknown one) are rendered under a generic block. The
        default implementation renders ``About {name}:\n- ...`` per
        participant; EmotionStatus overrides it to fold the baseline in once.
        """
        by_user: dict[str, list[MemoryItem]] = {}
        unattributed: list[MemoryItem] = []
        for it in items:
            uid = it.metadata.get("user_id")
            if isinstance(uid, str) and uid:
                by_user.setdefault(uid, []).append(it)
            else:
                unattributed.append(it)
        blocks: list[str] = []
        # Render known participants first, in the order they were given, so the
        # layout is stable across turns; any leftover speakers trail after.
        order = [u for u in participants if u in by_user]
        order += [u for u in by_user if u not in order]
        for uid in order:
            body = "\n".join(item_bullet(it, self.timestamp_style) for it in by_user[uid])
            blocks.append(f"About {uid}:\n{body}")
        if unattributed:
            body = "\n".join(item_bullet(it, self.timestamp_style) for it in unattributed)
            blocks.append(body)
        return "\n\n".join(blocks)

    def build_section_participants(
        self,
        query: Query,
        participants: list[str],
        limit: int,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> Optional[str]:
        """Recall + format for a multi-participant conversation.

        For a single participant this delegates to :meth:`build_section`
        (identical to the legacy single-user path). For several participants it
        recalls for everyone and renders grouped or flat depending on scope:
        CHARACTER memories are formatted with :meth:`format` (their items are
        not per-user); PER_USER memories are formatted with
        :meth:`format_grouped`.
        """
        return self.build_section_participants_result(
            query,
            participants,
            limit,
            state_changing=state_changing,
            temporal_resolution=temporal_resolution,
            temporal_resolution_engine=temporal_resolution_engine,
            temporal_weight=temporal_weight,
        ).body

    def build_section_participants_result(
        self,
        query: Query,
        participants: list[str],
        limit: int,
        state_changing: bool = True,
        *,
        temporal_resolution: bool | "TemporalResolution" | None = None,
        temporal_resolution_engine: Optional["TemporalResolutionEngine"] = None,
        temporal_weight: Optional[float] = None,
    ) -> RecallResult:
        """Group-aware equivalent of :meth:`build_section_result`."""
        if not participants:
            return RecallResult()
        if len(participants) == 1:
            return self.build_section_result(
                query,
                participants[0],
                limit,
                state_changing=state_changing,
                temporal_resolution=temporal_resolution,
                temporal_resolution_engine=temporal_resolution_engine,
                temporal_weight=temporal_weight,
            )
        if not self.enabled:
            return RecallResult()
        items = self.recall_participants(
            query,
            participants,
            limit,
            state_changing=state_changing,
            temporal_resolution=temporal_resolution,
            temporal_resolution_engine=temporal_resolution_engine,
            temporal_weight=temporal_weight,
        )
        if not items:
            return RecallResult(items=[])
        body = self.format(items) if self.scope is MemoryScope.CHARACTER else self.format_grouped(items, participants)
        return RecallResult(items=items, body=body)

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        """Return every memory stored in this memory's backend.

        `limit=0` (the default) returns everything; a positive `limit` caps
        the result. Memories without a backing store return an empty list.
        Override in subclasses that own a database or index.
        """
        return []

    # Extraction
    def extraction_spec(
        self, context: Optional["ExtractionContext"] = None
    ) -> Optional[ExtractionSpec]:
        """What this memory wants extracted from the conversation, or `None`.

        Returning `None` (the default) opts the memory out of extraction
        entirely. The agent only consults *enabled* memories, so a disabled
        memory is never asked to extract.

        ``context`` (when provided) carries the character/user names and known
        facts so a memory can phrase its instruction bullet in the character's
        point of view and reinforce the full-sentence rule. It is optional and
        ignored by memories that don't need it.
        """
        return None

    def apply_extraction(
        self, value: Any, user_id: str, *, chat_id: Optional[str] = None
    ) -> list[MemoryItem]:
        """Consume this memory's portion of an extraction result.

        `value` is whatever the LLM produced under this memory's `field`
        (a list, a dict, … depending on the schema). Returns the items that
        were actually added to the memory, so callers (e.g. a deduplicator)
        can act on the freshly-written rows. No-op implementations return `[]`.

        In multi-user (group chat) extraction each per-user item may carry its
        own ``user_id``; per-user memories honour ``item['user_id']`` and fall
        back to the caller's ``user_id`` when it is absent. ``user_id`` here is
        the chat's default (the owner / current speaker).

        ``chat_id`` (optional) is the conversation the extraction ran over.
        Memories that are chat-scoped (``user_facts``, ``episodic``) stamp it
        onto their rows so the knowledge graph can link facts and episodes
        learned in the same chat; other memories accept and ignore it.
        """
        return []
