"""Emotion status: a user-independent baseline plus configurable per-user dims."""

import json
from typing import TYPE_CHECKING, Any, Optional

from .base import ExtractionSpec, Memory, MemoryItem, MemoryScope
from ..chunking import Chunk
from .store import SQLiteStore

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext


class EmotionStatus(Memory):
    """Tracks how the character feels.

    A fixed `baseline` (user-independent: `joy`, `sadness`…) describes the
    character's resting state. On top of that, a set of per-user dimensions
    (default `affection`, `valence`, `trust` - configurable) tracks how
    the character feels *toward each user*, plus a per-user `comment`: a short
    relationship descriptor (colleague / friend / conflicting / …). The
    extractor only writes `comment` when the conversation establishes or
    changes the relationship, so a stable label is preserved across turns.
    """

    name = "emotion"

    def __init__(
        self,
        store: SQLiteStore,
        *,
        enabled: bool = True,
        baseline: Optional[dict[str, float]] = None,
        user_dims: Optional[dict[str, float]] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(enabled=enabled, name=name)
        self.store = store
        self.baseline = dict(baseline or {"neutral": 0.5, "joy": 0.2, "sadness": 0.1, "anger": 0, "anxiety": 0})
        self.user_dims = dict(user_dims or {"affection": 0.0, "valence": 0.0, "trust": 0.0})
        self.table = "emotion"
        self.store.create_table(
            self.table,
            {"user_id": "TEXT PRIMARY KEY", "state": "TEXT NOT NULL"},
        )

    # State
    # The per-user state is one JSON blob per user holding the numeric dims
    # (affection, valence, trust, …) plus a string `comment` describing the
    # relationship (colleague / friend / conflicting / …). Dims are floats
    # clamped to [-1, 1] and updated by signed deltas; `comment` is a string
    # set wholesale and only when the relationship changes. The accessors below
    # keep the two concerns separate so the dim API stays cleanly float-typed.

    def _default_user_state(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.user_dims.items()}

    def _read_blob(self, user_id: str) -> dict[str, Any]:
        """Full stored blob for `user_id` (dims + comment), or defaults."""
        rows = self.store.select(self.table, {"user_id": user_id})
        if rows:
            try:
                blob = json.loads(rows[0]["state"])
                if isinstance(blob, dict):
                    # Merge over the configured dims so new dims always appear,
                    # and tolerate legacy blobs without a `comment` key.
                    merged = {**self._default_user_state(), **blob}
                    if "comment" not in merged:
                        merged["comment"] = ""
                    return merged
            except (TypeError, ValueError):
                pass
        blob = self._default_user_state()
        blob["comment"] = ""
        return blob

    def _write_blob(self, user_id: str, blob: dict[str, Any]) -> None:
        self.store.upsert(
            self.table, {"user_id": user_id, "state": json.dumps(blob)}, pk="user_id"
        )

    def get_user_state(self, user_id: str) -> dict[str, float]:
        """Per-user numeric dims for `user_id` (no `comment`)."""
        blob = self._read_blob(user_id)
        return {k: float(v) for k, v in blob.items() if k != "comment"}

    def set_user_state(self, user_id: str, state: dict[str, float]) -> None:
        """Replace the dims, preserving the existing `comment`."""
        blob = self._read_blob(user_id)
        blob.update({k: float(v) for k, v in state.items()})
        self._write_blob(user_id, blob)

    def get_user_comment(self, user_id: str) -> str:
        """The relationship descriptor for `user_id` ("" when unset)."""
        return str(self._read_blob(user_id).get("comment") or "")

    def set_user_comment(self, user_id: str, comment: str) -> None:
        """Set the relationship descriptor, preserving the dims."""
        blob = self._read_blob(user_id)
        blob["comment"] = str(comment or "").strip()
        self._write_blob(user_id, blob)

    def update(self, user_id: str, deltas: dict[str, float]) -> dict[str, float]:
        """Apply signed `deltas` to the per-user dims, clamped to `[-1, 1]`."""
        current = self.get_user_state(user_id)
        for k, v in deltas.items():
            if k in current:
                current[k] = max(-1.0, min(1.0, current[k] + float(v)))
        self.set_user_state(user_id, current)
        return current

    # Recall
    def recall(self, query: str, user_id: str, limit: int, state_changing: bool = True) -> list[MemoryItem]:
        # Emotion recall never mutates state; `state_changing` is accepted
        # for interface symmetry but has no effect.
        state = self.get_user_state(user_id)
        parts = [f"{k}={v:.2f}" for k, v in self.baseline.items()]
        parts += [f"{k}(toward {user_id})={v:.2f}" for k, v in state.items()]
        comment = self.get_user_comment(user_id)
        if comment:
            parts.append(f"relationship(toward {user_id})={comment}")
        return [MemoryItem(text=p, score=1.0, kind=self.name) for p in parts]

    # Multi-participant recall ------------------------------------------------
    # The baseline is user-independent, so in a group chat we surface it once
    # and then append each participant's per-user dims. This is the only
    # PER_USER memory whose recall needs special handling, because the baseline
    # would otherwise be duplicated per participant.
    def recall_participants(
        self,
        query: str,
        participants: list[str],
        limit: int,
        state_changing: bool = True,
    ) -> list[MemoryItem]:
        if not participants:
            return []
        if len(participants) == 1:
            return self.recall(query, participants[0], limit, state_changing=state_changing)
        items: list[MemoryItem] = []
        # Baseline once.
        for k, v in self.baseline.items():
            items.append(
                MemoryItem(text=f"{k}={v:.2f}", score=1.0, kind=self.name,
                           metadata={"emotion": "baseline"})
            )
        # Each participant's per-user dims + relationship comment, tagged with
        # the user for grouping.
        for uid in participants:
            state = self.get_user_state(uid)
            for k, v in state.items():
                items.append(
                    MemoryItem(
                        text=f"{k}={v:.2f}", score=1.0, kind=self.name,
                        metadata={"user_id": uid, "emotion": "user"},
                    )
                )
            comment = self.get_user_comment(uid)
            if comment:
                items.append(
                    MemoryItem(
                        text=f"relationship={comment}", score=1.0, kind=self.name,
                        metadata={"user_id": uid, "emotion": "comment"},
                    )
                )
        return items

    def format_grouped(self, items: list[MemoryItem], participants: list[str]) -> str:
        """Baseline once, then a per-participant block of their dims."""
        baseline: list[MemoryItem] = []
        by_user: dict[str, list[MemoryItem]] = {}
        for it in items:
            uid = it.metadata.get("user_id")
            if isinstance(uid, str) and uid:
                by_user.setdefault(uid, []).append(it)
            else:
                baseline.append(it)
        blocks: list[str] = []
        if baseline:
            blocks.append("Baseline:\n" + "\n".join(f"- {it.text}" for it in baseline))
        order = [u for u in participants if u in by_user]
        order += [u for u in by_user if u not in order]
        for uid in order:
            body = "\n".join(f"- {it.text}" for it in by_user[uid])
            blocks.append(f"Toward {uid}:\n{body}")
        return "\n\n".join(blocks)

    def get_memories(self, limit: int = 0) -> list[MemoryItem]:
        """Every stored per-user emotion state, one item per user.

        `limit=0` returns all users; otherwise the first `limit` rows. Only rows
        actually persisted in the database are returned (the baseline is not stored).
        """
        rows = self.store.select(self.table, order_by="user_id")
        if limit and limit > 0:
            rows = rows[:limit]
        return [
            MemoryItem(text=f"{r['user_id']}: {r['state']}", score=1.0, kind=self.name, metadata=dict(r))
            for r in rows
        ]

    # Implement not needed methods
    def build(self, info_chunks: list[Chunk]) -> None:
        return None

    def load(self, path: str) -> None:
        return None

    def persist(self, path: str) -> None:
        return None

    # Extraction ----------------------------------------------------------
    # The relationship `comment` (colleague / friend / conflicting / …) is a
    # coarse label, not a delta. To avoid the extractor rewriting it every
    # turn with synonyms, it is only emitted when the conversation *changes*
    # the relationship (first time it's named, or a genuine shift). The dims
    # remain signed deltas as before.
    _COMMENT_NOTE = (
        "`comment` is an optional short relationship label for {who} "
        "(e.g. colleague, friend, rival, conflicting, mentor, partner). "
        "Set it ONLY when this exchange establishes or clearly changes the "
        "relationship; omit the key entirely otherwise so a stable label is "
        "preserved. Never paraphrase an already-fitting label."
    )

    def extraction_spec(self, context: "ExtractionContext | None" = None) -> ExtractionSpec:
        dims = ", ".join(self.user_dims) or "affection, valence, trust"
        char = context.character_name if context else "the character"
        user = context.user_name if context else "the user"
        participants = getattr(context, "participants", None) or []
        if participants and len(participants) > 1:
            # Group chat: one signed-delta object (+ optional comment) per
            # participant. The ``user_id`` enum is injected by the extractor
            # builder.
            return ExtractionSpec(
                field="emotion_deltas",
                per_user=True,
                schema={
                    "type": "array",
                    "description": (
                        f"Signed adjustments to {char}'s per-user emotion dims, "
                        f"one entry per participant. Allowed dims: {dims}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "deltas": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "comment": {"type": "string"},
                        },
                        "required": ["user_id", "deltas"],
                    },
                },
                instruction=(
                    f"- emotion_deltas: small signed adjustments to {char}'s feelings "
                    f"toward each of the participants ({', '.join(participants)}), based "
                    f"on what just happened. One entry per participant the exchange was "
                    f"about; omit participants with no shift. Allowed dims: {dims}. "
                    + self._COMMENT_NOTE.format(who="a participant")
                ),
            )
        return ExtractionSpec(
            field="emotion_deltas",
            schema={
                "type": "object",
                "description": (
                    f"Signed adjustments to {char}'s per-user emotion dims "
                    f"(any of: {dims}), plus an optional relationship `comment`."
                ),
                "properties": {
                    "comment": {"type": "string"},
                },
                "additionalProperties": {"type": "number"},
            },
            instruction=(
                f"- emotion_deltas: small signed adjustments to {char}'s feelings "
                f"toward {user}, based on what just happened. Allowed dims: {dims}. "
                + self._COMMENT_NOTE.format(who=user)
            ),
        )

    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        if not value:
            return []
        # Multi-user: a list of {user_id, deltas, comment?}. Apply each entry
        # to its own user; the comment is written only when provided (a stable
        # label is otherwise preserved).
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                uid = str(entry.get("user_id") or user_id)
                deltas = entry.get("deltas")
                if isinstance(deltas, dict):
                    self.update(uid, deltas)
                comment = entry.get("comment")
                if isinstance(comment, str) and comment.strip():
                    self.set_user_comment(uid, comment)
            return []
        # Single-user: a {dim: delta, comment?} object applied to the caller's
        # user. `update` preserves the comment; the comment is written only
        # when present.
        if isinstance(value, dict):
            comment = value.get("comment")
            deltas = {k: v for k, v in value.items() if k != "comment"}
            if deltas:
                self.update(user_id, deltas)
            if isinstance(comment, str) and comment.strip():
                self.set_user_comment(user_id, comment)
        return []
