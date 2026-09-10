"""Emotion status: baseline/current mood plus configurable per-user dims."""

from ..concurrency import synchronized

import json
from typing import TYPE_CHECKING, Any, Optional

from .base import ExtractionSpec, Memory, MemoryItem, MemoryScope
from ..chunking import Chunk
from ..emotion_vectors import decode_emotion_vector, emotion_vector, encode_emotion_vector
from .store_base import Store

if TYPE_CHECKING:  # avoid circular import at runtime
    from .extract import ExtractionContext


class EmotionStatus(Memory):
    """Tracks how the character feels.

    A fixed `baseline` (user-independent: `joy`, `sadness`…) describes the
    character's resting state. A persisted `current_mood` uses the same axes
    and is replaced by each extraction pass. On top of that, per-user dimensions
    (default `affection`, `valence`, `trust` - configurable) tracks how
    the character feels *toward each user*, plus a per-user `comment`: a short
    relationship descriptor (colleague / friend / conflicting / …). The
    extractor only writes `comment` when the conversation establishes or
    changes the relationship, so a stable label is preserved across turns.
    """

    name = "emotion"

    def __init__(
        self,
        store: Store,
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
        self.state_table = "emotion_state"
        self.store.create_table(
            self.table,
            {"user_id": "TEXT PRIMARY KEY", "state": "TEXT NOT NULL"},
        )
        self.store.create_table(
            self.state_table,
            {"key": "TEXT PRIMARY KEY", "state": "TEXT NOT NULL"},
        )
        if not self.store.select(self.state_table, {"key": "current_mood"}):
            self.set_current_mood(self.baseline)

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

    @synchronized
    def set_user_state(self, user_id: str, state: dict[str, float]) -> None:
        """Replace the dims, preserving the existing `comment`."""
        blob = self._read_blob(user_id)
        blob.update({k: float(v) for k, v in state.items()})
        self._write_blob(user_id, blob)

    def get_user_comment(self, user_id: str) -> str:
        """The relationship descriptor for `user_id` ("" when unset)."""
        return str(self._read_blob(user_id).get("comment") or "")

    @synchronized
    def set_user_comment(self, user_id: str, comment: str) -> None:
        """Set the relationship descriptor, preserving the dims."""
        blob = self._read_blob(user_id)
        blob["comment"] = str(comment or "").strip()
        self._write_blob(user_id, blob)

    @synchronized
    def update(self, user_id: str, deltas: dict[str, float]) -> dict[str, float]:
        """Apply signed `deltas` to the per-user dims, clamped to `[-1, 1]`."""
        current = self.get_user_state(user_id)
        for k, v in deltas.items():
            if k in current:
                current[k] = max(-1.0, min(1.0, current[k] + float(v)))
        self.set_user_state(user_id, current)
        return current

    def get_current_mood(self) -> dict[str, float]:
        """Return the persisted character-wide mood, or the baseline initially."""
        rows = self.store.select(self.state_table, {"key": "current_mood"})
        if not rows:
            return emotion_vector(self.baseline, allowed_axes=self.baseline)
        clean = decode_emotion_vector(rows[0]["state"], allowed_axes=self.baseline)
        return {axis: float(clean.get(axis, 0.0)) for axis in self.baseline}

    @synchronized
    def set_current_mood(self, mood: dict[str, float]) -> dict[str, float]:
        """Replace the character-wide mood with an absolute snapshot."""
        clean = emotion_vector(mood, allowed_axes=self.baseline)
        # An absolute snapshot has every configured axis; omissions mean zero.
        full = {axis: float(clean.get(axis, 0.0)) for axis in self.baseline}
        self.store.upsert(
            self.state_table,
            {
                "key": "current_mood",
                "state": encode_emotion_vector(full, allowed_axes=self.baseline),
            },
            pk="key",
        )
        return full

    # Recall
    def recall(self, query: str, user_id: str, limit: int, state_changing: bool = True) -> list[MemoryItem]:
        # Emotion recall never mutates state; `state_changing` is accepted
        # for interface symmetry but has no effect.
        state = self.get_user_state(user_id)
        parts = [f"baseline.{k}={v:.2f}" for k, v in self.baseline.items()]
        parts += [f"current.{k}={v:.2f}" for k, v in self.get_current_mood().items()]
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
        # Baseline and current mood once.
        for k, v in self.baseline.items():
            items.append(
                MemoryItem(text=f"baseline.{k}={v:.2f}", score=1.0, kind=self.name,
                           metadata={"emotion": "baseline"})
            )
        for k, v in self.get_current_mood().items():
            items.append(
                MemoryItem(text=f"current.{k}={v:.2f}", score=1.0, kind=self.name,
                           metadata={"emotion": "current"})
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
        mood_axes = list(self.baseline)
        char = context.character_name if context else "the character"
        user = context.user_name if context else "the user"
        participants = getattr(context, "participants", None) or []
        users = participants or [user]
        mood_properties = {
            axis: {"type": "number", "minimum": 0, "maximum": 1}
            for axis in mood_axes
        }
        return ExtractionSpec(
            field="emotion_deltas",
            schema={
                "type": "object",
                "description": (
                    f"{char}'s absolute current mood plus per-user relationship deltas."
                ),
                "properties": {
                    "current_mood": {
                        "type": "object",
                        "properties": mood_properties,
                        "required": mood_axes,
                        "additionalProperties": False,
                    },
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string", "enum": users},
                                "deltas": {
                                    "type": "object",
                                    "properties": {
                                        dim: {"type": "number"} for dim in self.user_dims
                                    },
                                    "additionalProperties": False,
                                },
                                "comment": {"type": "string"},
                            },
                            "required": ["user_id", "deltas"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["current_mood", "users"],
                "additionalProperties": False,
            },
            instruction=(
                f"- emotion_deltas: set current_mood to {char}'s complete absolute mood "
                f"after this exchange using 0..1 axes: {', '.join(mood_axes)}. In users, "
                f"include only people whose relationship state shifted, with small signed "
                f"adjustments on: {dims}. "
                + self._COMMENT_NOTE.format(who="that user")
            ),
        )

    @synchronized
    def apply_extraction(self, value: Any, user_id: str, *, chat_id: Optional[str] = None) -> list[MemoryItem]:
        if not value:
            return []
        if not isinstance(value, dict):
            return []
        mood = value.get("current_mood")
        changed = False
        if isinstance(mood, dict):
            # Structured-output support varies across OpenAI-compatible
            # models. Keep the public setter strict, but do not let a model's
            # invented axis (for example ``annoyance``) abort every memory
            # extracted from the turn.
            clean_mood = emotion_vector(
                mood,
                allowed_axes=self.baseline,
                ignore_unknown_axes=True,
            )
            # If every emitted axis was invented, preserve the prior mood
            # rather than interpreting the filtered empty object as an
            # absolute all-zero snapshot. A genuinely empty object retains
            # the existing public semantics and resets configured axes.
            if clean_mood or not mood:
                self.set_current_mood(clean_mood)
                changed = True
        changed_users: list[str] = []
        for entry in value.get("users") or []:
            if not isinstance(entry, dict):
                continue
            uid = str(entry.get("user_id") or user_id)
            deltas = entry.get("deltas")
            if isinstance(deltas, dict) and deltas:
                self.update(uid, deltas)
                changed = True
            comment = entry.get("comment")
            if isinstance(comment, str) and comment.strip():
                self.set_user_comment(uid, comment)
                changed = True
            if uid not in changed_users:
                changed_users.append(uid)
        if not changed:
            return []
        return [
            MemoryItem(
                text="emotion state updated",
                score=1.0,
                kind=self.name,
                metadata={"users": changed_users, "current_mood": self.get_current_mood()},
            )
        ]
