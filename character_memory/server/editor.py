"""Write-side adapter for the memory browser.

The browser reads every memory through :mod:`.adapters`; this module is the
matching, deliberately smaller write seam.  SQLite-backed structured memories
are editable generically from their declared ``extra_columns`` while a few
well-known memories add friendlier labels and controls.  Derived RAG chunks and
the knowledge graph remain read-only: their source of truth is the character's
files and a direct edit would disappear on the next rebuild.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

from ..emotion_vectors import emotion_vector, encode_emotion_vector
from ..memory.conversation_events import ConversationEventMemory
from ..memory.emotion import EmotionStatus
from ..memory.episodic import EpisodicMemory
from ..memory.structured import StructuredMemory


_FIELD_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "user_facts": {
        "type": {"label": "Fact type", "placeholder": "preference, occupation, goal…"},
        "content": {
            "label": "Fact",
            "type": "textarea",
            "placeholder": "A self-contained fact about this user.",
        },
        "confidence": {
            "label": "Confidence",
            "type": "range",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "default": 0.5,
        },
    },
    "user_directives": {
        "content": {
            "label": "Directive",
            "type": "textarea",
            "placeholder": "A standing instruction for the character.",
        },
        "retrieval_keywords": {
            "label": "Retrieval keywords",
            "type": "tags",
            "placeholder": "formal, concise, work",
        },
    },
    "episodic": {
        "summary": {
            "label": "Episode",
            "type": "textarea",
            "placeholder": "What happened, written as a concise memory.",
        },
        "emotional_shift": {
            "label": "Emotional shift",
            "type": "json",
            "placeholder": '{"joy": 0.7, "surprise": 0.2}',
            "default": {},
        },
    },
    "heartbeat": {
        "summary": {
            "label": "Journal entry",
            "type": "textarea",
            "placeholder": "What the character discovered or did.",
        },
        "kind": {
            "label": "Entry type",
            "type": "select",
            "options": ["discovery", "action"],
            "default": "discovery",
        },
    },
    "user_summary": {
        "name": {"label": "Display name", "placeholder": "The user's name"},
        "aliases": {"label": "Aliases", "type": "tags", "placeholder": "nickname, handle"},
        "summary": {
            "label": "Profile summary",
            "type": "textarea",
            "placeholder": "A compact, consolidated profile.",
        },
    },
}


def _default_from_sql(declaration: str) -> Any:
    match = re.search(r"\bDEFAULT\s+('(?:''|[^'])*'|\S+)", declaration, re.I)
    if not match:
        return None
    raw = match.group(1)
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("''", "'")
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


def _structured_schema(memory: StructuredMemory) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    if memory.name != "heartbeat":
        fields.append(
            {
                "name": "user_id",
                "label": "User ID",
                "type": "text",
                "required": True,
                "placeholder": "user",
                "readonly_on_edit": True,
            }
        )

    # Put the primary text first, then the remaining memory-specific fields.
    names = list(memory.extra_columns)
    text_column = getattr(memory, "text_column", "content")
    if text_column in names:
        names.remove(text_column)
        names.insert(0, text_column)

    overrides = _FIELD_OVERRIDES.get(memory.name, {})
    for name in names:
        declaration = memory.extra_columns[name]
        sql_type = declaration.split()[0].upper()
        spec: dict[str, Any] = {
            "name": name,
            "label": name.replace("_", " ").title(),
            "type": "number" if sql_type in {"REAL", "INTEGER"} else "text",
            "required": "NOT NULL" in declaration.upper(),
        }
        default = _default_from_sql(declaration)
        if default is not None:
            spec["default"] = default
        spec.update(overrides.get(name, {}))
        fields.append(spec)

    fields.append(
        {
            "name": "importance",
            "label": "Importance",
            "type": "range",
            "required": True,
            "min": 0,
            "max": 1,
            "step": 0.05,
            "default": getattr(memory, "default_importance", 0.5),
        }
    )
    return {
        "editable": True,
        "fields": fields,
        "description": "Changes are stored in SQLite and the retrieval index is refreshed.",
    }


def edit_schema(memory: Any) -> dict[str, Any]:
    """Return the small JSON form schema consumed by the WebUI."""
    if isinstance(memory, ConversationEventMemory):
        return {
            "editable": False,
            "fields": [],
            "description": (
                "Conversation events are immutable source records created from "
                "chat messages."
            ),
        }
    if isinstance(memory, EmotionStatus):
        fields: list[dict[str, Any]] = [
            {
                "name": "user_id",
                "label": "User ID",
                "type": "text",
                "required": True,
                "placeholder": "user",
                "readonly_on_edit": True,
            }
        ]
        for name, default in memory.user_dims.items():
            fields.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "type": "range",
                    "required": True,
                    "min": -1,
                    "max": 1,
                    "step": 0.05,
                    "default": float(default),
                }
            )
        fields.append(
            {
                "name": "comment",
                "label": "Relationship",
                "type": "text",
                "required": False,
                "placeholder": "friend, rival, colleague…",
            }
        )
        return {
            "editable": True,
            "fields": fields,
            "description": "Values describe how the character currently feels toward this user.",
        }
    if isinstance(memory, StructuredMemory):
        return _structured_schema(memory)
    return {
        "editable": False,
        "fields": [],
        "description": "This memory is derived from source files. Edit those files in Configure, then rebuild the indexes.",
    }


def _clip(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a number, got {value!r}.") from exc
    return max(lo, min(hi, number))


def _normalise(memory: Any, values: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    schema = edit_schema(memory)
    if not schema["editable"]:
        raise ValueError(schema["description"])
    allowed = {f["name"]: f for f in schema["fields"]}
    clean: dict[str, Any] = {}
    for name, field in allowed.items():
        if name not in values:
            if not partial and field.get("required") and field.get("default") is None:
                raise ValueError(f"{field['label']} is required.")
            if not partial and "default" in field:
                clean[name] = field["default"]
            continue
        value = values[name]
        if field["type"] == "range":
            clean[name] = _clip(value, float(field["min"]), float(field["max"]))
        elif field["type"] == "number":
            clean[name] = float(value)
        elif field["type"] == "tags":
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",") if part.strip()]
            if not isinstance(value, list):
                raise ValueError(f"{field['label']} must be a list.")
            clean[name] = [str(part).strip() for part in value if str(part).strip()]
        elif field["type"] == "json":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field['label']} must be a JSON object.") from exc
            if name == "emotional_shift" and isinstance(memory, EpisodicMemory):
                clean[name] = emotion_vector(value, allowed_axes=memory.emotion_baseline)
            elif not isinstance(value, dict):
                raise ValueError(f"{field['label']} must be a JSON object.")
            else:
                clean[name] = value
        else:
            clean[name] = str(value or "").strip()
        if field.get("required") and (clean[name] == "" or clean[name] is None):
            raise ValueError(f"{field['label']} is required.")
    return clean


def _refresh_index(
    agent: Any,
    memory: StructuredMemory,
    *,
    updated_ids: tuple[int, ...] = (),
    removed_ids: tuple[int, ...] = (),
) -> None:
    try:
        memory.apply_index_changes(
            updated_ids=updated_ids,
            removed_ids=removed_ids,
        )
    except Exception as exc:  # SQLite is still the durable source of truth.
        warnings.warn(
            f"Could not refresh index for {memory.name!r}: {exc!r}; "
            "the database change was saved and the index will refresh on rebuild.",
            stacklevel=2,
        )
    try:
        agent.persist_structured()
    except Exception as exc:
        warnings.warn(f"Could not persist structured indexes: {exc!r}.", stacklevel=2)


def create_record(agent: Any, memory: Any, values: dict[str, Any]) -> Any:
    clean = _normalise(memory, values, partial=False)
    if isinstance(memory, EmotionStatus):
        user_id = clean.pop("user_id")
        comment = clean.pop("comment", "")
        memory.set_user_state(user_id, clean)
        memory.set_user_comment(user_id, comment)
        return user_id

    assert isinstance(memory, StructuredMemory)
    user_id = clean.pop("user_id", "_self")
    importance = clean.pop("importance", getattr(memory, "default_importance", 0.5))
    if isinstance(memory, EpisodicMemory):
        row_id = memory.add_episode(user_id, importance=importance, **clean)
        _refresh_index(agent, memory)
        return row_id
    if memory.name == "user_summary" and hasattr(memory, "add_or_update"):
        aliases = clean.pop("aliases", [])
        row_id = memory.add_or_update(user_id, aliases=aliases, importance=importance, **clean)
        _refresh_index(agent, memory)
        return row_id
    for key, value in list(clean.items()):
        if isinstance(value, list):
            clean[key] = json.dumps(value, ensure_ascii=False)
    row_id = memory.add(user_id, importance, **clean)
    _refresh_index(agent, memory)
    return row_id


def update_record(
    agent: Any, memory: Any, record_id: str, values: dict[str, Any]
) -> Any:
    clean = _normalise(memory, values, partial=True)
    if isinstance(memory, EmotionStatus):
        user_id = record_id
        clean.pop("user_id", None)
        comment = clean.pop("comment", None)
        state = memory.get_user_state(user_id)
        state.update(clean)
        memory.set_user_state(user_id, state)
        if comment is not None:
            memory.set_user_comment(user_id, comment)
        return user_id

    assert isinstance(memory, StructuredMemory)
    try:
        row_id = int(record_id)
    except ValueError as exc:
        raise ValueError("Structured memory record IDs must be integers.") from exc
    row = memory.get_row(row_id)
    if row is None:
        raise KeyError(record_id)
    clean.pop("user_id", None)  # record ownership is immutable from the editor
    for key, value in clean.items():
        if key == "emotional_shift" and isinstance(memory, EpisodicMemory):
            row[key] = encode_emotion_vector(value, allowed_axes=memory.emotion_baseline)
        else:
            row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    memory.update_row(row)
    _refresh_index(agent, memory, updated_ids=(row_id,))
    return row_id


def delete_record(agent: Any, memory: Any, record_id: str) -> Any:
    schema = edit_schema(memory)
    if not schema["editable"]:
        raise ValueError(schema["description"])
    if isinstance(memory, EmotionStatus):
        rows = memory.store.select(memory.table, {"user_id": record_id})
        if not rows:
            raise KeyError(record_id)
        memory.store.delete(memory.table, {"user_id": record_id})
        return record_id

    assert isinstance(memory, StructuredMemory)
    try:
        row_id = int(record_id)
    except ValueError as exc:
        raise ValueError("Structured memory record IDs must be integers.") from exc
    if memory.get_row(row_id) is None:
        raise KeyError(record_id)
    memory.delete_row(row_id)
    _refresh_index(agent, memory, removed_ids=(row_id,))
    return row_id
