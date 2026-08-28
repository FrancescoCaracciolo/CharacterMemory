"""Synchronous client for the CharacterMemory HTTP server.

The bundled server deliberately keeps the model call on the client side.  A
typical turn therefore consists of :meth:`CharacterMemoryClient.context`, a
caller-owned LLM request, and :meth:`CharacterMemoryClient.save`.

This module uses :mod:`urllib` rather than a third-party HTTP library so it is
available with the core package.  It does not import the optional FastAPI
server package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.client import HTTPResponse
from typing import Any, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .memory.base import MemoryItem


class CharacterMemoryClientError(RuntimeError):
    """Base exception raised by :class:`CharacterMemoryClient`."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class CharacterMemoryHTTPError(CharacterMemoryClientError):
    """An HTTP response whose status code indicates failure.

    ``detail`` contains the server's JSON ``detail`` value when available;
    otherwise it contains the decoded response body.
    """


@dataclass
class ContextResponse:
    """Context returned by ``POST /context``.

    ``context_order`` is authoritative when assembling a prompt.  The server
    also supplies ``context_text`` for callers that want the already-rendered
    prompt section.  ``memories`` contains the exact retrieved memory items
    used for each rendered memory section, grouped by memory name.
    """

    chat_id: str
    context: dict[str, str]
    context_order: list[str]
    context_text: str
    memories: dict[str, list[MemoryItem]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextResponse":
        """Build a response object from a decoded server payload."""
        try:
            chat_id = str(payload["chat_id"])
            raw_context = payload["context"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise CharacterMemoryClientError(
                "Invalid /context response: expected chat_id and context."
            ) from exc

        if not isinstance(raw_context, Mapping):
            raise CharacterMemoryClientError(
                "Invalid /context response: context must be an object."
            )
        context = {str(key): str(value) for key, value in raw_context.items()}

        raw_order = payload.get("context_order")
        if raw_order is None:
            # Compatibility with early server responses that only returned the
            # context mapping. Current servers always send context_order.
            context_order = list(context)
        elif isinstance(raw_order, list):
            context_order = [str(item) for item in raw_order]
        else:
            raise CharacterMemoryClientError(
                "Invalid /context response: context_order must be a list."
            )

        raw_text = payload.get("context_text")
        context_text = (
            str(raw_text)
            if raw_text is not None
            else "\n\n".join(
                context[item] for item in context_order if item in context
            )
        )

        raw_memories = payload.get("memories")
        memories: dict[str, list[MemoryItem]] = {}
        if raw_memories is not None:
            if not isinstance(raw_memories, Mapping):
                raise CharacterMemoryClientError(
                    "Invalid /context response: memories must be an object."
                )
            for section, items in raw_memories.items():
                if not isinstance(items, list):
                    raise CharacterMemoryClientError(
                        f"Invalid /context response: memories['{section}'] must be a list."
                    )
                section_items: list[MemoryItem] = []
                for it in items:
                    if isinstance(it, MemoryItem):
                        section_items.append(it)
                    elif isinstance(it, Mapping):
                        section_items.append(
                            MemoryItem(
                                text=str(it.get("text", "")),
                                score=float(it.get("score", 0.0)),
                                kind=str(it.get("kind", "")),
                                metadata=dict(it.get("metadata") or {}),
                            )
                        )
                    else:
                        raise CharacterMemoryClientError(
                            f"Invalid /context response: items in memories['{section}'] must be objects."
                        )
                memories[str(section)] = section_items

        return cls(
            chat_id=chat_id,
            context=context,
            context_order=context_order,
            context_text=context_text,
            memories=memories,
        )


@dataclass
class SaveResponse:
    """Result returned by ``POST /save``."""

    ok: bool
    chat_id: str
    extracted: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SaveResponse":
        """Build a response object from a decoded server payload."""
        try:
            return cls(
                ok=bool(payload.get("ok", True)),
                chat_id=str(payload["chat_id"]),
                extracted=bool(payload["extracted"]),
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise CharacterMemoryClientError(
                "Invalid /save response: expected chat_id and extracted."
            ) from exc


class CharacterMemoryClient:
    """Client for the JSON HTTP API exposed by CharacterMemory's server.

    Parameters
    ----------
    base_url:
        Server origin, for example ``"http://localhost:8000"``. A trailing
        slash is optional.
    api_key:
        Optional API key. When supplied, it is sent as an
        ``Authorization: Bearer ...`` header, matching the server's auth
        middleware.
    timeout:
        Timeout in seconds applied to each request.
    headers:
        Additional request headers. These are copied at construction time.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must be a non-empty string")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Accept": "application/json"}
        if headers:
            self.headers.update(headers)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def list_characters(self) -> list[str]:
        """Return the character names available on the server."""
        payload = self._request("GET", "/")
        characters = payload.get("characters")
        if not isinstance(characters, list):
            raise CharacterMemoryClientError(
                "Invalid server response: characters must be a list."
            )
        return [str(character) for character in characters]

    def context(
        self,
        character: str,
        user: str,
        message: str,
        *,
        chat_id: Optional[str] = None,
        occurred_at: Optional[float] = None,
    ) -> ContextResponse:
        """Persist a user turn and retrieve the character's memory context.

        Omit ``chat_id`` for the first turn. The server creates a chat and the
        returned :attr:`ContextResponse.chat_id` should be passed on later
        turns and to :meth:`save`.
        """
        body: dict[str, Any] = {
            "character": character,
            "user": user,
            "message": message,
        }
        if chat_id is not None:
            body["chat_id"] = chat_id
        if occurred_at is not None:
            body["occurred_at"] = occurred_at
        return ContextResponse.from_dict(self._request("POST", "/context", body))

    # Explicit verb alias for callers who prefer method names that distinguish
    # the endpoint from the returned ContextResponse type.
    get_context = context

    def save(
        self,
        chat_id: str,
        answer: str,
        *,
        occurred_at: Optional[float] = None,
    ) -> SaveResponse:
        """Persist an assistant answer and run server-side extraction."""
        body: dict[str, Any] = {"chat_id": chat_id, "answer": answer}
        if occurred_at is not None:
            body["occurred_at"] = occurred_at
        return SaveResponse.from_dict(self._request("POST", "/save", body))

    save_answer = save

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make one JSON request and return its object response."""
        if not path.startswith("/"):
            path = f"/{path}"
        data = None
        headers = dict(self.headers)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._decode_response(response, method=method, path=path)
        except HTTPError as exc:
            detail = self._decode_error_body(exc)
            raise CharacterMemoryHTTPError(
                "CharacterMemory server returned "
                f"HTTP {exc.code} for {method} {path}: {detail}",
                status_code=exc.code,
                detail=detail,
            ) from exc
        except URLError as exc:
            raise CharacterMemoryClientError(
                f"Could not reach CharacterMemory server at {self.base_url}: {exc.reason}"
            ) from exc

    @staticmethod
    def _decode_response(
        response: HTTPResponse, *, method: str, path: str
    ) -> dict[str, Any]:
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CharacterMemoryClientError(
                f"CharacterMemory server returned invalid JSON for {method} {path}."
            ) from exc
        if not isinstance(payload, dict):
            raise CharacterMemoryClientError(
                "CharacterMemory server returned a non-object response "
                f"for {method} {path}."
            )
        return payload

    @staticmethod
    def _decode_error_body(error: HTTPError) -> Any:
        try:
            raw = error.read().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return error.reason
        if not raw:
            return error.reason
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict) and "detail" in payload:
            return payload["detail"]
        return payload


__all__ = [
    "CharacterMemoryClient",
    "CharacterMemoryClientError",
    "CharacterMemoryHTTPError",
    "ContextResponse",
    "MemoryItem",
    "SaveResponse",
]
