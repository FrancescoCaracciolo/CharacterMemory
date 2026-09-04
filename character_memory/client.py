"""Synchronous client for the CharacterMemory HTTP server.

The bundled server deliberately keeps the model call on the client side. A
typical turn therefore consists of :meth:`CharacterMemoryClient.context`, a
caller-owned LLM request, and :meth:`CharacterMemoryClient.save`. The same
client also exposes the memory browser, knowledge graph, calendar, and live
context-event APIs.

This module uses :mod:`urllib` rather than a third-party HTTP library so it is
available with the core package.  It does not import the optional FastAPI
server package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.client import HTTPResponse
from typing import Any, Iterator, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
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


@dataclass
class MemoryOverview:
    """Summary of one memory returned by ``GET /api/memories/{character}``."""

    name: str
    title: str
    kind: str
    enabled: bool
    count: int
    users: list[str] = field(default_factory=list)
    editable: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryOverview":
        """Build a memory summary from a decoded server payload."""
        try:
            raw_users = payload.get("users") or []
            if not isinstance(raw_users, list):
                raise TypeError("users must be a list")
            return cls(
                name=str(payload["name"]),
                title=str(payload.get("title", payload["name"])),
                kind=str(payload.get("kind", "generic")),
                enabled=bool(payload.get("enabled", True)),
                count=int(payload.get("count", 0)),
                users=[str(user) for user in raw_users],
                editable=bool(payload.get("editable", False)),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CharacterMemoryClientError(
                "Invalid memory overview response."
            ) from exc


@dataclass
class MemoryRecord:
    """One normalized record returned by the memory browser API."""

    id: Any
    user_id: Optional[str]
    text: str
    score: Optional[float]
    fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryRecord":
        """Build a memory record without discarding backend-specific fields."""
        try:
            raw_score = payload.get("score")
            raw_fields = payload.get("fields") or {}
            raw_meta = payload.get("meta") or {}
            if not isinstance(raw_fields, Mapping) or not isinstance(
                raw_meta, Mapping
            ):
                raise TypeError("fields and meta must be objects")
            raw_user = payload.get("user_id")
            return cls(
                id=payload.get("id"),
                user_id=None if raw_user is None else str(raw_user),
                text=str(payload.get("text", "")),
                score=None if raw_score is None else float(raw_score),
                fields=dict(raw_fields),
                meta=dict(raw_meta),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise CharacterMemoryClientError("Invalid memory record response.") from exc


@dataclass
class MemoryPage:
    """A page of records returned by ``GET /api/memories/...``."""

    character: str
    memory: str
    title: str
    kind: str
    enabled: bool
    page: int
    size: int
    total: int
    pages: int
    search: bool
    query: str
    user: Optional[str]
    users: list[str]
    records: list[MemoryRecord]
    extra: dict[str, Any] = field(default_factory=dict)
    editor: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryPage":
        """Build a typed memory page from a decoded server payload."""
        try:
            raw_records = payload.get("records") or []
            raw_users = payload.get("users") or []
            raw_extra = payload.get("extra") or {}
            raw_editor = payload.get("editor") or {}
            if not isinstance(raw_records, list) or not all(
                isinstance(record, Mapping) for record in raw_records
            ):
                raise TypeError("records must be a list of objects")
            if not isinstance(raw_users, list):
                raise TypeError("users must be a list")
            if not isinstance(raw_extra, Mapping) or not isinstance(
                raw_editor, Mapping
            ):
                raise TypeError("extra and editor must be objects")
            raw_user = payload.get("user")
            return cls(
                character=str(payload["character"]),
                memory=str(payload["memory"]),
                title=str(payload.get("title", payload["memory"])),
                kind=str(payload.get("kind", "generic")),
                enabled=bool(payload.get("enabled", True)),
                page=int(payload.get("page", 1)),
                size=int(payload.get("size", 25)),
                total=int(payload.get("total", len(raw_records))),
                pages=int(payload.get("pages", 1)),
                search=bool(payload.get("search", False)),
                query=str(payload.get("query", "")),
                user=None if raw_user is None else str(raw_user),
                users=[str(user) for user in raw_users],
                records=[MemoryRecord.from_dict(record) for record in raw_records],
                extra=dict(raw_extra),
                editor=dict(raw_editor),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CharacterMemoryClientError("Invalid memory page response.") from exc


@dataclass
class GraphResponse:
    """Knowledge-graph nodes and edges returned by ``GET /api/graph/...``.

    Nodes and edges intentionally remain dictionaries because their fields
    vary by node/edge kind and may be extended by custom graph backends.
    """

    character: str
    query: str
    user: Optional[str]
    mode: str
    self_node: Optional[str]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    activation_range: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    overview: dict[str, Any] = field(default_factory=dict)
    include_internal: bool = False
    min_degree: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphResponse":
        """Build a graph response while preserving kind-specific fields."""
        try:
            raw_nodes = payload.get("nodes") or []
            raw_edges = payload.get("edges") or []
            if not isinstance(raw_nodes, list) or not all(
                isinstance(node, Mapping) for node in raw_nodes
            ):
                raise TypeError("nodes must be a list of objects")
            if not isinstance(raw_edges, list) or not all(
                isinstance(edge, Mapping) for edge in raw_edges
            ):
                raise TypeError("edges must be a list of objects")
            activation_range = payload.get("activation_range") or {}
            overview = payload.get("overview") or {}
            if not isinstance(activation_range, Mapping) or not isinstance(
                overview, Mapping
            ):
                raise TypeError("activation_range and overview must be objects")
            raw_user = payload.get("user")
            raw_self_node = payload.get("self_node")
            return cls(
                character=str(payload["character"]),
                query=str(payload.get("query", "")),
                user=None if raw_user is None else str(raw_user),
                mode=str(payload.get("mode", "subgraph")),
                self_node=None if raw_self_node is None else str(raw_self_node),
                nodes=[dict(node) for node in raw_nodes],
                edges=[dict(edge) for edge in raw_edges],
                activation_range=dict(activation_range),
                truncated=bool(payload.get("truncated", False)),
                overview=dict(overview),
                include_internal=bool(payload.get("include_internal", False)),
                min_degree=int(payload.get("min_degree", 0)),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CharacterMemoryClientError("Invalid graph response.") from exc


@dataclass
class CalendarEventsResponse:
    """Calendar events returned by ``GET /api/calendar/.../events``."""

    character: str
    timezone: str
    events: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalendarEventsResponse":
        """Build a calendar response from a decoded server payload."""
        try:
            events = payload.get("events") or []
            if not isinstance(events, list) or not all(
                isinstance(event, Mapping) for event in events
            ):
                raise TypeError("events must be a list of objects")
            return cls(
                character=str(payload["character"]),
                timezone=str(payload.get("timezone", "UTC")),
                events=[dict(event) for event in events],
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise CharacterMemoryClientError("Invalid calendar response.") from exc


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

    def list_memories(self, character: str) -> list[MemoryOverview]:
        """Return every memory system configured for ``character``.

        Each summary includes its record count, known user ids, backend kind,
        enabled state, and whether the server permits direct record edits.
        """
        payload = self._request(
            "GET", self._path("api", "memories", character)
        )
        memories = payload.get("memories")
        if not isinstance(memories, list) or not all(
            isinstance(memory, Mapping) for memory in memories
        ):
            raise CharacterMemoryClientError(
                "Invalid memory overview response: memories must be a list of objects."
            )
        return [MemoryOverview.from_dict(memory) for memory in memories]

    # GET /api/memories/{character}
    get_memories = list_memories

    def get_memory(
        self,
        character: str,
        memory: str,
        *,
        page: int = 1,
        size: int = 25,
        user: Optional[str] = None,
        query: Optional[str] = None,
    ) -> MemoryPage:
        """Return one page of records from a named memory.

        ``query`` asks the server to use semantic search with its lexical
        fallback. ``user`` limits per-user memories to a single owner.
        """
        params: dict[str, Any] = {"page": page, "size": size}
        if user is not None:
            params["user"] = user
        if query is not None:
            params["q"] = query
        payload = self._request(
            "GET",
            self._path("api", "memories", character, memory),
            params=params,
        )
        return MemoryPage.from_dict(payload)

    read_memory = get_memory

    def iter_memory_records(
        self,
        character: str,
        memory: str,
        *,
        size: int = 100,
        user: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Iterator[MemoryRecord]:
        """Yield every record across the pages of a named memory."""
        page_number = 1
        while True:
            page = self.get_memory(
                character,
                memory,
                page=page_number,
                size=size,
                user=user,
                query=query,
            )
            yield from page.records
            if page_number >= page.pages or not page.records:
                break
            page_number += 1

    def get_memory_records(
        self,
        character: str,
        memory: str,
        *,
        size: int = 100,
        user: Optional[str] = None,
        query: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """Return all records from a memory, following server pagination."""
        return list(
            self.iter_memory_records(
                character, memory, size=size, user=user, query=query
            )
        )

    def add_memory_record(
        self, character: str, memory: str, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Add a record to an editable memory and return the server result."""
        return self._request(
            "POST",
            self._path("api", "memories", character, memory),
            {"values": dict(values)},
        )

    create_memory_record = add_memory_record

    def update_memory_record(
        self,
        character: str,
        memory: str,
        record_id: Any,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Update fields on an editable memory record."""
        return self._request(
            "PUT",
            self._path("api", "memories", character, memory, record_id),
            {"values": dict(values)},
        )

    def delete_memory_record(
        self, character: str, memory: str, record_id: Any
    ) -> dict[str, Any]:
        """Delete a record from an editable memory."""
        return self._request(
            "DELETE", self._path("api", "memories", character, memory, record_id)
        )

    def get_graph(
        self,
        character: str,
        *,
        query: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 50,
        hops: int = 1,
        max_edges: int = 400,
        include_co_occurrence: bool = False,
        full: bool = False,
        retrieve_k: int = 30,
        min_degree: int = 0,
        include_internal: bool = False,
    ) -> GraphResponse:
        """Return the character's activation-weighted knowledge graph."""
        params: dict[str, Any] = {
            "limit": limit,
            "hops": hops,
            "max_edges": max_edges,
            "include_co_occurrence": include_co_occurrence,
            "full": full,
            "retrieve_k": retrieve_k,
            "min_degree": min_degree,
            "include_internal": include_internal,
        }
        if query is not None:
            params["q"] = query
        if user is not None:
            params["user"] = user
        return GraphResponse.from_dict(
            self._request(
                "GET", self._path("api", "graph", character), params=params
            )
        )

    read_graph = get_graph

    def get_nodes(self, character: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Return only the nodes from :meth:`get_graph`."""
        return self.get_graph(character, **kwargs).nodes

    def get_edges(self, character: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Return only the edges from :meth:`get_graph`."""
        return self.get_graph(character, **kwargs).edges

    def get_calendar_events(
        self,
        character: str,
        *,
        start: Optional[str] = None,
        before: Optional[str] = None,
        query: Optional[str] = None,
        owners: Optional[Sequence[str]] = None,
        include_cancelled: bool = False,
        limit: int = 100,
    ) -> CalendarEventsResponse:
        """Search one-off events and routines in a character's calendar."""
        params: dict[str, Any] = {
            "include_cancelled": include_cancelled,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if before is not None:
            params["before"] = before
        if query is not None:
            params["q"] = query
        if owners is not None:
            params["owner"] = [owners] if isinstance(owners, str) else list(owners)
        return CalendarEventsResponse.from_dict(
            self._request(
                "GET",
                self._path("api", "calendar", character, "events"),
                params=params,
            )
        )

    list_calendar_events = get_calendar_events

    def create_calendar_event(
        self, character: str, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Create a one-off event or routine in an enabled calendar memory."""
        return self._request(
            "POST",
            self._path("api", "calendar", character, "events"),
            {"values": dict(values)},
        )

    def update_calendar_event(
        self, character: str, event_id: int, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Update an existing calendar event or routine."""
        return self._request(
            "PATCH",
            self._path("api", "calendar", character, "events", event_id),
            {"values": dict(values)},
        )

    def cancel_calendar_event(
        self, character: str, event_id: int
    ) -> dict[str, Any]:
        """Mark a calendar event or routine as cancelled."""
        return self._request(
            "POST",
            self._path("api", "calendar", character, "events", event_id, "cancel"),
        )

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

    def iter_context_events(
        self,
        character: str,
        *,
        user: Optional[str] = None,
        last_event_id: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield live ``/context`` snapshots from the server's SSE stream.

        Iteration blocks until a context event arrives. Stop consuming or
        close the generator to close the underlying HTTP response. SSE
        comments/keep-alives are ignored and each JSON ``data`` payload is
        yielded as a dictionary.
        """
        params = {"user": user} if user is not None else None
        headers = dict(self.headers)
        headers["Accept"] = "text/event-stream"
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        path = self._path("api", "context-events", character)
        request = Request(
            self._url(path, params), headers=headers, method="GET"
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data_lines: list[str] = []
                for raw_line in response:
                    try:
                        line = raw_line.decode("utf-8").rstrip("\r\n")
                    except UnicodeDecodeError as exc:
                        raise CharacterMemoryClientError(
                            "CharacterMemory context event stream returned invalid UTF-8."
                        ) from exc
                    if not line:
                        if data_lines:
                            yield self._decode_event_data("\n".join(data_lines))
                            data_lines.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    field, separator, value = line.partition(":")
                    if separator and value.startswith(" "):
                        value = value[1:]
                    if field == "data":
                        data_lines.append(value)
                if data_lines:
                    yield self._decode_event_data("\n".join(data_lines))
        except HTTPError as exc:
            detail = self._decode_error_body(exc)
            raise CharacterMemoryHTTPError(
                "CharacterMemory server returned "
                f"HTTP {exc.code} for GET {path}: {detail}",
                status_code=exc.code,
                detail=detail,
            ) from exc
        except URLError as exc:
            raise CharacterMemoryClientError(
                f"Could not reach CharacterMemory server at {self.base_url}: {exc.reason}"
            ) from exc

    stream_context_events = iter_context_events

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Call a JSON endpoint not yet covered by a convenience method.

        This escape hatch keeps custom server routes accessible while using
        the same authentication, timeout, JSON decoding, and error handling.
        """
        return self._request(method.upper(), path, payload, params=params)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        params: Optional[Mapping[str, Any]] = None,
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
            self._url(path, params),
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
    def _path(*segments: Any) -> str:
        """Build a URL path while escaping every dynamic segment."""
        return "/" + "/".join(quote(str(segment), safe="") for segment in segments)

    def _url(
        self, path: str, params: Optional[Mapping[str, Any]] = None
    ) -> str:
        """Build an absolute URL, including repeated and boolean query values."""
        url = f"{self.base_url}{path}"
        if not params:
            return url
        pairs: list[tuple[str, str]] = []
        for key, raw_value in params.items():
            if raw_value is None:
                continue
            values = (
                raw_value
                if isinstance(raw_value, (list, tuple))
                else (raw_value,)
            )
            for value in values:
                if value is None:
                    continue
                encoded = str(value).lower() if isinstance(value, bool) else str(value)
                pairs.append((str(key), encoded))
        if not pairs:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{urlencode(pairs)}"

    @staticmethod
    def _decode_event_data(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CharacterMemoryClientError(
                "CharacterMemory context event stream returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise CharacterMemoryClientError(
                "CharacterMemory context event stream returned a non-object event."
            )
        return payload

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
    "CalendarEventsResponse",
    "CharacterMemoryClient",
    "CharacterMemoryClientError",
    "CharacterMemoryHTTPError",
    "ContextResponse",
    "GraphResponse",
    "MemoryItem",
    "MemoryOverview",
    "MemoryPage",
    "MemoryRecord",
    "SaveResponse",
]
