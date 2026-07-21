"""Abstract chat-completions client."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    # Event types live in the tools package (the agent imports them through
    # there). Imported only for type hints to avoid an llm → tools runtime cycle.
    from ..tools.base import ToolStreamEvent


class LLMClient(ABC):
    """Interface for chat-completion style models."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the assistant text for `messages` (openai-style dicts)."""

    @abstractmethod
    def chat_structured(
        self,
        messages: list[dict],
        schema: dict[str, Any],
        *,
        temperature: Optional[float] = None,
    ) -> dict:
        """Return a JSON object conforming (best-effort) to `schema`.

        `schema` is a JSON-schema dict describing the expected object.
        Implementations should request JSON output and parse it leniently.
        """

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield assistant text chunks for `messages`.

        Default implementation (non-streaming fallback) yields the full
        :meth:`chat` reply at once, so every `LLMClient` works with
        ``generate_answer(stream=True)`` out of the box. Override to emit
        real incremental deltas from the backend.
        """
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------ #
    # Tool calling
    # ------------------------------------------------------------------ #
    # The tool-calling surface is opt-in: these raise NotImplementedError by
    # default, so a backend that can't do tool calls fails loudly rather than
    # silently ignoring the `tools=` argument. The bundled
    # `OpenAICompatibleLLM` implements both. `chat_with_tools_stream` has a
    # default that delegates to `chat_with_tools`, so a backend only needs to
    # implement the non-streaming variant to support `stream=True` too (just
    # non-incrementally). See `character_memory.tools` for the `Tool` /
    # `ToolRegistry` / event types.
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMResponse":
        """One model round with tools available; returns text and/or tool calls.

        ``tools`` is the OpenAI ``tools=[{"type":"function","function":{...}}]``
        list (render one with :meth:`character_memory.tools.Tool.schema`).
        ``tool_choice`` follows the OpenAI convention (``"auto"``, ``"none"``,
        ``"required"``, or ``{"type":"function","function":{"name":...}}``);
        pass ``None`` to let the backend pick its default.

        Implementations should populate :attr:`LLMResponse.tool_calls` from the
        model's response (parsing ``function.arguments`` leniently) and
        :attr:`LLMResponse.content` with any plain text the model returned
        alongside the calls.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling "
            f"(chat_with_tools). Implement it on the subclass."
        )

    def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "Iterator[ToolStreamEvent]":
        """Stream one tool-enabled round as :class:`~character_memory.tools.TextChunk`
        / :class:`~character_memory.tools.ToolCallEvent` events.

        Default implementation (non-streaming fallback): runs
        :meth:`chat_with_tools` once and emits the whole reply as a single
        :class:`TextChunk` (or the tool calls as one
        :class:`ToolCallEvent`). Override to emit real incremental deltas.

        Note: the default does *not* emit :class:`ToolResultEvent` — tool
        execution is the agent loop's job, not the client's. A streaming client
        only reports what the *model* did.
        """
        from ..tools.base import TextChunk, ToolCallEvent

        resp = self.chat_with_tools(
            messages, tools, tool_choice=tool_choice,
            temperature=temperature, max_tokens=max_tokens,
        )
        if resp.content:
            yield TextChunk(resp.content)
        if resp.tool_calls:
            yield ToolCallEvent(resp.tool_calls)


class LLMResponse:
    """Result of one :meth:`LLMClient.chat_with_tools` round.

    ``content`` is whatever plain text the model returned; ``tool_calls`` the
    tool invocations it requested (if any). Either may be empty; when both are
    empty the round produced nothing usable (the agent treats that as "done"
    with an empty reply).
    """

    __slots__ = ("content", "tool_calls")

    def __init__(self, content: str = "", tool_calls: Optional[list] = None) -> None:
        self.content = content or ""
        self.tool_calls = list(tool_calls) if tool_calls else []

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"LLMResponse(content={self.content!r}, "
            f"tool_calls={len(self.tool_calls)})"
        )
