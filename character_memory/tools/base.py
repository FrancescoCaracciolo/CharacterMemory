"""Tool-calling primitives shared by the LLM client and the agent loop.

This module is imported by :mod:`character_memory.llm.base` (for the stream
event types), so it must not import anything from the ``llm`` package — that
would create an import cycle. The streaming events and the ``Tool`` ABC live
here for exactly that reason.

Two ways to define a tool:

* Subclass :class:`Tool` — explicit, lets you hold state (e.g. a back-reference
  to the agent). Used by the built-in memory self-tools.
* Decorate a plain function with :func:`character_memory.tools.tool` — the
  ``@tool`` decorator infers the JSON-schema from the function's annotations
  and its docstring. Concise; best for stateless helpers.

Both produce a :class:`Tool` instance, so a :class:`ToolRegistry` treats them
identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Union


# --------------------------------------------------------------------------- #
# Tool model
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    ``id`` is the opaque correlation id the backend assigned (echoed back in
    the ``tool``-role reply so the API can pair request and result). ``name``
    and ``arguments`` are the dispatch key + parsed kwargs.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The text a tool produced (or an error message) for one :class:`ToolCall`.

    ``ok`` is False when :meth:`ToolRegistry.execute` caught an exception; the
    agent still feeds ``text`` back to the model so it can react to the failure.
    """

    call: ToolCall
    text: str
    ok: bool = True


class Tool(ABC):
    """Base class for every tool.

    A subclass declares three class attributes and implements :meth:`run`:

    * ``name``        — the dispatch key the model emits (unique per registry).
    * ``description`` — what the tool does, shown to the model.
    * ``parameters``  — a JSON-schema dict describing the arguments.

    :meth:`schema` renders those into the OpenAI ``function`` tool envelope, so
    a registry can hand the whole list straight to ``tools=[...]``. :meth:`run`
    returns a string: that is what the agent feeds back to the model as the
    ``tool``-role message content.
    """

    name: str = "tool"
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool with the model-supplied arguments; return text."""

    def schema(self) -> dict[str, Any]:
        """Render this tool as the OpenAI ``function`` tool envelope."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------- #
# Streaming events
# --------------------------------------------------------------------------- #
# These are yielded by ``LLMClient.chat_with_tools_stream`` and by the agent's
# streaming tool loop. Keeping them dataclasses (rather than opaque dicts) makes
# the consumer side read cleanly: ``isinstance(ev, TextChunk)`` etc.
@dataclass
class TextChunk:
    """A delta of the assistant's visible text reply."""

    text: str


@dataclass
class ToolCallEvent:
    """Emitted when the model requests one or more tool calls in a round.

    ``calls`` is the complete batch for that round (OpenAI may emit several
    tool calls in a single assistant turn); the agent executes them all before
    re-prompting.
    """

    calls: list[ToolCall]


@dataclass
class ToolResultEvent:
    """Emitted once a tool has executed, carrying its result text."""

    result: ToolResult


#: The union an ``LLMClient`` streaming tool round / the agent's tool loop yields.
ToolStreamEvent = Union[TextChunk, ToolCallEvent, ToolResultEvent]
