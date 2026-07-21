"""Tool registry: collect tools, render their schemas, dispatch calls.

Two entry points, mirroring the chunker registry:

* :class:`ToolRegistry` — the object the agent loop holds. ``add`` tools,
  ``schemas()`` for the ``tools=`` list, ``execute`` a model call (catching
  exceptions so one bad tool can't crash the loop).
* :func:`register_tool` / :func:`get_tool` — a module-level registry for tools
  that should be globally addressable, like :func:`register_chunker`.

The registry is permissive about what you hand it: a ``Tool``, a function
decorated with :func:`~character_memory.tools.decorator.tool`, a plain function
(wrapped on the fly), or another registry. That lets callers compose:

    tools = ToolRegistry(agent.memory_tools()) + ToolRegistry([GetWeather()])
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Iterable, Optional, Union

from .base import Tool, ToolCall, ToolResult

#: Anything :meth:`ToolRegistry.add` accepts.
ToolLike = Union[Tool, Callable[..., Any], "ToolRegistry", Iterable]


def _coerce_tool(obj: Any) -> Tool:
    """Normalise a ``ToolLike`` into a :class:`Tool` instance.

    A bare :class:`Tool` is returned as-is. A plain callable is wrapped with
    the :func:`~character_memory.tools.decorator.tool` decorator so its schema
    is inferred from annotations. (Callables already produced by the decorator
    are themselves ``Tool`` instances and take the first branch.)
    """
    if isinstance(obj, Tool):
        return obj
    if isinstance(obj, ToolRegistry):
        raise TypeError(
            "pass a ToolRegistry via ToolRegistry(another) or registry + registry, "
            "not add(); a nested registry would need merging."
        )
    if callable(obj):
        # Imported lazily to avoid a registry → decorator → (nothing) cycle;
        # decorator.py imports only from .base.
        from .decorator import tool as tool_decorator

        return tool_decorator(obj)
    raise TypeError(f"cannot turn {obj!r} into a Tool; expected a Tool or callable.")


class ToolRegistry:
    """An ordered, name-keyed collection of :class:`Tool` objects."""

    def __init__(self, tools: Optional[Iterable] = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools is not None:
            self.add(tools)

    # -- mutation ----------------------------------------------------------- #
    def add(self, tools: ToolLike) -> "ToolRegistry":
        """Register one tool or many (a Tool, a callable, an iterable of either).

        Returns ``self`` so calls chain. A later registration with the same name
        overwrites the earlier one — same semantics as the chunker registry.
        """
        if isinstance(tools, (Tool, ToolRegistry)) or callable(tools):
            items: Iterable = [tools]
        else:
            items = tools
        for obj in items:
            if isinstance(obj, ToolRegistry):
                # Merge another registry in by copying its tools.
                for t in obj._tools.values():
                    self._tools[t.name] = t
                continue
            t = _coerce_tool(obj)
            self._tools[t.name] = t
        return self

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __add__(self, other: "ToolRegistry | Iterable") -> "ToolRegistry":
        """``a + b`` (registry or iterable of tools) → a fresh registry."""
        merged = ToolRegistry()
        merged.add(self)
        if isinstance(other, ToolRegistry):
            merged.add(other)
        else:
            merged.add(other)
        return merged

    # -- introspection ------------------------------------------------------ #
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI ``tools=[...]`` envelope for every registered tool."""
        return [t.schema() for t in self._tools.values()]

    # -- dispatch ----------------------------------------------------------- #
    def execute(self, name: str, arguments: Any) -> ToolResult:
        """Run the tool named ``name`` with ``arguments`` (dict or JSON string).

        Unknown tool / bad arguments / raised exception → a ``ToolResult`` with
        ``ok=False`` whose ``text`` explains the failure. The agent feeds that
        text back to the model so it can recover, instead of crashing the loop.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                call=ToolCall(id="", name=name, arguments={}),
                text=f"Unknown tool: {name!r}. Available: {self.names()}.",
                ok=False,
            )
        # The model may send ``arguments`` as a JSON string (OpenAI wire format)
        # or as a parsed object (already-decoded clients). Normalise to kwargs.
        kwargs = _parse_arguments(arguments)
        if not isinstance(kwargs, dict):
            return ToolResult(
                call=ToolCall(id="", name=name, arguments={}),
                text=f"tool {name!r}: arguments must be a JSON object, got {arguments!r}.",
                ok=False,
            )
        try:
            text = tool.run(**kwargs)
            if text is None:
                text = ""
            text = str(text)
        except TypeError as e:
            # Most common failure: the model didn't supply a required kwarg.
            # Surface the signature mismatch rather than a bare traceback.
            text = f"tool {name!r} argument error: {e}"
            return ToolResult(
                call=ToolCall(id="", name=name, arguments=kwargs), text=text, ok=False
            )
        except Exception as e:  # noqa: BLE001 - one bad tool must not kill the loop
            text = f"tool {name!r} raised: {e!r}"
            return ToolResult(
                call=ToolCall(id="", name=name, arguments=kwargs), text=text, ok=False
            )
        return ToolResult(
            call=ToolCall(id="", name=name, arguments=kwargs), text=text, ok=True
        )


def _parse_arguments(arguments: Any) -> Any:
    """Decode OpenAI's ``function.arguments`` (string or object) into a dict."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None  # signal "not an object" to the caller
    return arguments


# --------------------------------------------------------------------------- #
# Module-level registry (mirrors chunking/registry.py)
# --------------------------------------------------------------------------- #
_GLOBAL: ToolRegistry = ToolRegistry()


def register_tool(tool: ToolLike) -> None:
    """Add a tool to the global registry (mirrors ``register_chunker``)."""
    _GLOBAL.add(tool)


def get_tool(name: str) -> Optional[Tool]:
    """Look up a tool by name in the global registry."""
    return _GLOBAL.get(name)


def global_registry() -> ToolRegistry:
    """The shared module-level registry (so the agent can read/extend it)."""
    return _GLOBAL
