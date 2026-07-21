"""``@tool``: build a :class:`Tool` from a plain function.

Schema inference rules (kept deliberately small):

* The tool ``name`` is the function's ``__name__`` (override with
  ``@tool(name="...")``).
* The ``description`` is the function's docstring, stripped.
* Each annotated parameter becomes a JSON-schema property:

    +---------------------------+---------------------+
    | Python annotation         | JSON-schema type    |
    +===========================+=====================+
    | ``str``                   | ``string``          |
    | ``int``                   | ``integer``         |
    | ``float``                 | ``number``          |
    | ``bool``                  | ``boolean``         |
    +---------------------------+---------------------+
    | ``list[X]`` / ``list``    | ``array``           |
    | ``dict`` / ``dict[K,V]``  | ``object``          |
    +---------------------------+---------------------+

  Any annotation outside that set (including ``Any`` or an unknown class) maps
  to ``string`` — the model is still asked for text, and your function can
  coerce it.
* A parameter is ``required`` unless it has a default. ``Optional[X]`` (which
  is ``X | None`` / ``Union[X, None]``) is unwrapped before mapping, so
  ``Optional[int]`` is ``integer`` and (because it usually carries a default)
  not required.

Anything fancier (``oneOf``, enums, ``$ref``) belongs on a hand-written
:class:`Tool` subclass; the decorator is for the common, stateless case.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Optional, TypeVar, Union, get_args, get_origin

from .base import Tool

F = TypeVar("F", bound=Callable[..., Any])


def _map_type(annotation: Any) -> str:
    """Reduce a Python annotation to a JSON-schema type string."""
    # Unwrap Optional[X] / X | None → the inner type. ``Optional[X]`` is just
    # ``Union[X, None]``; for ``X | None`` (PEP 604) get_origin is ``types.UnionType``.
    origin = get_origin(annotation)
    if origin is Union or (origin is not None and getattr(origin, "__name__", "") == "UnionType"):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]
            origin = get_origin(annotation)

    if annotation in (str, "str"):
        return "string"
    if annotation in (int, "int"):
        return "integer"
    if annotation in (float, "float"):
        return "number"
    if annotation in (bool, "bool"):
        return "boolean"
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "string"
    # list[...] / list / typing.List
    if annotation in (list, "list") or origin in (list,):
        return "array"
    # dict / dict[...] / typing.Dict
    if annotation in (dict, "dict") or origin in (dict,):
        return "object"
    # Fall back: we don't know, so ask the model for text.
    return "string"


def _infer_parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """Build the JSON-schema ``parameters`` dict from ``func``'s signature."""
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        properties[pname] = {"type": _map_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


class _FunctionTool(Tool):
    """A :class:`Tool` backed by a plain Python function."""

    def __init__(self, func: Callable[..., Any], *, name: Optional[str] = None,
                 description: Optional[str] = None,
                 parameters: Optional[dict[str, Any]] = None) -> None:
        self._func = func
        self.name = name or func.__name__
        self.description = (description or (func.__doc__ or "")).strip()
        self.parameters = parameters if parameters is not None else _infer_parameters(func)

    def run(self, **kwargs: Any) -> str:
        return self._func(**kwargs)


def tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> Any:
    """Turn a function into a :class:`Tool`.

    Usable bare (``@tool``) or with options (``@tool(name="get_weather")``).
    The returned object is a ``Tool`` instance, so it can be passed straight to
    a registry or ``generate_answer(tools=...)``; it is *not* callable as a
    function any more — call ``.run(...)`` to invoke the original function, or
    keep a separate reference to it.
    """
    def _wrap(f: F) -> Tool:
        return _FunctionTool(
            f, name=name, description=description, parameters=parameters
        )

    if func is None:
        # Called as @tool(name=...) — return a decorator.
        return _wrap
    # Called as @tool with no arguments.
    return _wrap(func)
