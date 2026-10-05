"""Request-local timing hooks, independent of the optional server package."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator, Optional


@dataclass
class ToolTiming:
    """One completed tool execution, as delivered to ``tool_timing_sink``."""

    name: str
    elapsed_ms: float = 0.0
    failed: bool = False
    # Nested ``time_phase`` spans (e.g. embedding network time) inside the tool.
    phases: dict[str, float] = field(default_factory=dict)


memory_timing_sink: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "memory_timing_sink", default=None
)
phase_timing_sink: ContextVar[Optional[Callable[[str, float], None]]] = ContextVar(
    "phase_timing_sink", default=None
)
tool_timing_sink: ContextVar[Optional[Callable[[ToolTiming], None]]] = ContextVar(
    "tool_timing_sink", default=None
)


@contextmanager
def time_phase(name: str) -> Iterator[None]:
    """Accumulate one nested phase (e.g. embedding network time).

    Active only inside a :func:`time_memory` block with an installed sink;
    otherwise this is a no-op with near-zero overhead, so hot paths can
    instrument unconditionally.
    """
    sink = phase_timing_sink.get()
    if sink is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        sink(name, (perf_counter() - started) * 1000)


@contextmanager
def time_memory(character: str, memory: str) -> Iterator[None]:
    """Report one recall/format operation when a caller installs a sink.

    Nested ``time_phase`` spans are collected for the duration of the block
    and forwarded to the sink as an optional fifth argument. Sinks that only
    accept the legacy four arguments still work: phases are passed only when
    at least one was recorded.
    """
    sink = memory_timing_sink.get()
    if sink is None:
        yield
        return
    phases: dict[str, float] = {}

    def record_phase(name: str, elapsed_ms: float) -> None:
        phases[name] = phases.get(name, 0.0) + elapsed_ms

    token = phase_timing_sink.set(record_phase)
    started = perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        phase_timing_sink.reset(token)
        elapsed_ms = (perf_counter() - started) * 1000
        if phases:
            sink(character, memory, elapsed_ms, failed, phases)
        else:
            sink(character, memory, elapsed_ms, failed)


@contextmanager
def time_tool(name: str) -> Iterator[ToolTiming]:
    """Time one tool execution and report it to ``tool_timing_sink``.

    Always measures (callers such as the MCP endpoint use the yielded record
    directly), and reports only when a sink is installed. Tools that signal
    failure through their return value rather than by raising set
    ``timing.failed`` themselves. Nested phases are also forwarded to any
    enclosing phase sink so outer breakdowns stay complete.
    """
    timing = ToolTiming(name=name)
    outer_phase_sink = phase_timing_sink.get()

    def record_phase(phase: str, elapsed_ms: float) -> None:
        timing.phases[phase] = timing.phases.get(phase, 0.0) + elapsed_ms
        if outer_phase_sink is not None:
            outer_phase_sink(phase, elapsed_ms)

    token = phase_timing_sink.set(record_phase)
    started = perf_counter()
    try:
        yield timing
    except BaseException:
        timing.failed = True
        raise
    finally:
        phase_timing_sink.reset(token)
        timing.elapsed_ms = (perf_counter() - started) * 1000
        sink = tool_timing_sink.get()
        if sink is not None:
            sink(timing)
