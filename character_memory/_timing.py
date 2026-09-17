"""Request-local timing hooks, independent of the optional server package."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Callable, Iterator, Optional


memory_timing_sink: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "memory_timing_sink", default=None
)
phase_timing_sink: ContextVar[Optional[Callable[[str, float], None]]] = ContextVar(
    "phase_timing_sink", default=None
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
