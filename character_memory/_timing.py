"""Request-local timing hooks, independent of the optional server package."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Callable, Iterator, Optional


memory_timing_sink: ContextVar[Optional[Callable[[str, str, float, bool], None]]] = (
    ContextVar("memory_timing_sink", default=None)
)


@contextmanager
def time_memory(character: str, memory: str) -> Iterator[None]:
    """Report one recall/format operation when a caller installs a sink."""
    sink = memory_timing_sink.get()
    if sink is None:
        yield
        return
    started = perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        sink(character, memory, (perf_counter() - started) * 1000, failed)
