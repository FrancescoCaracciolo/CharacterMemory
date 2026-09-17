"""Optional console timing for complete HTTP requests, including streams."""

from __future__ import annotations

import os
from time import perf_counter

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .._timing import memory_timing_sink


class RequestMeterMiddleware:
    """Time requests when CM_METER is enabled; leave other ASGI scopes alone."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Read at request time: the CLI runs after the app is imported, while
        # reload workers inherit this setting through their environment.
        if scope["type"] != "http" or os.environ.get("CM_METER", "").lower() not in {
            "1", "true", "yes", "on",
        }:
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status = 500
        failed = False

        async def metered_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        def report_memory(
            character: str,
            memory: str,
            elapsed_ms: float,
            failed: bool,
            phases: dict[str, float] | None = None,
        ) -> None:
            # Nested-phase breakdown (e.g. embed network time inside a
            # recall) turns "why is this memory slow" into a readout.
            suffix = ""
            if phases:
                rendered = " ".join(
                    f"{name}={ms:.1f}ms" for name, ms in sorted(phases.items())
                )
                suffix = f" [{rendered}]"
            print(
                f"[meter] {scope['method']} {scope['path']} "
                f"character={character!r} memory={memory!r} "
                f"{'ERROR ' if failed else ''}{elapsed_ms:.2f} ms{suffix}",
                flush=True,
            )

        token = memory_timing_sink.set(report_memory)
        try:
            await self.app(scope, receive, metered_send)
        except BaseException:
            failed = True
            raise
        finally:
            memory_timing_sink.reset(token)
            elapsed_ms = (perf_counter() - started) * 1000
            # Omit query strings, which may contain API keys or user content.
            print(
                f"[meter] {scope['method']} {scope['path']} "
                f"{status}{' ERROR' if failed else ''} {elapsed_ms:.2f} ms",
                flush=True,
            )
