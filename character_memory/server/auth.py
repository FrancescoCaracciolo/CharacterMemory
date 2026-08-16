"""Optional API-key authentication for the CharacterMemory server.

Auth is **off by default**: it turns on only when at least one key is
configured (``CM_API_KEY`` in the environment / ``.env``, or the
``--api-key`` flag of ``charactermemory-server``). With no key configured
every endpoint stays open — the historical behaviour, bit-for-bit.

One :class:`APIKeyMiddleware` on the single FastAPI app in :file:`api.py`
gates everything at once: the thin-client API (``/``, ``/context``, ``/save``),
the memory browser + admin JSON endpoints (``/api/...``), and the MCP
endpoint (``POST /mcp``) — auth failures there are a plain HTTP ``401``,
which is what streamable-HTTP MCP clients expect, *not* a JSON-RPC envelope.

A request may present its key in any of three places (first one found wins):

1. ``Authorization: Bearer <key>`` header — HTTP API clients and MCP clients
   (most MCP client configs have a ``headers`` field for exactly this);
2. ``X-API-Key: <key>`` header — what the browser GUI sends;
3. ``?api_key=<key>`` query parameter — the fallback for channels that
   cannot set headers at all, i.e. the GUI's SSE ``EventSource``. Query
   strings can end up in access logs, so prefer the headers when possible.

The GUI page itself (``/gui`` and its ``/gui/static/*`` assets) stays
public: the shell ships inside the package and holds no user data. It loads
unauthenticated, then prompts for the key and attaches it to every data
request.

``CM_API_KEY`` accepts a comma-separated list so keys can be rotated or
issued per client without downtime; any match passes. Comparisons use
:func:`hmac.compare_digest` to avoid leaking the key through timing.
"""

from __future__ import annotations

import hmac
import os
from typing import Iterable, Optional, Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

ENV_VAR = "CM_API_KEY"

# Path prefixes that never require a key. ``/gui`` is the SPA shell and
# ``/gui/static`` its JS/CSS/images — all package assets with zero user
# data. Every endpoint they call is still protected.
DEFAULT_PUBLIC_PREFIXES = ("/gui", "/gui/static")


def parse_api_keys(raw: Optional[str]) -> list[str]:
    """Split a raw ``CM_API_KEY`` value into the list of accepted keys.

    Comma-separated, whitespace-trimmed, empty entries dropped. An unset,
    empty or all-commas value yields ``[]`` — meaning auth is disabled.
    """
    if not raw:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def read_api_keys(env: Optional[os.Mapping[str, str]] = None) -> list[str]:
    """Read the accepted keys from the environment (``CM_API_KEY``)."""
    env = os.environ if env is None else env
    return parse_api_keys(env.get(ENV_VAR))


def _provided_key(request: Request) -> Optional[str]:
    """Extract the caller's key from the request, or ``None`` if absent.

    Checked in the documented order: Bearer header, ``X-API-Key`` header,
    ``api_key`` query parameter.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate:
            return candidate
    candidate = request.headers.get("X-API-Key", "").strip()
    if candidate:
        return candidate
    return request.query_params.get("api_key") or None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid API key when keys are configured.

    ``api_keys`` is stored **by reference**: passing a module-level list
    (as :file:`api.py` does) lets ``main()`` flip auth on in-process for the
    ``--api-key`` flag by mutating that list — the middleware was already
    built over it at import time. With an empty list every request passes
    through untouched.
    """

    def __init__(
        self,
        app,
        api_keys: Sequence[str],
        public_prefixes: Iterable[str] = DEFAULT_PUBLIC_PREFIXES,
    ) -> None:
        super().__init__(app)
        # Kept by reference (no copy): api.py passes its module-level list so
        # main()'s --api-key flag can flip auth on in-process by mutating it.
        self.api_keys = api_keys
        self.public_prefixes = tuple(public_prefixes)

    def _is_public(self, path: str) -> bool:
        """Exact match, or a proper prefix boundary (``/gui`` covers
        ``/gui/...`` but not ``/guile``)."""
        for prefix in self.public_prefixes:
            root = prefix.rstrip("/")
            if path == root or path.startswith(root + "/"):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        if not self.api_keys or self._is_public(request.url.path):
            return await call_next(request)
        provided = _provided_key(request)
        if provided is not None and any(
            hmac.compare_digest(provided, key) for key in self.api_keys
        ):
            return await call_next(request)
        return JSONResponse(
            {
                "detail": (
                    "Invalid or missing API key. Authenticate with an "
                    "'Authorization: Bearer <key>' or 'X-API-Key: <key>' "
                    "header, or an 'api_key' query parameter."
                )
            },
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
