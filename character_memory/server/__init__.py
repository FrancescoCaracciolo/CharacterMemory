"""Optional server subpackage for :mod:`character_memory`.

This pulls in FastAPI/uvicorn, so it is *not* imported by the core
``character_memory`` package. Install it via the ``server`` extra::

    pip install charactermemory[server]

then run either with the console script::

    charactermemory-server

or with uvicorn::

    uvicorn character_memory.server:app

Importing this subpackage (or ``character_memory.server:app``) requires the
``server`` extra to be installed; otherwise the FastAPI import below raises an
``ImportError``.
"""

from .api import app, main
from .mcp import build_router as build_mcp_router

__all__ = ["app", "main", "build_mcp_router"]
