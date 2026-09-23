"""Acchan's Discord-facing tools.

These are :class:`character_memory.tools.Tool` subclasses, passed to
``CharacterAgent.generate_answer(tools=...)`` so the model can call them
mid-turn. They are **plain synchronous** tools: the bot already runs
generation inside ``asyncio.to_thread``, so ``run`` executes on a worker
thread and never blocks the event loop — no per-tool threading needed.

Two flavours:

* **Arch-flavoured** — search Arch Wiki / AUR / official repos, and fetch a
  wiki page. These reimplement the old Arch-chan tool methods against the
  library's synchronous ``Tool`` API (the legacy ``ToolResult`` +
  ``threading.Thread`` pattern is gone — the registry handles errors).
* **Web** — ``web_search`` (DuckDuckGo via ``ddgs``) and ``web_fetch``
  (clean page extraction via ``trafilatura``). These optional deps are
  imported lazily so a missing package degrades to a clear per-tool error
  rather than breaking the whole registry.

:func:`build_tools` wires everything up around one ``CharacterAgent`` and
returns a fresh list each call (drop it straight into ``generate_answer``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

import requests

from character_memory.tools.base import Tool

if TYPE_CHECKING:  # avoid an import-time dependency on the agent package
    from character_memory.agent import CharacterAgent


# --------------------------------------------------------------------------- #
# Shared HTTP + text helpers.
# --------------------------------------------------------------------------- #
_HTTP_TIMEOUT = 15.0  # seconds — generous enough for the MediaWiki/AUR APIs

_ARCH_WIKI_API = "https://wiki.archlinux.org/api.php"
_AUR_RPC = "https://aur.archlinux.org/rpc/"
_OFFICIAL_PKG = "https://archlinux.org/packages/search/json/"


def _http_get_json(url: str, params: Optional[dict[str, Any]] = None) -> Any:
    """GET ``url`` and return the decoded JSON, or raise with a clear message.

    A network failure / non-200 / bad JSON surfaces as an exception whose
    ``str`` is fed back to the model by the registry (as an ``ok=False``
    tool result) — so one bad request never crashes the turn.
    """
    resp = requests.get(url, params=params, timeout=_HTTP_TIMEOUT,
                        headers={"User-Agent": "Acchan-Discord-bot/1.0"})
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as exc:  # not JSON
        raise RuntimeError(f"bad JSON from {resp.url}: {exc}") from exc


def _clean(text: str) -> str:
    """Collapse blank runs and trim each line (for wiki-page markdown)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Trim trailing whitespace on every line, then collapse 3+ blank lines.
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks <= 1:
                out.append("")
            continue
        blanks = 0
        out.append(ln)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# search_character_info — the one built-in memory tool we keep.
# --------------------------------------------------------------------------- #
class SearchCharacterInfo(Tool):
    """Actively dig into the character's indexed Arch-chan information.

    The prompt already auto-recalls a few character_info hits each turn; this
    lets the model run a *second*, targeted lookup when it needs more detail
    on a specific topic (a package, a driver, an install step) than the
    auto-injected slice gave it. Recall is read-only (``state_changing=False``)
    so querying does not bump decay bookkeeping.
    """

    name = "search_character_info"
    description = (
        "Search the character's own indexed knowledge base (the Arch-chan "
        "information / wiki memory) for details on a topic — packages, "
        "drivers, install steps, config, etc. Use this to look something up "
        "before answering when you want more detail than you already recall."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    def run(self, query: str, limit: int = 5) -> str:
        mems = getattr(self.agent, "memories", {}) or {}
        mem = mems.get("character_info")
        if mem is None:
            return "character_info memory is not available."
        if not getattr(mem, "enabled", True):
            return "character_info memory is disabled."
        limit = max(1, min(20, int(limit)))
        try:
            items = mem.recall(query, "default", limit, state_changing=False)
        except Exception as e:  # noqa: BLE001 - surface to the model
            return f"character_info recall error: {e!r}"
        if not items:
            return "No matching entries in the character's knowledge base."
        return "\n".join(f"- {it.text}" for it in items if it.text)


# --------------------------------------------------------------------------- #
# search_arch_wiki
# --------------------------------------------------------------------------- #
class SearchArchWiki(Tool):
    name = "search_arch_wiki"
    description = (
        "Search the Arch Linux Wiki for pages matching a query. Returns up "
        "to 10 ranked results with title, link, and a snippet. Use this to "
        "find relevant wiki pages, then call get_wiki_page to read one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 10,
            "srqiprofile": "engine_autoselect",
        }
        data = _http_get_json(_ARCH_WIKI_API, params)
        results = data.get("query", {}).get("search", [])
        if not results:
            return "No results found on Arch Wiki."
        lines: list[str] = []
        for res in results:
            title = res.get("title", "")
            snippet = (res.get("snippet", "")
                       .replace('<span class="searchmatch">', "")
                       .replace("</span>", ""))
            url = f"https://wiki.archlinux.org/title/{title.replace(' ', '_')}"
            lines.append(f"### [{title}]({url})\n{snippet}…")
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# search_aur
# --------------------------------------------------------------------------- #
class SearchAur(Tool):
    name = "search_aur"
    description = (
        "Search the Arch User Repository (AUR) for packages matching a "
        "query. Returns up to 10 results, optionally sorted. Sort options: "
        "'relevance' (default), 'votes', 'popularity', 'modified'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Package name / keyword."},
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "votes", "popularity", "modified"],
                "default": "relevance",
                "description": "How to order results.",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, sort_by: str = "relevance") -> str:
        data = _http_get_json(_AUR_RPC, {"v": 5, "type": "search", "arg": query})
        results = data.get("results", [])

        if sort_by == "votes":
            results.sort(key=lambda x: x.get("NumVotes", 0), reverse=True)
        elif sort_by == "popularity":
            results.sort(key=lambda x: x.get("Popularity", 0), reverse=True)
        elif sort_by == "modified":
            results.sort(key=lambda x: x.get("LastModified", 0), reverse=True)
        # 'relevance' uses the API's own order.

        if not results:
            return "No results found in AUR."
        lines: list[str] = []
        for pkg in results[:10]:
            name = pkg.get("Name", "")
            version = pkg.get("Version", "")
            desc = pkg.get("Description", "")
            votes = pkg.get("NumVotes", 0)
            pop = pkg.get("Popularity", 0)
            lines.append(
                f"**{name}** {version} ({votes} votes, {pop} popularity)\n{desc}"
            )
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# get_official_package_info
# --------------------------------------------------------------------------- #
class GetOfficialPackageInfo(Tool):
    name = "get_official_package_info"
    description = (
        "Get details for an official Arch Linux package by exact name "
        "(version, repo, architecture, licenses, sizes, URL, maintainers). "
        "Use this for packages in the core/extra/multilib repos; AUR packages "
        "are not here — use search_aur for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": "Exact package name to look up.",
            },
        },
        "required": ["package_name"],
    }

    def run(self, package_name: str) -> str:
        data = _http_get_json(_OFFICIAL_PKG, {"name": package_name})
        results = data.get("results", [])
        if not results:
            return (f"Package '{package_name}' not found in the official "
                    f"repositories.")
        pkg = results[0]
        info = [
            f"# {pkg.get('pkgname')} {pkg.get('pkgver')}-{pkg.get('pkgrel')}",
            f"**Description:** {pkg.get('pkgdesc', '')}",
            f"**Repository:** {pkg.get('repo', '')}",
            f"**Architecture:** {pkg.get('arch', '')}",
            f"**URL:** {pkg.get('url', '')}",
            f"**Licenses:** {', '.join(pkg.get('licenses', []) or [])}",
            f"**Maintainers:** {', '.join(pkg.get('maintainers', []) or [])}",
            f"**Package Size:** "
            f"{(pkg.get('compressed_size', 0) or 0) / 1024 / 1024:.2f} MB",
            f"**Installed Size:** "
            f"{(pkg.get('installed_size', 0) or 0) / 1024 / 1024:.2f} MB",
        ]
        return "\n".join(info)


# --------------------------------------------------------------------------- #
# get_wiki_page
# --------------------------------------------------------------------------- #
class GetWikiPage(Tool):
    name = "get_wiki_page"
    description = (
        "Fetch and read an Arch Wiki page by title (returns the page text as "
        "markdown). Use search_arch_wiki first to find the right title."
    )
    parameters = {
        "type": "object",
        "properties": {
            "page_title": {
                "type": "string",
                "description": "Exact Arch Wiki page title (e.g. 'NVIDIA').",
            },
        },
        "required": ["page_title"],
    }

    def run(self, page_title: str) -> str:
        params = {
            "action": "parse",
            "page": page_title,
            "format": "json",
            "prop": "text",
            "disableeditsection": "1",
            "disabletoc": "1",
        }
        data = _http_get_json(_ARCH_WIKI_API, params)
        if "error" in data:
            err = data["error"].get("info") or data["error"]
            return f"Could not fetch '{page_title}': {err}"
        html = (data.get("parse", {}).get("text", {}) or {}).get("*")
        if not html:
            return (f"Page '{page_title}' has no readable content on "
                    f"Arch Wiki.")
        try:
            import markdownify
            content = markdownify.markdownify(html)
        except ImportError:
            content = html  # fall back to raw HTML
        return _clean(content) or f"Page '{page_title}' was empty after cleanup."


# --------------------------------------------------------------------------- #
# web_search — DuckDuckGo via ddgs.
# --------------------------------------------------------------------------- #
class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web with DuckDuckGo and return up to N results (default "
        "5), each with title, URL, and a short snippet. Use this for general "
        "lookups beyond the Arch Wiki / the character's own knowledge."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            return ("web_search unavailable: the 'ddgs' package is not "
                    "installed.")
        max_results = max(1, min(20, int(max_results)))
        try:
            results = DDGS().text(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001 - surface to the model
            return f"web search failed: {e!r}"
        if not results:
            return "No web results found."
        lines: list[str] = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "") or r.get("url", "")
            body = r.get("body", "")
            lines.append(f"### [{title}]({href})\n{body}")
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# web_fetch — clean page extraction via trafilatura.
# --------------------------------------------------------------------------- #
class WebFetch(Tool):
    name = "web_fetch"
    description = (
        "Download a web page and return its main text content (boilerplate / "
        "nav / ads stripped). Good for reading a specific article or doc page "
        "whose URL you already have (e.g. from web_search or search_arch_wiki). "
        "Optionally cap the length with max_chars."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The page URL to fetch."},
            "max_chars": {
                "type": "integer",
                "minimum": 100,
                "default": 8000,
                "description": "Maximum characters of text to return.",
            },
        },
        "required": ["url"],
    }

    def run(self, url: str, max_chars: int = 8000) -> str:
        try:
            import trafilatura
        except ImportError:
            return ("web_fetch unavailable: the 'trafilatura' package is not "
                    "installed.")
        max_chars = max(100, int(max_chars))
        try:
            downloaded = trafilatura.fetch_url(url)
        except Exception as e:  # noqa: BLE001 - surface to the model
            return f"fetch failed for {url}: {e!r}"
        if not downloaded:
            return (f"Could not download {url} (empty response or blocked).")
        try:
            text = trafilatura.extract(downloaded, include_links=False,
                                      include_tables=True)
        except Exception as e:  # noqa: BLE001 - surface to the model
            return f"extraction failed for {url}: {e!r}"
        if not text:
            return (f"No readable main text on {url}.")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n…[truncated]"
        return text


# --------------------------------------------------------------------------- #
# Factory.
# --------------------------------------------------------------------------- #
def build_tools(agent: "CharacterAgent") -> list[Tool]:
    """Build Acchan's tool set around one ``CharacterAgent``.

    Returns a fresh list each call; pass it to
    ``generate_answer(tools=…)`` (or wrap in a :class:`ToolRegistry`). The
    Arch/web tools are stateless; only :class:`SearchCharacterInfo` closes
    over the agent.
    """
    return [
        SearchCharacterInfo(agent),
        SearchArchWiki(),
        SearchAur(),
        GetOfficialPackageInfo(),
        GetWikiPage(),
        WebSearch(),
        WebFetch(),
    ]
