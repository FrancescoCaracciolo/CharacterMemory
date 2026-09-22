"""Interactive server configuration, without importing or starting the server."""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import os
from pathlib import Path
import secrets
import shlex
import tempfile
from urllib.parse import urlsplit

from ._dotenv import quote_value, read_text, read_values, update_text
from .config import EmbeddingConfig, LLMConfig

_SECRETS = {"OPENAI_API_KEY", "OPENAI_EMBEDDINGS_API_KEY", "CM_DATABASE_URL", "CM_API_KEY",
            "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"}


def _single_line(value: str) -> str:
    quote_value(value)
    return value


def _http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter a complete http:// or https:// API base URL, usually ending in /v1.")
    return value


def _database_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"postgres", "postgresql"} and (parsed.hostname or parsed.path.strip("/"))
        parsed.port  # Validate a supplied port without displaying credentials.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter a postgres:// or postgresql:// connection URL for an existing database.")
    return value


def _port(value: str) -> str:
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535.")
    return str(int(value))


def _directory(value: str) -> str:
    path = Path(value).expanduser()
    try:
        if path.exists() and not path.is_dir():
            raise ValueError("This path is a file; enter a directory.")
        parent = path
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError("A parent of this path is not a directory.")
    except OSError:
        raise ValueError("Enter a valid directory path.") from None
    return str(path) if value.startswith("~") else value


def _host(value: str) -> str:
    if any(char.isspace() for char in value) or any(char in value for char in "/?#@"):
        raise ValueError("Enter a bind address or hostname, without a URL scheme or path.")
    if ":" in value:
        import ipaddress
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            raise ValueError("Enter a hostname, IPv4 address, or unbracketed IPv6 address (port is separate).") from None
    return value


def _display(key: str, value: str | None) -> str:
    if value is None:
        return "inherit default"
    if key in _SECRETS:
        return "[configured]" if value else "[empty]"
    if "URL" in key:
        try:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                return "[configured URL]"
        except ValueError:
            return "[invalid URL]"
    return value


def _ask(label: str, key: str, default: str, validator=None, *, hidden=False, allow_empty=False) -> str:
    while True:
        suffix = f" [{_display(key, default)}]" if default else ""
        prompt = f"{label} ({key}){suffix}: "
        if hidden:
            # getpass otherwise falls back to echoing the secret on some terminals.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    value = getpass.getpass(prompt)
                except getpass.GetPassWarning:
                    raise ValueError("A terminal with hidden input is required for credentials.") from None
        else:
            value = input(prompt).strip()
        value = default if value == "" else value
        try:
            _single_line(value)
            if not value and not allow_empty:
                raise ValueError("A value is required.")
            return validator(value) if validator and value else value
        except ValueError as exc:
            print(f"  {exc}")


def _choice(label: str, choices: tuple[str, ...], default: str) -> str:
    if default not in choices:
        default = choices[0]
    while True:
        answer = input(f"{label} ({'/'.join(choices)}) [{default}]: ").strip().lower() or default
        if answer in choices:
            return answer
        print("  Choose " + ", ".join(choices) + ".")


def _secret(label: str, key: str, current: str, *, required=False, generate=False, validator=None) -> str:
    choices = ("keep", "change") if current else ("change",)
    if generate:
        choices += ("generate",)
    if not required:
        choices += ("clear",)
    default = "keep" if current else ("change" if required else "clear")
    while True:
        action = _choice(f"{label} ({key}; {'configured' if current else 'empty'})", choices, default)
        if action == "clear":
            return ""
        if action == "generate":
            return secrets.token_urlsafe(32)
        if action == "change":
            return _ask(label, key, "", validator, hidden=True, allow_empty=not required)
        try:
            _single_line(current)
            return validator(current) if validator else current
        except ValueError as exc:
            print(f"  {exc}")
            default = "change"


def _defaults() -> dict[str, str]:
    llm, embedding = LLMConfig(), EmbeddingConfig()
    return {
        "OPENAI_BASE_URL": llm.base_url, "OPENAI_MODEL": llm.model,
        "OPENAI_API_KEY": llm.api_key,
        "OPENAI_EMBEDDINGS_BASE_URL": embedding.base_url,
        "OPENAI_EMBEDDINGS_MODEL": embedding.model,
        "CM_STORAGE_BACKEND": "sqlite", "CM_DATABASE_URL": "",
        "CM_RETRIEVAL_BACKEND": "hybrid", "CM_ASSETS_DIR": "./assets",
        "CM_SAVE_DIR": "./.cm_servers", "CM_HOST": "0.0.0.0",
        "CM_PORT": "8000", "CM_API_KEY": "",
        "CM_DECISION_PROVIDER": "", "CM_DECISION_MODEL": "",
        "CM_DEDUP_ENABLED": "false",
    }


def collect_settings(current: dict[str, str]) -> dict[str, str | None]:
    """Collect a draft only; nothing is persisted or contacted here."""
    draft: dict[str, str | None] = {}
    print("\nLLM — OpenAI-compatible chat endpoint used for extraction and server chat.")
    draft["OPENAI_BASE_URL"] = _ask("API base URL", "OPENAI_BASE_URL", current["OPENAI_BASE_URL"], _http_url)
    draft["OPENAI_MODEL"] = _ask("Chat model", "OPENAI_MODEL", current["OPENAI_MODEL"])
    print("API keys may be empty for local endpoints that do not require authentication.")
    draft["OPENAI_API_KEY"] = _secret("LLM API key", "OPENAI_API_KEY", current["OPENAI_API_KEY"])

    print("\nEmbeddings — endpoint and model used to index and retrieve memories.")
    draft["OPENAI_EMBEDDINGS_BASE_URL"] = _ask("API base URL", "OPENAI_EMBEDDINGS_BASE_URL", current["OPENAI_EMBEDDINGS_BASE_URL"], _http_url)
    draft["OPENAI_EMBEDDINGS_MODEL"] = _ask("Embedding model", "OPENAI_EMBEDDINGS_MODEL", current["OPENAI_EMBEDDINGS_MODEL"])
    mode = _choice("Embedding API key (OPENAI_EMBEDDINGS_API_KEY)", ("shared", "separate"),
                   "separate" if "OPENAI_EMBEDDINGS_API_KEY" in current else "shared")
    draft["OPENAI_EMBEDDINGS_API_KEY"] = (
        _secret("Embedding API key", "OPENAI_EMBEDDINGS_API_KEY", current.get("OPENAI_EMBEDDINGS_API_KEY", ""))
        if mode == "separate" else None
    )

    print("\nDecision model — optional reconciliation of duplicate or corrected memories.")
    provider = _choice("Decision provider (CM_DECISION_PROVIDER)",
                       ("none", "typesafe", "openrouter", "llm"),
                       current.get("CM_DECISION_PROVIDER") or "none")
    draft["CM_DECISION_PROVIDER"] = "" if provider == "none" else provider
    draft["CM_DECISION_MODEL"] = ""
    draft["CM_DEDUP_ENABLED"] = current.get("CM_DEDUP_ENABLED", "false")
    if provider != "none":
        draft["CM_DEDUP_ENABLED"] = "true"
        if provider == "llm":
            print("Uses the configured chat model and LLM API key above.")
        else:
            from .decisions.clients import TypeSafeDecisionClient, OpenRouterDecisionClient
            client = TypeSafeDecisionClient if provider == "typesafe" else OpenRouterDecisionClient
            model = (current.get("CM_DECISION_MODEL")
                     if current.get("CM_DECISION_PROVIDER") == provider else None)
            draft["CM_DECISION_MODEL"] = _ask("Decision model", "CM_DECISION_MODEL", model or client.default_model)
            draft[client.key_env] = _secret("Decision API key", client.key_env,
                                           current.get(client.key_env, ""))
        print("Deduplication enabled by default. Existing per-character config.yaml settings take precedence.")

    print("\nDatabase — SQLite needs no database service; PostgreSQL uses an existing database.")
    backend = _choice("Storage (CM_STORAGE_BACKEND)", ("sqlite", "postgres"), current["CM_STORAGE_BACKEND"])
    draft["CM_STORAGE_BACKEND"] = backend
    draft["CM_DATABASE_URL"] = None
    draft["CM_RETRIEVAL_BACKEND"] = "hybrid"
    if backend == "postgres":
        draft["CM_DATABASE_URL"] = _secret("Database connection URL", "CM_DATABASE_URL", current["CM_DATABASE_URL"], required=True, validator=_database_url)
        print("hybrid: local FAISS/BM25 indexes; postgres: database search, requires pgvector >= 0.8 in public.")
        draft["CM_RETRIEVAL_BACKEND"] = _choice("Search indexes (CM_RETRIEVAL_BACKEND)", ("hybrid", "postgres"), current["CM_RETRIEVAL_BACKEND"])
    print("Database namespaces remain automatic per character unless already configured.")

    print("\nDirectories — relative paths resolve from the directory where you start the server.")
    draft["CM_ASSETS_DIR"] = _ask("Character assets directory", "CM_ASSETS_DIR", current["CM_ASSETS_DIR"], _directory)
    draft["CM_SAVE_DIR"] = _ask("Saved data and indexes directory", "CM_SAVE_DIR", current["CM_SAVE_DIR"], _directory)
    print("\nServer — bind address, listening port, and optional client authentication.")
    draft["CM_HOST"] = _ask("Bind host", "CM_HOST", current["CM_HOST"], _host)
    draft["CM_PORT"] = _ask("Bind port", "CM_PORT", current["CM_PORT"], _port)
    print("CM_API_KEY is the key clients send to this server; clear disables authentication.")
    draft["CM_API_KEY"] = _secret("Server API key", "CM_API_KEY", current["CM_API_KEY"], generate=True)
    return draft


def missing_dependencies(settings: dict[str, str | None]) -> list[str]:
    instructions = []
    if any(importlib.util.find_spec(name) is None for name in ("fastapi", "uvicorn", "multipart")):
        instructions.append("Install server dependencies: pip install 'charactermemory[server]'")
    if settings.get("CM_STORAGE_BACKEND") == "postgres" and any(
        importlib.util.find_spec(name) is None for name in ("psycopg", "psycopg_pool")
    ):
        instructions.append("Install PostgreSQL dependencies: pip install 'charactermemory[postgres]'")
    return instructions


def _check_llm(settings):
    from openai import OpenAI
    with OpenAI(base_url=settings["OPENAI_BASE_URL"], api_key=settings["OPENAI_API_KEY"], timeout=15, max_retries=0) as client:
        response = client.chat.completions.create(
            model=settings["OPENAI_MODEL"], messages=[{"role": "user", "content": "Reply OK."}], max_tokens=32,
        )
        if not response.choices:
            raise ValueError("Empty chat response")


def _check_embeddings(settings):
    from openai import OpenAI
    key = settings.get("OPENAI_EMBEDDINGS_API_KEY")
    if key is None:
        key = settings["OPENAI_API_KEY"]
    with OpenAI(base_url=settings["OPENAI_EMBEDDINGS_BASE_URL"], api_key=key, timeout=15, max_retries=0) as client:
        response = client.embeddings.create(model=settings["OPENAI_EMBEDDINGS_MODEL"], input=["connection test"])
        if not response.data or not response.data[0].embedding:
            raise ValueError("Empty embedding response")


def _check_postgres(settings):
    import psycopg
    # Do not instantiate PostgresStore: its constructor creates schemas/tables.
    with psycopg.connect(settings["CM_DATABASE_URL"], connect_timeout=10,
                         options="-c default_transaction_read_only=on -c statement_timeout=10000") as connection:
        connection.execute("SELECT 1").fetchone()
        if settings["CM_RETRIEVAL_BACKEND"] == "postgres":
            row = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector' "
                "AND extnamespace = 'public'::regnamespace"
            ).fetchone()
            if not row or tuple(int(part) for part in row[0].split(".")[:2]) < (0, 8):
                raise ValueError("pgvector >= 0.8 required in public")


def check_connections(settings: dict[str, str | None]) -> bool:
    """Run independent, bounded probes; never print service exception text."""
    checks = [("LLM", _check_llm), ("Embeddings", _check_embeddings)]
    if settings["CM_STORAGE_BACKEND"] == "postgres":
        checks.append(("PostgreSQL", _check_postgres))
    success = True
    for label, check in checks:
        try:
            check(settings)
            print(f"  {label}: connected.")
        except Exception as exc:
            success = False
            # Provider errors may contain URLs, passwords, keys, or response bodies.
            reason = "dependency missing" if isinstance(exc, ImportError) else (
                "request timed out" if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
                else "check endpoint, credentials, model, and service availability"
            )
            print(f"  {label}: failed ({reason}).")
            if label == "PostgreSQL" and settings["CM_RETRIEVAL_BACKEND"] == "postgres":
                print("    PostgreSQL search requires pgvector >= 0.8 installed in the public schema.")
    return success


def write_env(path: Path, content: str) -> None:
    """Replace atomically; failed writes leave the previous file in place."""
    fd, temporary = tempfile.mkstemp(prefix=".cm-env-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp creates mode 0600 on POSIX, including when replacing a file.
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_setup() -> int:
    path = Path.cwd() / ".env"
    original = read_text(path) if path.exists() else ""
    current = {**_defaults(), **read_values(original), **os.environ}
    print("CharacterMemory server setup")
    print(f"Configuration file: {path}")
    print("Enter keeps the displayed default. Ctrl-C cancels before saving.")
    print("Exported environment variables override .env; per-character config.yaml overrides library defaults.")
    print("Only configuration is saved. Services, characters, and database schemas are not created here.")
    while True:
        settings = collect_settings(current)
        for instruction in missing_dependencies(settings):
            print(instruction)
        if _choice("Test connections? Sends small LLM/embedding requests which may incur usage charges", ("yes", "no"), "no") == "yes":
            while not check_connections(settings):
                action = _choice("Connection tests failed", ("retry", "revise", "save"), "revise")
                if action == "save":
                    break
                if action == "revise":
                    break
            else:
                action = "save"
            if action == "revise":
                current = {**current, **{key: value for key, value in settings.items() if value is not None}}
                for key, value in settings.items():
                    if value is None:
                        current.pop(key, None)
                current.setdefault("CM_DATABASE_URL", "")
                continue
        print("\nConfiguration summary (credentials hidden):")
        for key, value in settings.items():
            print(f"  {key}: {_display(key, value)}")
        overridden = [key for key, value in settings.items() if key in os.environ and os.environ[key] != value]
        if overridden:
            print("Existing process environment differs for: " + ", ".join(overridden))
            print("If these values are exported in your shell, clear or update them before starting the server.")
        if _choice("Save .env?", ("yes", "no"), "yes") == "no":
            print("Cancelled; no configuration was saved.")
            return 0
        # Avoid silently replacing edits made in another terminal during prompts.
        if (read_text(path) if path.exists() else "") != original:
            raise ValueError(".env changed during setup. Run setup again to preserve those changes.")
        write_env(path, update_text(original, settings))
        break
    print(f"\nSaved {path} (credentials are stored in this file).")
    for key in ("CM_ASSETS_DIR", "CM_SAVE_DIR"):
        if not Path(settings[key]).is_dir():
            print(f"{key} does not exist yet: {settings[key]}")
    print("Put character folders under CM_ASSETS_DIR, or create characters in the GUI after starting.")
    print("The server creates its saved-data directories when needed.")
    print("Start from this directory:")
    print(f"  cd {shlex.quote(str(path.parent))}")
    print("  charactermemory-server")
    host = settings["CM_HOST"]
    if host in {"0.0.0.0", "::"}:
        host = "localhost"
    elif ":" in host:
        host = f"[{host}]"
    print(f"GUI: http://{host}:{settings['CM_PORT']}/gui")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="charactermemory-server-setup",
        description="Interactively configure the server in ./.env without starting it.",
    )
    parser.parse_args()
    try:
        code = run_setup()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; no configuration was saved.")
        code = 130
    except (OSError, ValueError):
        # Do not echo filesystem errors: a path can contain sensitive input.
        print("Setup could not save configuration. Check the .env path, permissions, concurrent edits, and terminal input.")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
