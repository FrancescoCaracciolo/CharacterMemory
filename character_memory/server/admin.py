"""Admin API: create / configure / delete characters and chat with them.

This is the write-side companion to :mod:`character_memory.server.adapters`
(read side) and the GUI configurator at ``/gui`` (Configure tab). It backs
``character_memory/server/static/config.js``.

The router is built by :func:`build_admin_router`, which receives the live
``AGENTS`` registry plus the assets/save roots so every mutation is reflected
in the running server in-place — no restart needed. Persisted state lives in:

* ``<character_dir>/character.json``      — persona + memory toggles (manifest)
* ``<character_dir>/.knowledge_graph``    — marker file opting the KG on
* ``<character_dir>/Information/*.md``    — wiki chunks source
* ``<character_dir>/Dialogues/*.md``      — dialogue chunks source
* ``<save_root>/<name>/``                 — built indexes + SQLite memory.db

Long-running builds (re-index + optional KG extraction) run on a background
thread; progress is reported through the in-memory ``JOBS`` map polled by the
frontend via ``GET /api/jobs/{job_id}``.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Optional

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from character_memory import (
    CharacterAgent,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
)
from character_memory.character_config import (
    default_config_yaml,
    load_config,
    path_for as config_path_for,
    save_config,
)
from character_memory.manifest import MEMORY_NAMES

# Two file buckets the GUI knows about. Map to the on-disk directories the
# chunkers read (see CharacterAgent._INFO_GLOB / _DIALOGUE_GLOB).
BUCKETS = {
    "information": "Information",
    "dialogues": "Dialogues",
}

# Memories the GUI can toggle. KG is handled through the `.knowledge_graph`
# marker rather than the MemoryConfig toggle so discovery stays the single
# source of truth.
TOGGLEABLE_MEMORIES = tuple(n for n in MEMORY_NAMES if n != "knowledge_graph")

# Name validation: letters/digits/space/dash/underscore/dot only. Anything else
# would either collide with relative-path traversal or break on filesystems
# that hate weird characters.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,62}$")


# --------------------------------------------------------------------------- #
# In-memory job runner (rebuild progress).
# --------------------------------------------------------------------------- #
class _Job:
    """A background rebuild job. Updated by the worker thread, read by polling."""

    __slots__ = ("id", "character", "state", "stage", "progress", "detail", "started", "ended")

    def __init__(self, job_id: str, character: str) -> None:
        self.id = job_id
        self.character = character
        self.state = "pending"        # pending | running | done | error
        self.stage = "queued"
        self.progress = 0.0           # 0..1
        self.detail = ""
        self.started = time.time()
        self.ended: Optional[float] = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "character": self.character,
            "state": self.state,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "detail": self.detail,
            "started": self.started,
            "ended": self.ended,
        }


JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()       # guards the JOBS dict itself
# One rebuild at a time per character; concurrent rebuilds on the same agent
# would corrupt its indexes. The lock is process-wide — fine for a single
# uvicorn worker (the documented deployment).
_REBUILD_LOCKS: dict[str, threading.Lock] = {}
_REBUILD_LOCKS_LOCK = threading.Lock()


def _rebuild_lock(name: str) -> threading.Lock:
    with _REBUILD_LOCKS_LOCK:
        lock = _REBUILD_LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _REBUILD_LOCKS[name] = lock
        return lock


# --------------------------------------------------------------------------- #
# Request / response schemas.
# --------------------------------------------------------------------------- #
class CreateCharacterRequest(BaseModel):
    name: str = Field(..., description="Character name (becomes the folder name).")


class ConfigMemoryPatch(BaseModel):
    # Allow arbitrary enabled_<name> / <name>_k fields without modelling each.
    model_config = {"extra": "allow"}


class ConfigPatch(BaseModel):
    persona: Optional[str] = None
    kg_enabled: Optional[bool] = None
    memory: Optional[dict[str, Any]] = None


class FileWriteRequest(BaseModel):
    content: str


class ChatRequest(BaseModel):
    message: str
    user: str = "user"
    chat_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Router builder.
# --------------------------------------------------------------------------- #
def build_admin_router(
    agents: "dict[str, CharacterAgent]",
    assets_dir: str,
    save_root: str,
) -> APIRouter:
    """Return an :class:`APIRouter` implementing the configurator contract.

    ``agents`` is the live registry shared with the rest of the server — every
    mutation (create / delete / config save / rebuild) updates it in place.
    """
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    # ----------------------------- helpers ----------------------------- #
    def _char_dir(name: str) -> str:
        return os.path.join(assets_dir, name)

    def _save_dir(name: str) -> str:
        return os.path.join(save_root, name)

    def _require_name(name: str) -> None:
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid character name. Use 1-63 letters/digits/spaces, "
                    "'-', '_' or '.'; must start with a letter or digit."
                ),
            )

    def _require_existing(name: str) -> CharacterAgent:
        _require_name(name)
        agent = agents.get(name)
        if agent is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown character {name!r}.",
            )
        return agent

    def _bucket_dir(name: str, bucket: str) -> str:
        if bucket not in BUCKETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown bucket {bucket!r}. Use one of {sorted(BUCKETS)}.",
            )
        return os.path.join(_char_dir(name), BUCKETS[bucket])

    def _scan_character(name: str) -> dict[str, Any]:
        info_dir = os.path.join(_char_dir(name), BUCKETS["information"])
        dlg_dir = os.path.join(_char_dir(name), BUCKETS["dialogues"])
        marker = os.path.join(_char_dir(name), ".knowledge_graph")
        has_built = os.path.isfile(os.path.join(_save_dir(name), "memory.db"))
        return {
            "name": name,
            "has_info": bool(os.path.isdir(info_dir) and os.listdir(info_dir)),
            "has_dialogue": bool(os.path.isdir(dlg_dir) and os.listdir(dlg_dir)),
            "has_kg": os.path.isfile(marker),
            "has_built": has_built,
        }

    def _reload_agent(name: str, *, rebuild_indexes: bool = False) -> CharacterAgent:
        """Drop the cached agent and rebuild it from disk (config.yaml + files).

        ``rebuild_indexes=False`` keeps the existing persisted indexes (cheap —
        used after a config change). ``True`` forces a full re-chunk + re-index
        via :meth:`CharacterAgent.rebuild`.
        """
        char_dir = _char_dir(name)
        config_path = config_path_for(char_dir)
        old = agents.get(name)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        agent = CharacterAgent(
            directory=char_dir,
            name=name,
            save_directory=_save_dir(name),
        )
        # Prefer config.yaml; fall back to defaults when absent (legacy
        # character with only Information/ + Dialogues/ on disk).
        if os.path.isfile(config_path):
            agent.load_from_config(config_path)
            if os.path.isfile(os.path.join(char_dir, ".knowledge_graph")) and agent.config is not None:
                agent.config.memory.enabled_knowledge_graph = True
                agent.load_from_config(agent.config)
        else:
            mem_cfg = MemoryConfig()
            if os.path.isfile(os.path.join(char_dir, ".knowledge_graph")):
                mem_cfg.enabled_knowledge_graph = True
            agent.load_from_config(LLMConfig(), EmbeddingConfig(), mem_cfg)
        if rebuild_indexes:
            agent.rebuild()
        else:
            agent.build()
        agents[name] = agent
        return agent

    # ----------------------------- characters ----------------------------- #
    @router.get("/characters")
    def list_characters() -> dict:
        """Every character folder + which inputs it has + whether it's built."""
        if not os.path.isdir(assets_dir):
            return {"characters": []}
        rows = []
        for name in sorted(os.listdir(assets_dir)):
            if not os.path.isdir(_char_dir(name)):
                continue
            rows.append(_scan_character(name))
        return {"characters": rows}

    @router.post("/characters", status_code=201)
    def create_character(req: CreateCharacterRequest) -> dict:
        name = req.name.strip()
        _require_name(name)
        if os.path.exists(_char_dir(name)):
            raise HTTPException(
                status_code=409,
                detail=f"A character named {name!r} already exists.",
            )
        # Scaffold the folder layout the chunkers expect + a fresh config.yaml
        # carrying every default (prompts + sub-configs) the GUI can later edit.
        os.makedirs(os.path.join(_char_dir(name), BUCKETS["information"]))
        os.makedirs(os.path.join(_char_dir(name), BUCKETS["dialogues"]))
        config_path = config_path_for(_char_dir(name))
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(default_config_yaml(name=name))
        agent = _reload_agent(name)
        return _scan_character(agent.character_name)

    @router.delete("/characters/{name}")
    def delete_character(name: str) -> dict:
        _require_existing(name)
        agent = agents.pop(name, None)
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
        char_dir = _char_dir(name)
        if os.path.isdir(char_dir):
            shutil.rmtree(char_dir)
        save_dir = _save_dir(name)
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        return {"ok": True, "deleted": name}

    # ----------------------------- config (persona + memories) ----------------------------- #
    @router.get("/characters/{name}/config")
    def get_config(name: str) -> dict:
        _require_existing(name)
        loaded = load_config(_char_dir(name))
        mem = loaded.config.memory
        memory_view = {}
        for m in MEMORY_NAMES:
            memory_view[f"enabled_{m}"] = getattr(mem, f"enabled_{m}", False)
            if hasattr(mem, f"{m}_k"):
                memory_view[f"{m}_k"] = getattr(mem, f"{m}_k")
        marker = os.path.join(_char_dir(name), ".knowledge_graph")
        return {
            "name": name,
            "persona": loaded.persona,
            "kg_enabled": os.path.isfile(marker) or mem.enabled_knowledge_graph,
            "memory": memory_view,
        }

    @router.put("/characters/{name}/config")
    def put_config(name: str, patch: ConfigPatch) -> dict:
        _require_existing(name)
        char_dir = _char_dir(name)

        # Load the current file (or defaults), apply the patch, persist back.
        loaded = load_config(char_dir)
        cfg = loaded.config
        prompts = loaded.prompts
        persona = loaded.persona
        if patch.persona is not None:
            persona = patch.persona
        if patch.memory:
            mem = cfg.memory
            for key, val in patch.memory.items():
                if key.startswith("enabled_") and key[len("enabled_"):] in MEMORY_NAMES:
                    setattr(mem, key, bool(val))
                elif key.endswith("_k") and key[:-2] in MEMORY_NAMES:
                    try:
                        setattr(mem, key, int(val))
                    except (TypeError, ValueError):
                        pass
        # KG toggle: set the YAML field AND manage the `.knowledge_graph`
        # marker the discovery step consults.
        marker = os.path.join(char_dir, ".knowledge_graph")
        if patch.kg_enabled is True:
            cfg.memory.enabled_knowledge_graph = True
            if not os.path.isfile(marker):
                with open(marker, "w", encoding="utf-8") as f:
                    f.write("# knowledge graph enabled via the configurator\n")
        elif patch.kg_enabled is False:
            cfg.memory.enabled_knowledge_graph = False
            if os.path.isfile(marker):
                os.remove(marker)

        save_config(
            char_dir,
            config=cfg,
            prompts=prompts,
            persona=persona,
            name=name,
        )
        # Reload so toggles/persona take effect in the live agent.
        _reload_agent(name)
        return get_config(name)

    # ----------------------------- files ----------------------------- #
    @router.get("/characters/{name}/files")
    def list_files(name: str, bucket: str = Query(...)) -> dict:
        _require_existing(name)
        bdir = _bucket_dir(name, bucket)
        files = []
        if os.path.isdir(bdir):
            for fname in sorted(os.listdir(bdir)):
                fp = os.path.join(bdir, fname)
                if os.path.isfile(fp):
                    files.append({"name": fname, "size": os.path.getsize(fp)})
        return {"bucket": bucket, "files": files}

    @router.get(
        "/characters/{name}/files/{bucket}/{file:path}",
        response_class=PlainTextResponse,
    )
    def read_file(name: str, bucket: str, file: str) -> PlainTextResponse:
        _require_existing(name)
        fp = os.path.join(_bucket_dir(name, bucket), file)
        if not os.path.isfile(fp):
            raise HTTPException(status_code=404, detail=f"No such file {file!r}.")
        with open(fp, encoding="utf-8") as f:
            return PlainTextResponse(f.read())

    @router.put("/characters/{name}/files/{bucket}/{file:path}")
    def write_file(name: str, bucket: str, file: str, req: FileWriteRequest) -> dict:
        _require_existing(name)
        # Keep names sane: no traversal, must look like a wiki/dialogue file.
        if "/" in file or ".." in file:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        bdir = _bucket_dir(name, bucket)
        os.makedirs(bdir, exist_ok=True)
        fp = os.path.join(bdir, file)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(req.content)
        return {"ok": True, "name": file, "size": os.path.getsize(fp)}

    @router.post("/characters/{name}/files/{bucket}")
    async def upload_files(
        name: str,
        bucket: str,
        files: list[UploadFile] = File(...),
    ) -> dict:
        _require_existing(name)
        bdir = _bucket_dir(name, bucket)
        os.makedirs(bdir, exist_ok=True)
        saved = []
        for upload in files:
            fname = os.path.basename(upload.filename or "upload.md")
            if not fname or "/" in fname or ".." in fname:
                continue
            fp = os.path.join(bdir, fname)
            data = await upload.read()
            with open(fp, "wb") as f:
                f.write(data)
            saved.append({"name": fname, "size": len(data)})
        return {"ok": True, "bucket": bucket, "files": saved}

    @router.delete("/characters/{name}/files/{bucket}/{file:path}")
    def delete_file(name: str, bucket: str, file: str) -> dict:
        _require_existing(name)
        fp = os.path.join(_bucket_dir(name, bucket), file)
        if not os.path.isfile(fp):
            raise HTTPException(status_code=404, detail=f"No such file {file!r}.")
        os.remove(fp)
        return {"ok": True, "deleted": file}

    # ----------------------------- rebuild (async job) ----------------------------- #
    @router.post("/characters/{name}/rebuild")
    def rebuild(name: str) -> dict:
        _require_existing(name)
        job_id = uuid.uuid4().hex[:16]
        job = _Job(job_id, name)
        with _JOBS_LOCK:
            JOBS[job_id] = job

        def _work() -> None:
            lock = _rebuild_lock(name)
            acquired = lock.acquire(timeout=2)
            if not acquired:
                job.state = "error"
                job.stage = "busy"
                job.detail = "Another rebuild is already running for this character."
                job.ended = time.time()
                return
            try:
                job.state = "running"
                job.stage = "loading"
                job.progress = 0.05
                # Reload from disk so freshly-uploaded files + toggles apply.
                agent = _reload_agent(name)
                job.stage = "indexing"
                job.progress = 0.25
                if "knowledge_graph" in agent.memories:
                    job.detail = "Rebuilding indexes + knowledge graph (LLM extraction; slow)."
                else:
                    job.detail = "Rebuilding RAG indexes."
                agent.rebuild()
                job.stage = "persisting"
                job.progress = 0.9
                agent.persist_structured()
                job.state = "done"
                job.stage = "done"
                job.progress = 1.0
                job.detail = "Indexes rebuilt."
            except Exception as exc:  # noqa: BLE001 - surface to the poller
                job.state = "error"
                job.stage = "failed"
                job.detail = f"{type(exc).__name__}: {exc}"
            finally:
                job.ended = time.time()
                lock.release()

        threading.Thread(target=_work, daemon=True).start()
        return {"job_id": job_id, "character": name}

    # ----------------------------- mini chat ----------------------------- #
    @router.post("/characters/{name}/chat")
    def chat(name: str, req: ChatRequest) -> dict:
        """Generate a reply server-side (the configurator's mini-chat).

        Unlike the thin-client ``/context`` + ``/save`` flow, this runs the LLM
        on the server. The user turn is persisted and the reply is stored +
        (on the extract interval) extracted.
        """
        agent = _require_existing(name)
        if req.chat_id:
            chat = agent.load_chat(req.chat_id)
            if chat is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown chat_id {req.chat_id!r}.",
                )
        else:
            chat = agent.create_chat(req.user, title=req.message[:60])
        chat.add_message("user", req.message, user_id=req.user)
        reply = agent.generate_answer(chat, save=True, user_id=req.user)
        agent.persist_structured()
        return {"chat_id": chat.id, "reply": reply}

    return router


# --------------------------------------------------------------------------- #
# Jobs polling endpoint — mounted at the app root (no /api/admin prefix).
# --------------------------------------------------------------------------- #
def build_jobs_router() -> APIRouter:
    """Router exposing ``GET /api/jobs/{job_id}`` for rebuild progress polling."""
    router = APIRouter(prefix="/api/jobs", tags=["admin"])

    @router.get("/{job_id}")
    def get_job(job_id: str) -> dict:
        with _JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}.")
        snap = job.snapshot()
        # Reap finished jobs after a grace period so the map stays bounded.
        if job.state in {"done", "error"} and job.ended and time.time() - job.ended > 300:
            with _JOBS_LOCK:
                JOBS.pop(job_id, None)
        return snap

    return router
