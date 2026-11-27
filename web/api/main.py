"""HTTP API. It validates uploads, stores them, enqueues jobs and relays job events.

It never loads PyTorch: the detector runs in the worker (:mod:`api.worker`).
Run with ``uvicorn api.main:app`` from the ``web/`` directory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from redis.asyncio import Redis

from codeplag.core.config import load_settings
from codeplag.similarity.combiner import load_combiners

from .events import EventLog, lock_key
from .pdf import render_pdf
from .reducer import combine_many, initial_state
from .schemas import combiner_params
from .settings import WebSettings, settings
from .storage import job_dir, save_job
from .validation import SHA_RE, check_job_id, validate_uploads


class JobQueue(Protocol):
    """What the API needs from arq (lets tests inject a fake queue)."""

    async def enqueue_job(self, function: str, *args: Any, _job_id: str | None = None) -> Any: ...


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from arq import create_pool
    from arq.connections import RedisSettings

    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.redis = pool  # ArqRedis is a redis.asyncio.Redis client too
    app.state.queue = pool
    yield
    await pool.aclose()


def create_app(cfg: WebSettings = settings, use_lifespan: bool = True) -> FastAPI:
    """Build the FastAPI app. Tests pass ``use_lifespan=False`` and set ``app.state``."""
    app = FastAPI(
        title="CodePlag API", version="1.0.0", lifespan=lifespan if use_lifespan else None
    )
    app.state.cfg = cfg
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_redis(request: Request) -> Redis:
        return request.app.state.redis

    def get_queue(request: Request) -> JobQueue:
        return request.app.state.queue

    async def fold(redis: Redis, job_id: str) -> dict[str, Any]:
        events = await EventLog(redis, job_id, cfg.job_ttl_seconds).replay()
        if not events:
            exists = await redis.exists(lock_key(job_id))
            raise HTTPException(404, "Job pending." if exists else "Job not found or expired.")
        return combine_many(initial_state(job_id), events)

    @app.get("/healthz")
    async def healthz(redis: Redis = Depends(get_redis)) -> dict[str, Any]:
        s = load_settings()
        ok = bool(await redis.ping())
        return {
            "ok": ok,
            "combiner": s.paths.combiner_path.is_file(),
            "model": (s.paths.checkpoints_dir / s.neural.checkpoint_name).is_file(),
        }

    @app.get("/api/combiner")
    async def combiner() -> dict[str, Any]:
        """Fitted combiners (the browser uses them for what-if re-scoring)."""
        path = load_settings().paths.combiner_path
        if not path.is_file():
            raise HTTPException(404, "No fitted combiner; the detector uses weighted fallbacks.")
        return {name: combiner_params(c) for name, c in load_combiners(path).items()}

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        files: list[UploadFile] = File(...),
        redis: Redis = Depends(get_redis),
        queue: JobQueue = Depends(get_queue),
    ) -> dict[str, Any]:
        valid = await validate_uploads(files, cfg)
        # Same contents => same job: re-uploading returns the existing report.
        digest = hashlib.sha256("|".join(sorted(f.sha256 for f in valid)).encode()).hexdigest()
        job_id = digest[:20]
        if not await redis.set(lock_key(job_id), "1", nx=True, ex=cfg.job_ttl_seconds):
            return {"jobId": job_id, "status": "exists", "deduplicated": True}
        save_job(cfg.jobs_dir, job_id, valid)
        await queue.enqueue_job("run_job", job_id, _job_id=job_id)  # arq ignores duplicates
        return {"jobId": job_id, "status": "queued", "deduplicated": False}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str, redis: Redis = Depends(get_redis)) -> dict[str, Any]:
        check_job_id(job_id)
        try:
            s = await fold(redis, job_id)
        except HTTPException as exc:
            if exc.detail == "Job pending.":
                return {"jobId": job_id, "status": "queued", "done": 0, "total": 0, "flagged": 0}
            raise
        return {k: s[k] for k in ("jobId", "status", "done", "total", "flagged", "error")}

    @app.get("/api/jobs/{job_id}/report")
    async def report(job_id: str, redis: Redis = Depends(get_redis)) -> dict[str, Any]:
        """Snapshot = fold of every event so far (same reducer as the browser)."""
        return await fold(redis, check_job_id(job_id))

    @app.get("/api/jobs/{job_id}/events")
    async def events(
        job_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None),
        redis: Redis = Depends(get_redis),
    ) -> StreamingResponse:
        check_job_id(job_id)
        known = await redis.exists(lock_key(job_id))
        if not known and not await EventLog(redis, job_id).last_seq():
            raise HTTPException(404, "Job not found or expired.")
        # Last-Event-ID on automatic reconnections; ?after= on the first connection.
        raw = last_event_id or request.query_params.get("after", "0")
        after = int(raw) if raw.isdigit() else 0
        log = EventLog(redis, job_id, cfg.job_ttl_seconds)

        async def stream() -> AsyncIterator[str]:
            yield "retry: 3000\n\n"
            async for ev in log.tail(after, block_ms=cfg.heartbeat_ms):
                if await request.is_disconnected():
                    return
                if ev["type"] == "_heartbeat":
                    yield ": ping\n\n"
                    continue
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/files/{file_id}", response_class=PlainTextResponse)
    async def file_source(job_id: str, file_id: str) -> str:
        check_job_id(job_id)
        if not SHA_RE.match(file_id):
            raise HTTPException(400, "Invalid file id.")
        path = job_dir(cfg.jobs_dir, job_id) / file_id
        if not path.is_file():
            raise HTTPException(404, "File not found or expired.")
        return Path(path).read_text(encoding="utf-8")

    @app.get("/api/jobs/{job_id}/report.pdf")
    async def report_pdf(job_id: str, redis: Redis = Depends(get_redis)) -> Response:
        state = await fold(redis, check_job_id(job_id))
        if state["status"] != "completed":
            raise HTTPException(409, "The PDF is available when the job completes.")
        return Response(
            render_pdf(state),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="codeplag-{job_id}.pdf"'},
        )

    return app


app = create_app()
