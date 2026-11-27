"""Background worker (arq). Loads the detector once per process and runs jobs.

Run with ``arq api.worker.WorkerSettings`` (from the ``web/`` directory, or with
``PYTHONPATH=web``).

:func:`execute_job` is idempotent and resumable: on a retry it folds the events already
written, skips the pairs that reached their final score and continues the ``seq``
numbering where the previous attempt stopped.
"""

from __future__ import annotations

import time
from typing import Any

from arq.connections import RedisSettings
from arq.cron import cron
from redis.asyncio import Redis

from codeplag.core.config import load_settings
from codeplag.core.logging import get_logger
from codeplag.similarity.detector import PlagiarismDetector

from .events import EventLog
from .reducer import combine_many, initial_state
from .settings import WebSettings, settings
from .storage import load_job, purge_old_jobs
from .streaming import iter_scan, log_event

logger = get_logger(__name__)
MAX_TRIES = 3


async def execute_job(
    redis: Redis,
    detector: PlagiarismDetector,
    job_id: str,
    cfg: WebSettings,
    attempt: int = 1,
) -> dict[str, Any]:
    """Run one job, publishing its events. Returns the final report state.

    Raises:
        Exception: Whatever the detector raised, after publishing ``job.failed`` (arq then
            retries up to :data:`MAX_TRIES` times).
    """
    log = EventLog(redis, job_id, cfg.job_ttl_seconds)
    state = combine_many(initial_state(job_id), await log.replay())
    if state["status"] in ("completed", "failed"):
        return state  # already finished: running it again must not change anything
    seq = state["lastSeq"]
    skip = frozenset(k for k, p in state["pairs"].items() if p["stage"] == "final")
    t0 = time.perf_counter()
    try:
        files = load_job(cfg.jobs_dir, job_id)
        for event in iter_scan(detector, files, skip, cfg.neural_batch):
            if event["type"] == "job.started" and state["status"] == "running":
                continue  # emitted by a previous attempt
            seq += 1
            state = combine_many(state, [await log.publish(event, seq)])
        seq += 1
        done = {
            "type": "job.completed",
            "seconds": round(time.perf_counter() - t0, 3),
            "flagged": state["flagged"],
        }
        state = combine_many(state, [await log.publish(done, seq)])
        return state
    except Exception as exc:
        logger.exception("Job %s failed (attempt %d)", job_id, attempt)
        retryable = attempt < MAX_TRIES
        seq += 1
        if retryable:
            # Not terminal: the retry will resume from here.
            await log.publish(log_event(f"Attempt {attempt} failed: {exc}", "error"), seq)
        else:
            await log.publish(
                {"type": "job.failed", "error": str(exc)[:500], "retryable": False}, seq
            )
        raise


async def startup(ctx: dict[str, Any]) -> None:
    detector = PlagiarismDetector(load_settings())
    _ = detector.scorer  # load the checkpoint now, not during the first job
    ctx["detector"] = detector
    logger.info("Worker ready (neural model: %s)", detector.scorer is not None)


async def run_job(ctx: dict[str, Any], job_id: str) -> str:
    state = await execute_job(
        ctx["redis"], ctx["detector"], job_id, settings, attempt=ctx.get("job_try", 1)
    )
    return str(state["status"])


async def cleanup(ctx: dict[str, Any]) -> None:
    removed = purge_old_jobs(settings.jobs_dir, settings.job_ttl_seconds)
    if removed:
        logger.info("Removed %d expired job directories", removed)


class WorkerSettings:
    """arq configuration."""

    functions = [run_job]
    cron_jobs = [cron(cleanup, minute=0)]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 1  # the model is not shared between concurrent jobs; scale with replicas
    max_tries = MAX_TRIES
    job_timeout = 1800
