"""On-disk job files: ``<jobs_dir>/<job_id>/manifest.json`` plus one file per sha256."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .validation import ValidFile

MANIFEST = "manifest.json"


def job_dir(jobs_dir: str | Path, job_id: str) -> Path:
    return Path(jobs_dir) / job_id


def save_job(jobs_dir: str | Path, job_id: str, files: list[ValidFile]) -> Path:
    """Write the sources and the manifest (order of ``files`` is preserved)."""
    root = job_dir(jobs_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        (root / f.sha256).write_bytes(f.code.encode("utf-8"))
    manifest = [{"name": f.name, "language": f.language, "sha256": f.sha256} for f in files]
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def load_job(jobs_dir: str | Path, job_id: str) -> list[ValidFile]:
    """Read back the files saved by :func:`save_job`.

    Raises:
        FileNotFoundError: If the job directory or manifest is missing (e.g. expired).
    """
    root = job_dir(jobs_dir, job_id)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    return [
        ValidFile(
            name=m["name"],
            language=m["language"],
            code=(root / m["sha256"]).read_bytes().decode("utf-8"),
            sha256=m["sha256"],
        )
        for m in manifest
    ]


def purge_old_jobs(jobs_dir: str | Path, max_age_seconds: int) -> int:
    """Delete job directories older than ``max_age_seconds``; returns how many were removed."""
    root = Path(jobs_dir)
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
