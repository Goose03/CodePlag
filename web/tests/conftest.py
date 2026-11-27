"""Fixtures for the web API tests: in-memory Redis, a fake queue and a detector without
a neural checkpoint (plus a deterministic fake neural scorer)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

WEB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WEB_DIR.parent
SHARED = WEB_DIR / "shared"
SAMPLES = REPO_DIR / "data" / "samples"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import fakeredis  # noqa: E402

from api.settings import WebSettings  # noqa: E402
from codeplag.core.config import Settings  # noqa: E402
from codeplag.similarity.detector import PlagiarismDetector  # noqa: E402


class FakeScorer:
    """Deterministic stand-in for NeuralScorer: similarity from length ratio."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return np.array(
            [min(len(a), len(b)) / max(len(a), len(b), 1) for a, b in pairs], dtype=float
        )

    def score(self, a: str, b: str) -> float:
        return float(self.score_pairs([(a, b)])[0])


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple[Any, ...]]] = []

    async def enqueue_job(self, function: str, *args: Any, _job_id: str | None = None) -> None:
        self.jobs.append((function, args))


@pytest.fixture
def redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def redis(redis_server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    """Client for code run with ``asyncio.run`` in the test itself."""
    return fakeredis.FakeAsyncRedis(server=redis_server)


@pytest.fixture
def web_cfg(tmp_path: Path) -> WebSettings:
    return WebSettings(jobs_dir=str(tmp_path / "jobs"), neural_batch=2, heartbeat_ms=50)


@pytest.fixture
def detector(tmp_path: Path) -> PlagiarismDetector:
    s = Settings()
    s.paths.checkpoints_dir = tmp_path / "ckpt"
    s.paths.combiner_path = tmp_path / "ckpt" / "combiner.json"
    s.scan.workers = 1
    return PlagiarismDetector(s, scorer=FakeScorer(), combiners={})  # type: ignore[arg-type]


@pytest.fixture
def ast_only_detector(tmp_path: Path) -> PlagiarismDetector:
    s = Settings()
    s.paths.checkpoints_dir = tmp_path / "ckpt"
    s.paths.combiner_path = tmp_path / "ckpt" / "combiner.json"
    return PlagiarismDetector(s, combiners={}, use_neural=False)


def sample_files() -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.py"))]
