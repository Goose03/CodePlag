"""Streaming scan, event log, worker idempotency/resume and the HTTP API."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import worker
from api.events import EventLog
from api.main import create_app
from api.reducer import combine_many, initial_state
from api.storage import load_job, save_job
from api.streaming import iter_scan, pair_key
from api.validation import ValidFile, safe_name
from codeplag.similarity.combiner import Combiner
from conftest import SHARED, FakeQueue, sample_files


def valid_samples() -> list[ValidFile]:
    import hashlib

    from codeplag.ast_analysis.parser import EXTENSIONS

    return [
        ValidFile(n, EXTENSIONS[Path(n).suffix], b.decode(), hashlib.sha256(b).hexdigest())
        for n, b in sample_files()
    ]


def test_pair_key_is_order_independent() -> None:
    assert pair_key("a" * 64, "b" * 64) == pair_key("b" * 64, "a" * 64)
    assert len(pair_key("a" * 64, "b" * 64)) == 16


def test_safe_name_strips_directories() -> None:
    assert safe_name("../../etc/passwd.py") == "passwd.py"
    assert safe_name("C:\\x\\y\\a.py") == "a.py"


def test_iter_scan_matches_cli_compare(detector) -> None:
    files = valid_samples()
    events = list(iter_scan(detector, files, neural_batch=2))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "job.started"
    n_pairs = len(files) * (len(files) - 1) // 2
    assert kinds.count("pair.partial") == kinds.count("pair.scored") == n_pairs
    # Every partial comes before the final result of the same pair.
    first = {}
    for i, e in enumerate(events):
        if e["type"].startswith("pair."):
            first.setdefault((e["pairKey"], e["type"]), i)
    for key, kind in list(first):
        if kind == "pair.scored":
            assert first[(key, "pair.partial")] < first[(key, "pair.scored")]
    # Same numbers as the CLI path (PlagiarismDetector.compare on the same files).
    by_name = {f.name: f for f in files}
    for e in (e for e in events if e["type"] == "pair.scored"):
        a, b = by_name[e["fileA"]], by_name[e["fileB"]]
        ref = detector.compare(
            detector.from_code(a.code, a.language, a.name),
            detector.from_code(b.code, b.language, b.name),
        )
        assert math.isclose(e["score"], ref.score, abs_tol=1e-12)
        assert e["verdict"] == ref.verdict
    assert events[-1] == {"type": "progress", "done": n_pairs, "total": n_pairs}


def test_iter_scan_skip_resumes(detector) -> None:
    files = valid_samples()
    first_key = pair_key(files[0].sha256, files[1].sha256)
    events = list(iter_scan(detector, files, skip=frozenset({first_key})))
    keys = {e["pairKey"] for e in events if e["type"] == "pair.scored"}
    assert first_key not in keys


def test_ast_only_reports_warning(ast_only_detector) -> None:
    events = list(iter_scan(ast_only_detector, valid_samples()))
    started = events[0]
    assert started["modelVersion"] is None
    scored = [e for e in events if e["type"] == "pair.scored"]
    assert all(e["neuralScore"] is None for e in scored)
    assert all(any("Neural model unavailable" in w for w in e["warnings"]) for e in scored)


def test_event_log_roundtrip(redis) -> None:
    async def run() -> None:
        log = EventLog(redis, "j" * 20)
        for seq in range(1, 4):
            await log.publish({"type": "progress", "done": seq, "total": 3}, seq)
        assert await log.last_seq() == 3
        assert [e["seq"] for e in await log.replay(after_seq=1)] == [2, 3]
        with pytest.raises(Exception):  # noqa: B017 - redis rejects non-increasing ids
            await log.publish({"type": "progress", "done": 0, "total": 3}, 2)
        await log.publish({"type": "job.completed", "seconds": 0.1, "flagged": 0}, 4)
        got = [e["seq"] async for e in log.tail(after_seq=2, block_ms=10)]
        assert got == [3, 4]

    asyncio.run(run())


def test_execute_job_is_idempotent_and_resumable(redis, detector, web_cfg) -> None:
    job = "a" * 20
    files = valid_samples()
    save_job(web_cfg.jobs_dir, job, files)
    assert [f.sha256 for f in load_job(web_cfg.jobs_dir, job)] == [f.sha256 for f in files]

    async def run() -> None:
        log = EventLog(redis, job)
        # Simulate a crashed first attempt that finished only the first 3 events.
        partial_events = list(iter_scan(detector, files, neural_batch=2))[:3]
        for seq, e in enumerate(partial_events, start=1):
            await log.publish(e, seq)
        state = await worker.execute_job(redis, detector, job, web_cfg, attempt=2)
        assert state["status"] == "completed"
        events = await log.replay()
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert sum(e["type"] == "job.started" for e in events) == 1
        n_pairs = len(files) * (len(files) - 1) // 2
        assert len(state["pairs"]) == n_pairs
        # Running it again changes nothing.
        again = await worker.execute_job(redis, detector, job, web_cfg)
        assert again == state
        assert len(await log.replay()) == len(events)

    asyncio.run(run())


def test_execute_job_failure_publishes_job_failed(redis, detector, web_cfg) -> None:
    async def run() -> None:
        with pytest.raises(FileNotFoundError):
            await worker.execute_job(redis, detector, "b" * 20, web_cfg, attempt=worker.MAX_TRIES)
        events = await EventLog(redis, "b" * 20).replay()
        assert events[-1]["type"] == "job.failed"

    asyncio.run(run())


@pytest.fixture
def client(redis_server, web_cfg):
    import fakeredis

    app = create_app(web_cfg, use_lifespan=False)
    # Separate client on the same server: TestClient runs its own event loop.
    app.state.redis = fakeredis.FakeAsyncRedis(server=redis_server)
    app.state.queue = FakeQueue()
    with TestClient(app) as c:
        yield c


def upload(client: TestClient, files: list[tuple[str, bytes]]):
    return client.post("/api/jobs", files=[("files", (n, b, "text/plain")) for n, b in files])


def test_api_full_flow(client, redis, detector, web_cfg) -> None:
    r = upload(client, sample_files())
    assert r.status_code == 202, r.text
    job = r.json()["jobId"]
    assert r.json()["deduplicated"] is False
    assert client.app.state.queue.jobs == [("run_job", (job,))]
    # Same contents again: deduplicated, not enqueued twice.
    assert upload(client, sample_files()).json() == {
        "jobId": job,
        "status": "exists",
        "deduplicated": True,
    }
    assert client.get(f"/api/jobs/{job}").json()["status"] == "queued"
    assert client.get(f"/api/jobs/{job}/report").status_code == 404

    asyncio.run(worker.execute_job(redis, detector, job, web_cfg))

    report = client.get(f"/api/jobs/{job}/report").json()
    assert report["status"] == "completed"
    assert report["flagged"] == sum(
        p["verdict"] == "likely_plagiarism" for p in report["pairs"].values()
    )
    status = client.get(f"/api/jobs/{job}").json()
    assert status["status"] == "completed"

    with client.stream("GET", f"/api/jobs/{job}/events", headers={"Last-Event-ID": "2"}) as s:
        body = "".join(s.iter_text())
    ids = [int(line[4:]) for line in body.splitlines() if line.startswith("id: ")]
    assert ids[0] == 3 and ids == sorted(ids)
    assert "event: job.completed" in body

    with client.stream("GET", f"/api/jobs/{job}/events?after={report['lastSeq'] - 1}") as s:
        body = "".join(s.iter_text())
    assert [line for line in body.splitlines() if line.startswith("id: ")] == [
        f"id: {report['lastSeq']}"
    ]

    fid = report["files"][0]["fileId"]
    src = client.get(f"/api/jobs/{job}/files/{fid}")
    assert src.status_code == 200 and src.text
    assert client.get(f"/api/jobs/{job}/files/..%2F..%2Fx").status_code in (400, 404)

    pdf = client.get(f"/api/jobs/{job}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ([("a.py", b"x=1")], 422),
        ([("a.py", b"x=1"), ("b.txt", b"y")], 415),
        ([("a.py", b"x=1"), ("b.py", b"\xff\xfe\x00")], 422),
        ([("a.py", b"x=1"), ("b.py", b"x=1")], 422),  # identical content
        ([("a.py", b"x=1"), ("b.py", b"#" * (600 * 1024))], 413),
    ],
)
def test_api_rejects_bad_uploads(client, files, status) -> None:
    assert upload(client, files).status_code == status


def test_api_rejects_bad_ids(client) -> None:
    assert client.get("/api/jobs/not-a-job/report").status_code == 400
    assert client.get(f"/api/jobs/{'c' * 20}/report").status_code == 404
    assert client.get(f"/api/jobs/{'c' * 20}/events").status_code == 404


def test_pdf_requires_completed_job(client, redis) -> None:
    job = "d" * 20

    async def seed() -> None:
        await EventLog(redis, job).publish({"type": "progress", "done": 0, "total": 1}, 1)

    asyncio.run(seed())
    assert client.get(f"/api/jobs/{job}/report.pdf").status_code == 409


def test_check_job_id() -> None:
    from api.validation import check_job_id

    with pytest.raises(HTTPException):
        check_job_id("../x")


@pytest.mark.skipif(not (SHARED / "combiner_golden.json").is_file(), reason="no golden yet")
def test_combiner_golden_matches_python() -> None:
    for case in json.loads((SHARED / "combiner_golden.json").read_text(encoding="utf-8")):
        c = {k: v for k, v in case["combiner"].items() if k != "version"}
        got = Combiner(**c).score(case["features"])
        assert math.isclose(got, case["expected"], rel_tol=0, abs_tol=1e-12)


@pytest.mark.skipif(not (SHARED / "events_golden.json").is_file(), reason="no golden yet")
def test_events_golden_is_consistent() -> None:
    events = json.loads((SHARED / "events_golden.json").read_text(encoding="utf-8"))
    state = combine_many(initial_state(events[0]["jobId"]), events)
    assert state["status"] == "completed" and not state["gap"]
