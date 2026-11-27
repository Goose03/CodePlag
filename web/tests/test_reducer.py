"""Reducer rules (see api.reducer docstring) and parity with the TypeScript reducer."""

from __future__ import annotations

import json
import random

import pytest

from api.reducer import MAX_LOGS, combine_many, combine_results, initial_state
from conftest import SHARED


def started(seq: int = 1, version: str = "v1") -> dict:
    return {
        "type": "job.started",
        "seq": seq,
        "files": [],
        "totalPairs": 3,
        "candidatePairs": 3,
        "prefilter": "none",
        "modelVersion": None,
        "combiner": {
            "name": "c",
            "method": "logistic",
            "features": ["ted"],
            "coef": [1.0],
            "intercept": 0.0,
            "threshold": 0.5,
            "version": version,
        },
    }


def partial(seq: int, key: str = "p1") -> dict:
    return {
        "type": "pair.partial",
        "seq": seq,
        "stage": "ast",
        "pairKey": key,
        "fileA": "a",
        "fileB": "b",
        "astFeatures": {"ted": 0.5},
        "astScore": 0.5,
    }


def scored(
    seq: int,
    key: str = "p1",
    verdict: str = "likely_plagiarism",
    version: str = "v1",
    score: float = 0.9,
) -> dict:
    return {
        "type": "pair.scored",
        "seq": seq,
        "stage": "final",
        "pairKey": key,
        "fileA": "a",
        "fileB": "b",
        "astFeatures": {"ted": 0.5},
        "astScore": 0.5,
        "neuralScore": 0.8,
        "score": score,
        "threshold": 0.5,
        "verdict": verdict,
        "detector": "combined",
        "combinerVersion": version,
        "matches": [],
        "warnings": [],
    }


def fold(events: list[dict]) -> dict:
    return combine_many(initial_state("j"), events)


EVENTS = [
    started(),
    partial(2),
    scored(3),
    partial(4, "p2"),
    scored(5, "p2", "unlikely_plagiarism", score=0.1),
    {"type": "progress", "seq": 6, "done": 2, "total": 3},
    {"type": "log", "seq": 7, "level": "info", "message": "hi", "ts": "t"},
    {"type": "job.completed", "seq": 8, "seconds": 1.5, "flagged": 1},
]


def test_fold_basic() -> None:
    s = fold(EVENTS)
    assert s["status"] == "completed"
    assert s["flagged"] == 1
    assert s["pairs"]["p1"]["stage"] == "final"
    assert s["pairs"]["p2"]["verdict"] == "unlikely_plagiarism"
    assert (s["done"], s["total"], s["elapsed"]) == (2, 3, 1.5)
    assert s["gap"] is False


def test_idempotent() -> None:
    assert fold(EVENTS + EVENTS) == fold(EVENTS)
    s = fold(EVENTS)
    assert combine_results(s, EVENTS[3]) is s


def test_order_independent() -> None:
    for seed in range(10):
        shuffled = EVENTS[:]
        random.Random(seed).shuffle(shuffled)
        assert fold(shuffled) == fold(EVENTS)


def test_partial_never_downgrades_final() -> None:
    s = fold([started(), scored(2)])
    s = combine_results(s, partial(3))
    assert s["pairs"]["p1"]["stage"] == "final"
    assert s["lastSeq"] == 3


def test_conflict_prefers_current_combiner_version() -> None:
    s = fold([started(version="v1"), scored(2, version="v1", score=0.9)])
    stale = combine_results(s, scored(3, version="old", verdict="unlikely_plagiarism"))
    assert stale["pairs"]["p1"]["score"] == 0.9
    newer = combine_results(s, scored(3, version="v1", verdict="unlikely_plagiarism", score=0.2))
    assert newer["pairs"]["p1"]["score"] == 0.2
    assert newer["flagged"] == 0  # verdict flipped: counter follows


def test_gap_detection() -> None:
    assert fold([started(), scored(5)])["gap"] is True


def test_terminal_state_only_accepts_logs() -> None:
    s = fold(EVENTS)
    after = combine_results(s, scored(9, "p3"))
    assert "p3" not in after["pairs"]
    after = combine_results(
        after, {"type": "log", "seq": 10, "level": "info", "message": "late", "ts": ""}
    )
    assert after["logs"][-1]["message"] == "late"


def test_logs_are_capped() -> None:
    events = [started()] + [
        {"type": "log", "seq": i, "level": "info", "message": str(i), "ts": ""}
        for i in range(2, MAX_LOGS + 50)
    ]
    assert len(fold(events)["logs"]) == MAX_LOGS


def test_input_not_mutated() -> None:
    s0 = fold(EVENTS[:2])
    snapshot = json.dumps(s0, sort_keys=True)
    combine_results(s0, EVENTS[2])
    assert json.dumps(s0, sort_keys=True) == snapshot


@pytest.mark.skipif(not (SHARED / "state_golden.json").is_file(), reason="no TS golden yet")
def test_parity_with_typescript() -> None:
    events = json.loads((SHARED / "events_golden.json").read_text(encoding="utf-8"))
    expected = json.loads((SHARED / "state_golden.json").read_text(encoding="utf-8"))
    assert combine_many(initial_state(expected["jobId"]), events) == expected
