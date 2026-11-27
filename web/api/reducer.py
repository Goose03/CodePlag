"""Fold job events into a report state.

Same semantics as ``web/src/lib/report-reducer.ts`` (checked by the parity tests):

1. Idempotent: an event whose ``seq`` is ``<= lastSeq`` is ignored.
2. A jump in ``seq`` sets ``gap`` (the client then refetches the server snapshot).
3. A pair that reached the ``final`` stage never goes back to ``ast``.
4. Two ``final`` results for the same pair: the one computed with the current combiner
   version wins; otherwise the higher ``seq`` wins.
5. ``flagged`` is maintained incrementally from verdict changes, never by counting events.
6. After ``job.completed``/``job.failed`` only ``log`` events are applied.
7. At most :data:`MAX_LOGS` log lines are kept.

The functions never mutate their input; they return shallow copies.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from typing import Any

MAX_LOGS = 2000
FLAGGED = "likely_plagiarism"

State = dict[str, Any]
Event = dict[str, Any]


def initial_state(job_id: str) -> State:
    """Empty state for a job that has not emitted any event yet."""
    return {
        "jobId": job_id,
        "status": "queued",
        "lastSeq": 0,
        "gap": False,
        "files": [],
        "combiner": None,
        "modelVersion": None,
        "prefilter": None,
        "totalPairs": 0,
        "candidatePairs": 0,
        "done": 0,
        "total": 0,
        "pairs": {},
        "flagged": 0,
        "elapsed": None,
        "error": None,
        "logs": [],
    }


def _is_flagged(pair: dict[str, Any] | None) -> int:
    return int(pair is not None and pair.get("verdict") == FLAGGED)


def _new_wins(prev: dict[str, Any], event: Event, current: str | None) -> bool:
    prev_ok = prev.get("combinerVersion") == current
    next_ok = event.get("combinerVersion") == current
    if prev_ok != next_ok:
        return next_ok
    return int(event["seq"]) > int(prev["updatedSeq"])


def _pair_from_partial(e: Event) -> dict[str, Any]:
    return {
        "pairKey": e["pairKey"],
        "fileA": e["fileA"],
        "fileB": e["fileB"],
        "stage": "ast",
        "astFeatures": e["astFeatures"],
        "astScore": e["astScore"],
        "neuralScore": None,
        "score": None,
        "threshold": None,
        "verdict": None,
        "detector": None,
        "combinerVersion": None,
        "matches": [],
        "warnings": [],
        "updatedSeq": e["seq"],
    }


def _pair_from_scored(e: Event) -> dict[str, Any]:
    return {
        "pairKey": e["pairKey"],
        "fileA": e["fileA"],
        "fileB": e["fileB"],
        "stage": "final",
        "astFeatures": e["astFeatures"],
        "astScore": e["astScore"],
        "neuralScore": e.get("neuralScore"),
        "score": e["score"],
        "threshold": e["threshold"],
        "verdict": e["verdict"],
        "detector": e["detector"],
        "combinerVersion": e["combinerVersion"],
        "matches": e.get("matches", []),
        "warnings": e.get("warnings", []),
        "updatedSeq": e["seq"],
    }


def combine_results(state: State, event: Event) -> State:
    """Apply one event. Pure and idempotent; returns ``state`` itself when nothing changes.

    Complexity: O(1) plus the O(P) copy of ``pairs`` when a pair changes (P = pairs).
    """
    seq = int(event["seq"])
    if seq <= state["lastSeq"]:
        return state
    s = copy(state)
    s["lastSeq"] = seq
    s["gap"] = bool(state["gap"] or seq > state["lastSeq"] + 1)
    kind = event["type"]
    if state["status"] in ("completed", "failed") and kind != "log":
        return s

    if kind == "job.started":
        s.update(
            status="running",
            files=event["files"],
            combiner=event["combiner"],
            modelVersion=event.get("modelVersion"),
            prefilter=event["prefilter"],
            totalPairs=event["totalPairs"],
            candidatePairs=event["candidatePairs"],
            total=event["candidatePairs"],
        )
    elif kind == "pair.partial":
        prev = state["pairs"].get(event["pairKey"])
        if prev is not None and prev["stage"] == "final":
            return s
        s["pairs"] = {**state["pairs"], event["pairKey"]: _pair_from_partial(event)}
    elif kind == "pair.scored":
        prev = state["pairs"].get(event["pairKey"])
        current = (state["combiner"] or {}).get("version")
        if prev is not None and prev["stage"] == "final" and not _new_wins(prev, event, current):
            return s
        pair = _pair_from_scored(event)
        s["pairs"] = {**state["pairs"], event["pairKey"]: pair}
        s["flagged"] = state["flagged"] - _is_flagged(prev) + _is_flagged(pair)
    elif kind == "progress":
        s.update(done=event["done"], total=event["total"])
    elif kind == "log":
        line = {
            "seq": seq,
            "level": event.get("level", "info"),
            "message": event.get("message", ""),
            "ts": event.get("ts", ""),
        }
        s["logs"] = [*state["logs"], line][-MAX_LOGS:]
    elif kind == "job.completed":
        s.update(status="completed", elapsed=event["seconds"])
    elif kind == "job.failed":
        s.update(status="failed", error=event["error"])
    return s


def combine_many(state: State, events: Iterable[Event]) -> State:
    """Apply a batch of events in ``seq`` order (tolerates reordering inside the batch)."""
    for event in sorted(events, key=lambda e: int(e["seq"])):
        state = combine_results(state, event)
    return state
