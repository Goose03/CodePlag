"""Wire format shared with the frontend (mirror of ``web/src/lib/types.ts``).

Events are plain JSON-able dicts so they can be stored in Redis Streams as-is. The helpers
here turn detector objects (:class:`Combiner`, :class:`ComparisonReport`) into the
camelCase payloads the browser expects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from codeplag.similarity.combiner import Combiner
from codeplag.similarity.report import ComparisonReport

EVENT_TYPES = (
    "job.started",
    "pair.partial",
    "pair.scored",
    "progress",
    "log",
    "job.completed",
    "job.failed",
)
TERMINAL_EVENTS = ("job.completed", "job.failed")


def combiner_version(combiner: Combiner) -> str:
    """Short, stable hash of the combiner parameters (not its fit metrics)."""
    params = asdict(combiner)
    params.pop("fit_metrics", None)
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def combiner_params(combiner: Combiner) -> dict[str, Any]:
    """``CombinerParams`` payload: everything the browser needs to re-score a pair."""
    return {
        "name": combiner.name,
        "method": combiner.method,
        "features": list(combiner.features),
        "coef": [float(c) for c in combiner.coef],
        "intercept": float(combiner.intercept),
        "threshold": float(combiner.threshold),
        "version": combiner_version(combiner),
    }


def report_payload(report: ComparisonReport) -> dict[str, Any]:
    """camelCase fields of a :class:`ComparisonReport` for a ``pair.scored`` event."""
    return {
        "fileA": report.file_a,
        "fileB": report.file_b,
        "languageA": report.language_a,
        "languageB": report.language_b,
        "score": float(report.score),
        "threshold": float(report.threshold),
        "verdict": report.verdict,
        "detector": report.detector,
        "astScore": float(report.ast_score),
        "astFeatures": {k: float(v) for k, v in report.ast_features.items()},
        "neuralScore": None if report.neural_score is None else float(report.neural_score),
        "matches": [
            {
                "linesA": list(m.lines_a),
                "linesB": list(m.lines_b),
                "size": m.size,
                "node": m.label,
            }
            for m in report.matches
        ],
        "warnings": list(report.warnings),
    }
