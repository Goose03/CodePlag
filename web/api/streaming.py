"""Incremental scan: the same steps as :meth:`PlagiarismDetector.scan`, yielding events.

``scan()`` only returns when every pair is done. Here the pairs are processed in batches:
for each batch the cheap structural (AST) comparison is emitted first as a partial result,
then the neural model scores the whole batch and the final, combined result of each pair is
emitted. The final score always comes from :meth:`PlagiarismDetector.compare`, i.e. the same
code path as ``codeplag compare``, so the web and the CLI give identical numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

from codeplag.ast_analysis.tree_distance import AstComparison, compare_normalized
from codeplag.similarity.detector import PlagiarismDetector, SourceFile

from .schemas import combiner_params, combiner_version, report_payload
from .validation import ValidFile

Event = dict[str, Any]


def pair_key(sha_a: str, sha_b: str) -> str:
    """Order-independent key of a pair of files, based on their content hashes."""
    lo, hi = sorted((sha_a, sha_b))
    return hashlib.sha256(f"{lo}:{hi}".encode()).hexdigest()[:16]


def log_event(message: str, level: str = "info") -> Event:
    """A ``log`` event (``seq`` is assigned when it is published)."""
    return {
        "type": "log",
        "level": level,
        "message": message,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def iter_scan(
    det: PlagiarismDetector,
    files: Sequence[ValidFile],
    skip: frozenset[str] = frozenset(),
    neural_batch: int = 32,
) -> Iterator[Event]:
    """Compare ``files`` pairwise and yield events without ``seq``.

    Args:
        det: Detector (with the neural model already loaded, if any).
        files: Validated uploads.
        skip: Pair keys already finished in a previous attempt (resume after a retry).
        neural_batch: Pairs scored per neural batch before their results are emitted.

    Yields:
        ``job.started``, ``log``, ``pair.partial``, ``pair.scored`` and ``progress`` events.
    """
    sources: list[SourceFile] = [det.from_code(f.code, f.language, f.name) for f in files]
    candidates, method = det.candidate_pairs(sources)
    all_pairs = sorted(candidates)
    pairs = [(i, j) for i, j in all_pairs if pair_key(files[i].sha256, files[j].sha256) not in skip]
    scorer = det.scorer
    combiner = det.combiner(scorer is not None)
    version = combiner_version(combiner)
    n = len(files)

    yield {
        "type": "job.started",
        "files": [
            {"fileId": f.sha256, "name": f.name, "language": f.language, "sha256": f.sha256}
            for f in files
        ],
        "totalPairs": n * (n - 1) // 2,
        "candidatePairs": len(all_pairs),
        "prefilter": method,
        "combiner": combiner_params(combiner),
        "modelVersion": det.settings.neural.checkpoint_name if scorer is not None else None,
    }
    yield log_event(
        f"{n} files, {len(all_pairs)} candidate pairs of {n * (n - 1) // 2} (prefilter: {method})."
    )
    yield log_event(
        f"Combiner '{combiner.name}' (threshold {combiner.threshold:.4f}); "
        + ("neural model loaded." if scorer is not None else "neural model unavailable."),
        "info" if scorer is not None else "warning",
    )
    if skip:
        yield log_event(f"Resuming: {len(all_pairs) - len(pairs)} pairs already done.")

    done = len(all_pairs) - len(pairs)
    for start in range(0, len(pairs), max(1, neural_batch)):
        batch = pairs[start : start + neural_batch]
        # 1) Structural comparison: cheap, emitted right away as a partial result.
        ast: dict[tuple[int, int], AstComparison] = {}
        for i, j in batch:
            cmp = compare_normalized(sources[i].normalized, sources[j].normalized, det.settings.ast)
            ast[(i, j)] = cmp
            yield {
                "type": "pair.partial",
                "stage": "ast",
                "pairKey": pair_key(files[i].sha256, files[j].sha256),
                "fileA": files[i].name,
                "fileB": files[j].name,
                "astFeatures": {k: float(v) for k, v in cmp.features.items()},
                "astScore": float(cmp.score),
            }
        # 2) Neural model: one batch at a time (much faster than pair by pair).
        neural: dict[tuple[int, int], float] = {}
        if scorer is not None:
            scores = scorer.score_pairs([(sources[i].code, sources[j].code) for i, j in batch])
            neural = {p: float(s) for p, s in zip(batch, scores, strict=True)}
        # 3) Combiner: the exact code path of `codeplag compare`.
        for i, j in batch:
            report = det.compare(sources[i], sources[j], neural.get((i, j)), ast[(i, j)])
            yield {
                "type": "pair.scored",
                "stage": "final",
                "pairKey": pair_key(files[i].sha256, files[j].sha256),
                "combinerVersion": version,
                **report_payload(report),
            }
        done += len(batch)
        yield {"type": "progress", "done": done, "total": len(all_pairs)}
