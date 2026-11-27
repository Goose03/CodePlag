"""Generate the golden fixtures shared by the Python and TypeScript web tests.

Writes into ``web/shared/``:

* ``combiner_golden.json``: random and edge-case feature vectors scored by every fitted
  combiner in ``<checkpoints>/combiner.json`` (parity of ``Combiner.score`` vs
  ``scoreCombiner`` in ``web/src/lib/combiner.ts``).
* ``events_golden.json``: the event log of a real job over ``data/samples`` (parity of the
  Python and TypeScript reducers). ``web/src/lib/report-reducer.test.ts`` then writes
  ``state_golden.json`` when run with ``UPDATE_GOLDEN=1``.

Usage::

    python scripts/gen_web_golden.py --checkpoints models/checkpoints [--no-neural]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import fakeredis  # noqa: E402

from api.schemas import combiner_version  # noqa: E402
from api.settings import WebSettings  # noqa: E402
from api.storage import save_job  # noqa: E402
from api.validation import ValidFile  # noqa: E402
from api.worker import execute_job  # noqa: E402
from codeplag.ast_analysis.parser import EXTENSIONS  # noqa: E402
from codeplag.core.config import load_settings  # noqa: E402
from codeplag.similarity.combiner import load_combiners  # noqa: E402
from codeplag.similarity.detector import PlagiarismDetector  # noqa: E402

SHARED = ROOT / "web" / "shared"


def combiner_cases(path: Path, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for combiner in load_combiners(path).values():
        params = asdict(combiner)
        params.pop("fit_metrics")
        params["version"] = combiner_version(combiner)
        vectors = [{f: rng.random() for f in combiner.features} for _ in range(50)]
        vectors += [{f: v for f in combiner.features} for v in (0.0, 0.5, 1.0)]
        for feats in vectors:
            cases.append({"combiner": params, "features": feats, "expected": combiner.score(feats)})
    return cases


async def sample_job(detector: PlagiarismDetector, tmp: Path) -> list[dict]:
    files = []
    for p in sorted((ROOT / "data" / "samples").iterdir()):
        if p.suffix in EXTENSIONS:
            raw = p.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            files.append(ValidFile(p.name, EXTENSIONS[p.suffix], raw.decode(), digest))
    job_id = hashlib.sha256("|".join(sorted(f.sha256 for f in files)).encode()).hexdigest()[:20]
    cfg = WebSettings(jobs_dir=str(tmp), neural_batch=4)
    save_job(cfg.jobs_dir, job_id, files)
    redis = fakeredis.FakeAsyncRedis()
    await execute_job(redis, detector, job_id, cfg)
    rows = await redis.xrange(f"job:{job_id}:events")
    events = [json.loads(fields[b"e"]) for _, fields in rows]
    for e in events:  # timestamps would make the fixture change on every run
        if e["type"] == "log":
            e["ts"] = "2026-01-01T00:00:00+00:00"
        if e["type"] == "job.completed":
            e["seconds"] = 1.0
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "checkpoints")
    ap.add_argument("--no-neural", action="store_true")
    args = ap.parse_args()
    settings = load_settings(ROOT / "configs" / "default.yaml")
    settings.paths.checkpoints_dir = args.checkpoints
    settings.paths.combiner_path = args.checkpoints / "combiner.json"
    SHARED.mkdir(parents=True, exist_ok=True)

    cases = combiner_cases(settings.paths.combiner_path)
    (SHARED / "combiner_golden.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"combiner_golden.json: {len(cases)} cases")

    detector = PlagiarismDetector(settings, use_neural=not args.no_neural)
    print(f"neural model: {detector.scorer is not None}")
    tmp = ROOT / "data" / "processed" / "golden_jobs"
    events = asyncio.run(sample_job(detector, tmp))
    (SHARED / "events_golden.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    scored = [e for e in events if e["type"] == "pair.scored"]
    flagged = [e for e in scored if e["verdict"] == "likely_plagiarism"]
    print(f"events_golden.json: {len(events)} events, {len(flagged)} flagged pairs")
    for e in (e for e in events if e["type"] == "pair.scored"):
        print(f"  {e['fileA']:>14} vs {e['fileB']:<14} score={e['score']:.4f} {e['verdict']}")


if __name__ == "__main__":
    main()
