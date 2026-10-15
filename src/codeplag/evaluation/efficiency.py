"""Latency, memory and scan-throughput benchmarks on the current machine.

* AST latency: parse + normalize + compare for single pairs.
* Neural latency: one uncached pair (two encoder passes + head), on CPU and on CUDA when
  available; batched throughput on the configured device.
* Memory: peak resident set size of the process and peak CUDA memory.
* Scan: ``scan`` over corpora of ``N`` files built from test pairs (each synthetic test pair
  contributes the original and its plagiarized copy), with and without the MinHash
  prefilter. Prefilter recall is measured on the known plagiarized links.
"""

from __future__ import annotations

import os
import platform
import random
import shutil
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil
import torch

from codeplag.ast_analysis.tree_distance import compare_sources
from codeplag.core.config import Settings
from codeplag.core.logging import get_logger
from codeplag.neural.infer import NeuralScorer
from codeplag.similarity.detector import PlagiarismDetector

logger = get_logger(__name__)

SCAN_SIZES: tuple[int, ...] = (25, 50, 100, 200, 400)
EXHAUSTIVE_LIMIT = 200


def _stats_ms(samples: Sequence[float]) -> dict[str, float]:
    ms = sorted(s * 1000 for s in samples)
    return {
        "mean_ms": round(statistics.fmean(ms), 3),
        "p50_ms": round(ms[len(ms) // 2], 3),
        "p95_ms": round(ms[int(len(ms) * 0.95) - 1], 3),
        "n": len(ms),
    }


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def hardware_info() -> dict[str, Any]:
    """Describe the machine the benchmarks ran on."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _neural_latency(scorer: NeuralScorer, pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    times = []
    for a, b in pairs:
        scorer.cache.clear()
        start = time.perf_counter()
        scorer.score(a, b)
        if scorer.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return _stats_ms(times[1:] if len(times) > 1 else times)  # drop warm-up


def _scan_corpus(
    records: Sequence[Mapping[str, Any]], n_files: int, root: Path, seed: int
) -> set[tuple[str, str]]:
    synthetic = [r for r in records if r["kind"] == "synthetic"]
    rng = random.Random(seed)
    chosen = rng.sample(synthetic, k=min(len(synthetic), n_files // 2))
    links: set[tuple[str, str]] = set()
    for i, r in enumerate(chosen):
        a, b = root / f"f{i:04d}_a.py", root / f"f{i:04d}_b.py"
        a.write_text(str(r["code_a"]), encoding="utf-8")
        b.write_text(str(r["code_b"]), encoding="utf-8")
        links.add((str(a), str(b)))
    return links


def run_efficiency(
    settings: Settings, test_records: Sequence[Mapping[str, Any]], *, use_neural: bool = True
) -> dict[str, Any]:
    """Run all benchmarks and write ``efficiency.json``.

    Args:
        settings: Loaded settings.
        test_records: Test pairs used as benchmark inputs.
        use_neural: Include neural benchmarks (requires a checkpoint).

    Returns:
        The benchmark results.
    """
    import json

    rng = random.Random(settings.seed)
    sample = rng.sample(list(test_records), k=min(200, len(test_records)))
    out: dict[str, Any] = {"hardware": hardware_info()}

    rss_before = _rss_mb()
    ast_times = []
    for r in sample:
        start = time.perf_counter()
        compare_sources(
            str(r["code_a"]), str(r["code_b"]), str(r["lang_a"]), str(r["lang_b"]), settings.ast
        )
        ast_times.append(time.perf_counter() - start)
    out["ast_latency"] = _stats_ms(ast_times)

    ckpt = settings.paths.checkpoints_dir / settings.neural.checkpoint_name
    if use_neural and ckpt.is_file():
        pairs = [(str(r["code_a"]), str(r["code_b"])) for r in sample[:101]]
        out["neural_latency"] = {}
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            dev_settings = settings.model_copy(deep=True)
            dev_settings.neural.device = device
            scorer = NeuralScorer.from_checkpoint(ckpt, dev_settings)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            out["neural_latency"][device] = _neural_latency(scorer, pairs)
            scorer.cache.clear()
            all_pairs = [(str(r["code_a"]), str(r["code_b"])) for r in test_records[:1000]]
            start = time.perf_counter()
            scorer.score_pairs(all_pairs)
            if device == "cuda":
                torch.cuda.synchronize()
                out["neural_latency"][device]["peak_cuda_mb"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 1
                )
            elapsed = time.perf_counter() - start
            out["neural_latency"][device]["batched_pairs_per_s"] = round(
                len(all_pairs) / elapsed, 2
            )
            del scorer
    out["peak_rss_mb"] = round(max(_rss_mb(), rss_before), 1)

    scans = []
    detector = PlagiarismDetector(settings.model_copy(deep=True), use_neural=use_neural)
    for n in SCAN_SIZES:
        tmp = Path(tempfile.mkdtemp(prefix=f"codeplag_scan_{n}_"))
        try:
            links = _scan_corpus(test_records, n, tmp, settings.seed + n)
            for prefilter in ("none", "minhash"):
                if prefilter == "none" and n > EXHAUSTIVE_LIMIT:
                    continue
                detector.settings.scan.prefilter = prefilter  # type: ignore[assignment]
                if detector.scorer is not None:
                    detector.scorer.cache.clear()
                result = detector.scan(tmp)
                compared = {(r.file_a, r.file_b) for r in result.reports}
                compared |= {(b, a) for a, b in compared}
                flagged = {(r.file_a, r.file_b) for r in result.flagged}
                flagged |= {(b, a) for a, b in flagged}
                scans.append(
                    {
                        "files": result.files,
                        "prefilter": result.prefilter,
                        "total_pairs": result.total_pairs,
                        "candidate_pairs": result.candidate_pairs,
                        "seconds": round(result.seconds, 3),
                        "pairs_per_s": round(result.candidate_pairs / max(result.seconds, 1e-9), 2),
                        "files_per_s": round(result.files / max(result.seconds, 1e-9), 2),
                        "prefilter_recall_on_links": round(
                            len(links & compared) / max(len(links), 1), 4
                        ),
                        "detected_links": round(len(links & flagged) / max(len(links), 1), 4),
                        "flagged_pairs": len(result.flagged),
                    }
                )
                logger.info("scan %s", scans[-1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    out["scan"] = scans
    out["peak_rss_mb_after_scan"] = round(_rss_mb(), 1)
    settings.paths.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.paths.results_dir / "efficiency.json").write_text(json.dumps(out, indent=2))
    return out
