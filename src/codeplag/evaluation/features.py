"""Feature extraction for evaluation: AST features (multi-process, cached) and neural scores."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from codeplag.ast_analysis.tree_distance import compare_sources
from codeplag.core.config import AstConfig
from codeplag.core.logging import get_logger
from codeplag.neural.infer import NeuralScorer

logger = get_logger(__name__)

Record = Mapping[str, Any]


def _ast_job(
    job: tuple[str, str, str, str, str, dict[str, Any]],
) -> tuple[str, dict[str, float], float]:
    pair_id, code_a, lang_a, code_b, lang_b, cfg = job
    start = time.perf_counter()
    result = compare_sources(code_a, code_b, lang_a, lang_b, AstConfig.model_validate(cfg))
    features = dict(result.features)
    features["ast_score"] = result.score
    features["approximate"] = float(result.approximate)
    return pair_id, features, time.perf_counter() - start


def ast_features(
    records: Sequence[Record],
    config: AstConfig,
    *,
    workers: int = 0,
    cache_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Compute AST features for every record, reusing a JSONL cache when present.

    Args:
        records: Pair records (``id``, ``code_a``, ``code_b``, ``lang_a``, ``lang_b``).
        config: AST settings (part of the cache key).
        workers: Worker processes (0 = ``os.cpu_count() - 1``; 1 = in-process).
        cache_path: JSONL cache file.

    Returns:
        Mapping pair id -> features (``ted``, ``subtree``, ``ngram``, ``ast_score``,
        ``approximate``, ``seconds``).
    """
    cfg = config.model_dump()
    signature = json.dumps(cfg, sort_keys=True)
    cached: dict[str, dict[str, float]] = {}
    if cache_path is not None and cache_path.is_file():
        with cache_path.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("config") == signature:
                for line in fh:
                    item = json.loads(line)
                    cached[item["id"]] = item["features"]
    todo = [r for r in records if str(r["id"]) not in cached]
    if todo:
        jobs = [
            (
                str(r["id"]),
                str(r["code_a"]),
                str(r["lang_a"]),
                str(r["code_b"]),
                str(r["lang_b"]),
                cfg,
            )
            for r in todo
        ]
        n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        results: list[tuple[str, dict[str, float], float]] = []
        if n_workers == 1:
            results = [_ast_job(j) for j in tqdm(jobs, desc="ast", mininterval=5)]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                iterator = pool.map(_ast_job, jobs, chunksize=16)
                results = list(tqdm(iterator, total=len(jobs), desc="ast", mininterval=5))
        for pair_id, features, seconds in results:
            features["seconds"] = seconds
            cached[pair_id] = features
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"config": signature}) + "\n")
                for pair_id, features in cached.items():
                    fh.write(json.dumps({"id": pair_id, "features": features}) + "\n")
    return {str(r["id"]): cached[str(r["id"])] for r in records}


def neural_scores(records: Sequence[Record], scorer: NeuralScorer) -> dict[str, float]:
    """Neural similarity for every record (embeddings computed once per unique file)."""
    pairs = [(str(r["code_a"]), str(r["code_b"])) for r in records]
    scores = scorer.score_pairs(pairs)
    return {str(r["id"]): float(s) for r, s in zip(records, scores, strict=True)}
