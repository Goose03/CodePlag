"""End-to-end evaluation: fit combiners on validation, measure everything on held-out data.

Outputs (in ``paths.results_dir``):

* ``metrics.json``: test metrics for ``ast_only``, ``neural_only`` and ``combined`` at the
  validation-chosen thresholds (FPR-constrained and F1-optimal) and at 0.5.
* ``threshold_analysis.csv``: precision/recall/F1/FPR on test for a sweep of thresholds.
* ``per_kind.csv`` / ``per_transform.csv``: recall per positive kind and per transformation,
  FPR on negatives.
* ``robustness.json``: hard suites and the unseen-language (Java) set.
* ``curves.json``: ROC and PR curve points for plotting.
* ``scores_<set>.csv``: per-pair features and scores (no source code).
* ``efficiency.json``: latency, memory and scan throughput (see :mod:`.efficiency`).

The test split is never used to fit anything: combiners and thresholds come from validation.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from codeplag.core.config import Settings
from codeplag.core.logging import get_logger
from codeplag.core.metrics import binary_metrics, threshold_for_f1
from codeplag.data.pairs import read_jsonl
from codeplag.evaluation.features import ast_features, neural_scores
from codeplag.neural.infer import NeuralScorer
from codeplag.similarity.combiner import Combiner, fit_all, save_combiners

logger = get_logger(__name__)

EVAL_SETS: tuple[str, ...] = (
    "val",
    "test",
    "hard_obfuscation",
    "hard_equivalent",
    "hard_boilerplate",
    "unseen_java",
)


def _load_sets(processed: Path) -> dict[str, list[dict[str, Any]]]:
    sets: dict[str, list[dict[str, Any]]] = {}
    for name in EVAL_SETS:
        path = processed / f"{name}.jsonl"
        if path.is_file():
            sets[name] = [dict(r) for r in read_jsonl(path)]
        elif name in ("val", "test"):
            raise FileNotFoundError(f"{path} not found; run `codeplag prepare-data` first.")
    return sets


def _rows(
    records: Sequence[Mapping[str, Any]],
    ast: Mapping[str, Mapping[str, float]],
    neural: Mapping[str, float] | None,
) -> list[dict[str, float]]:
    rows = []
    for r in records:
        f = ast[str(r["id"])]
        row = {"ted": f["ted"], "subtree": f["subtree"], "ngram": f["ngram"]}
        if neural is not None:
            row["neural"] = neural[str(r["id"])]
        rows.append(row)
    return rows


def _curve_points(
    labels: np.ndarray, scores: np.ndarray, max_points: int = 200
) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)

    def thin(xs: np.ndarray) -> list[float]:
        idx = np.linspace(0, len(xs) - 1, min(len(xs), max_points)).astype(int)
        return [round(float(xs[i]), 5) for i in idx]

    return {"fpr": thin(fpr), "tpr": thin(tpr), "precision": thin(prec), "recall": thin(rec)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _subset_metrics(
    labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float
) -> dict[str, float]:
    y, s = labels[mask], scores[mask]
    pred = s >= threshold
    out: dict[str, float] = {"n": float(mask.sum())}
    if (y == 1).any():
        out["recall"] = float(pred[y == 1].mean())
    if (y == 0).any():
        out["fpr"] = float(pred[y == 0].mean())
    out["mean_score"] = float(s.mean()) if len(s) else float("nan")
    return out


def run_evaluation(
    settings: Settings,
    *,
    use_neural: bool = True,
    workers: int = 0,
    efficiency: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation and write result files.

    Args:
        settings: Loaded settings.
        use_neural: Include the neural model (requires a checkpoint).
        workers: Processes for AST feature extraction (0 = all cores but one).
        efficiency: Also run latency/throughput benchmarks.

    Returns:
        A compact summary (also written to ``summary.json``).
    """
    paths = settings.paths
    results = paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    sets = _load_sets(paths.processed_dir)

    scorer: NeuralScorer | None = None
    if use_neural:
        ckpt = paths.checkpoints_dir / settings.neural.checkpoint_name
        scorer = NeuralScorer.from_checkpoint(ckpt, settings)

    rows: dict[str, list[dict[str, float]]] = {}
    labels: dict[str, np.ndarray] = {}
    ast_raw: dict[str, dict[str, dict[str, float]]] = {}
    for name, records in sets.items():
        logger.info("Features for %s (%d pairs)", name, len(records))
        ast_raw[name] = ast_features(
            records,
            settings.ast,
            workers=workers,
            cache_path=paths.cache_dir / f"ast_{name}.jsonl",
        )
        nn = neural_scores(records, scorer) if scorer is not None else None
        rows[name] = _rows(records, ast_raw[name], nn)
        labels[name] = np.array([int(r["label"]) for r in records])

    combiners = fit_all(rows["val"], list(labels["val"]), settings.combiner.max_fpr, settings.seed)
    f1_thresholds = {
        name: threshold_for_f1(labels["val"], c.score_many(rows["val"]))
        for name, c in combiners.items()
    }
    save_combiners(combiners, paths.combiner_path)
    logger.info("Saved combiners to %s", paths.combiner_path)

    scores: dict[str, dict[str, np.ndarray]] = {
        name: {c: comb.score_many(rows[name]) for c, comb in combiners.items()} for name in sets
    }

    # ---- headline metrics on test
    metrics: dict[str, Any] = {"combiners": {}, "val_fit": {}}
    y_test = labels["test"]
    for cname, comb in combiners.items():
        s = scores["test"][cname]
        metrics["combiners"][cname] = {
            "features": comb.features,
            "coef": comb.coef,
            "intercept": comb.intercept,
            "threshold_fpr": comb.threshold,
            "threshold_f1": f1_thresholds[cname],
            "test_at_fpr_threshold": binary_metrics(y_test, s, comb.threshold),
            "test_at_f1_threshold": binary_metrics(y_test, s, f1_thresholds[cname]),
            "test_at_0.5": binary_metrics(y_test, s, 0.5),
        }
        metrics["val_fit"][cname] = comb.fit_metrics
    metrics["max_fpr"] = settings.combiner.max_fpr
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- threshold sweep on test
    sweep = []
    for t in np.round(np.linspace(0.0, 1.0, 101), 2):
        for cname in combiners:
            m = binary_metrics(y_test, scores["test"][cname], float(t))
            sweep.append(
                {
                    "combiner": cname,
                    "threshold": float(t),
                    **{k: round(m[k], 5) for k in ("precision", "recall", "f1", "fpr", "accuracy")},
                }
            )
    _write_csv(results / "threshold_analysis.csv", sweep)

    # ---- per kind / per transformation on test
    test_records = sets["test"]
    kinds = np.array([str(r["kind"]) for r in test_records])
    per_kind, per_transform = [], []
    for cname, comb in combiners.items():
        s = scores["test"][cname]
        for kind in sorted({str(k) for k in kinds}):
            per_kind.append(
                {
                    "combiner": cname,
                    "kind": kind,
                    **_subset_metrics(y_test, s, kinds == kind, comb.threshold),
                }
            )
        transforms = sorted({t for r in test_records for t in r.get("transforms", [])})
        for t in transforms:
            mask = np.array([t in r.get("transforms", []) for r in test_records])
            only = np.array([r.get("transforms", []) == [t] for r in test_records])
            per_transform.append(
                {
                    "combiner": cname,
                    "transform": t,
                    "recall_any": _subset_metrics(y_test, s, mask, comb.threshold).get("recall"),
                    "n_any": int(mask.sum()),
                    "recall_only": _subset_metrics(y_test, s, only, comb.threshold).get("recall")
                    if only.any()
                    else None,
                    "n_only": int(only.sum()),
                    "mean_score_any": float(s[mask].mean()),
                }
            )
    _write_csv(results / "per_kind.csv", per_kind)
    _write_csv(results / "per_transform.csv", per_transform)

    # ---- robustness: hard suites and unseen language
    robustness: dict[str, Any] = {}
    neg_mask = y_test == 0
    pos_mask = y_test == 1
    for name in ("hard_obfuscation", "hard_equivalent", "hard_boilerplate", "unseen_java"):
        if name not in sets:
            continue
        robustness[name] = {}
        for cname, comb in combiners.items():
            s, y = scores[name][cname], labels[name]
            entry: dict[str, Any] = {"at_threshold": binary_metrics(y, s, comb.threshold)}
            if name == "unseen_java":
                entry["at_threshold_by_kind"] = {
                    kind: _subset_metrics(
                        y, s, np.array([r["kind"] == kind for r in sets[name]]), comb.threshold
                    )
                    for kind in sorted({r["kind"] for r in sets[name]})
                }
            else:
                # Pair single-class suites with the opposite class from test to get an AUC.
                other = neg_mask if y.mean() > 0.5 else pos_mask
                y_mix = np.concatenate([y, y_test[other]])
                s_mix = np.concatenate([s, scores["test"][cname][other]])
                entry["vs_test_opposite_class"] = binary_metrics(y_mix, s_mix, comb.threshold)
            robustness[name][cname] = entry
    (results / "robustness.json").write_text(json.dumps(robustness, indent=2))

    # ---- curves and per-pair scores
    curves = {
        name: {c: _curve_points(labels[name], scores[name][c]) for c in combiners}
        for name in ("test", "unseen_java")
        if name in sets and len(np.unique(labels[name])) == 2
    }
    (results / "curves.json").write_text(json.dumps(curves))
    for name, records in sets.items():
        out_rows = []
        for i, r in enumerate(records):
            row: dict[str, Any] = {
                "id": r["id"],
                "label": r["label"],
                "kind": r["kind"],
                "transforms": "+".join(r.get("transforms", [])),
                **{k: round(v, 5) for k, v in rows[name][i].items()},
                "ast_seconds": round(ast_raw[name][str(r["id"])]["seconds"], 5),
            }
            for cname in combiners:
                row[f"score_{cname}"] = round(float(scores[name][cname][i]), 5)
            out_rows.append(row)
        _write_csv(results / f"scores_{name}.csv", out_rows)

    summary = _summary(combiners, metrics, robustness)
    if efficiency:
        from codeplag.evaluation.efficiency import run_efficiency

        summary["efficiency"] = run_efficiency(
            settings, sets["test"], use_neural=scorer is not None
        )
    (results / "summary.json").write_text(json.dumps(summary, indent=2))

    from codeplag.evaluation.plots import make_all_plots

    make_all_plots(results)
    return summary


def _summary(
    combiners: Mapping[str, Combiner], metrics: Mapping[str, Any], robustness: Mapping[str, Any]
) -> dict[str, Any]:
    keys = ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")
    out: dict[str, Any] = {"test": {}, "robustness": {}}
    for cname in combiners:
        m = metrics["combiners"][cname]["test_at_fpr_threshold"]
        out["test"][cname] = {k: round(m[k], 4) for k in keys}
        out["test"][cname]["threshold"] = round(metrics["combiners"][cname]["threshold_fpr"], 4)
    for suite, by_comb in robustness.items():
        out["robustness"][suite] = {
            c: {
                k: round(v, 4)
                for k, v in e["at_threshold"].items()
                if k in ("recall", "fpr", "roc_auc", "f1")
            }
            for c, e in by_comb.items()
        }
    return out
