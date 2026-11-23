"""Reproducible plots generated from the result files in ``docs/results``.

Run ``codeplag plots`` (or ``python -m codeplag.evaluation.plots``) to regenerate every
figure from the JSON/CSV results without recomputing anything.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"ast_only": "#1f77b4", "neural_only": "#ff7f0e", "combined": "#2ca02c"}
LABELS = {"ast_only": "AST only", "neural_only": "Neural only", "combined": "Combined"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_curves(results: Path, out: Path) -> list[Path]:
    """ROC and precision-recall curves on test (and unseen Java)."""
    curves = _read_json(results / "curves.json")
    metrics = _read_json(results / "metrics.json")
    if not curves:
        return []
    written = []
    for set_name, by_comb in curves.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for cname, pts in by_comb.items():
            auc = ""
            if set_name == "test" and metrics:
                auc = f" (AUC {metrics['combiners'][cname]['test_at_0.5']['roc_auc']:.3f})"
            ax1.plot(
                pts["fpr"],
                pts["tpr"],
                color=COLORS.get(cname),
                label=LABELS.get(cname, cname) + auc,
            )
            ax2.plot(
                pts["recall"],
                pts["precision"],
                color=COLORS.get(cname),
                label=LABELS.get(cname, cname),
            )
        ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax1.set(
            xlabel="False positive rate", ylabel="True positive rate", title=f"ROC ({set_name})"
        )
        ax2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall ({set_name})")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower left")
        written.append(_save(fig, out / f"roc_pr_{set_name}.png"))
    return written


def plot_thresholds(results: Path, out: Path) -> list[Path]:
    """Precision, recall, F1 and FPR against the decision threshold."""
    rows = _read_csv(results / "threshold_analysis.csv")
    metrics = _read_json(results / "metrics.json")
    if not rows or not metrics:
        return []
    combiners = list(dict.fromkeys(r["combiner"] for r in rows))
    fig, axes = plt.subplots(1, len(combiners), figsize=(5 * len(combiners), 4), sharey=True)
    axes = axes if len(combiners) > 1 else [axes]
    for ax, cname in zip(axes, combiners, strict=True):
        sub = [r for r in rows if r["combiner"] == cname]
        t = [float(r["threshold"]) for r in sub]
        for key, style in (("precision", "-"), ("recall", "-"), ("f1", "-"), ("fpr", "--")):
            ax.plot(
                t, [float(r[key]) for r in sub], style, label=key.upper() if key == "fpr" else key
            )
        chosen = metrics["combiners"][cname]["threshold_fpr"]
        ax.axvline(chosen, color="red", lw=1, label=f"val threshold {chosen:.2f}")
        ax.axhline(metrics.get("max_fpr", 0.05), color="gray", lw=0.6, ls=":")
        ax.set(title=LABELS.get(cname, cname), xlabel="threshold")
    axes[0].set_ylabel("value on test")
    axes[-1].legend(loc="center right", fontsize=8)
    return [_save(fig, out / "threshold_analysis.png")]


def plot_confusion(results: Path, out: Path) -> list[Path]:
    """Confusion matrices on test at the validation-chosen thresholds."""
    metrics = _read_json(results / "metrics.json")
    if not metrics:
        return []
    combiners = list(metrics["combiners"])
    fig, axes = plt.subplots(1, len(combiners), figsize=(4 * len(combiners), 3.6))
    axes = axes if len(combiners) > 1 else [axes]
    for ax, cname in zip(axes, combiners, strict=True):
        m = metrics["combiners"][cname]["test_at_fpr_threshold"]
        grid = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        ax.imshow(grid, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(grid[i][j])}", ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{LABELS.get(cname, cname)} (t={m['threshold']:.2f})")
    return [_save(fig, out / "confusion_matrices.png")]


def plot_per_transform(results: Path, out: Path) -> list[Path]:
    """Recall per plagiarism transformation and per pair kind."""
    rows = _read_csv(results / "per_transform.csv")
    kinds = _read_csv(results / "per_kind.csv")
    if not rows:
        return []
    combiners = list(dict.fromkeys(r["combiner"] for r in rows))
    transforms = list(dict.fromkeys(r["transform"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    width = 0.8 / len(combiners)
    for k, cname in enumerate(combiners):
        vals = [
            float(
                next(r for r in rows if r["combiner"] == cname and r["transform"] == t)[
                    "recall_any"
                ]
            )
            for t in transforms
        ]
        ax1.bar(
            [i + k * width for i in range(len(transforms))],
            vals,
            width,
            color=COLORS.get(cname),
            label=LABELS.get(cname, cname),
        )
    ax1.set_xticks(
        [i + width * (len(combiners) - 1) / 2 for i in range(len(transforms))], transforms
    )
    ax1.set(
        ylim=(0, 1.05),
        ylabel="recall (synthetic test positives)",
        title="Recall per transformation",
    )
    ax1.legend(fontsize=8, loc="lower right")
    kind_names = list(dict.fromkeys(r["kind"] for r in kinds))
    for k, cname in enumerate(combiners):
        vals = []
        for kind in kind_names:
            r = next(x for x in kinds if x["combiner"] == cname and x["kind"] == kind)
            vals.append(float(r["recall"]) if r.get("recall") else float(r["fpr"]))
        ax2.bar(
            [i + k * width for i in range(len(kind_names))], vals, width, color=COLORS.get(cname)
        )
    ax2.set_xticks(
        [i + width * (len(combiners) - 1) / 2 for i in range(len(kind_names))],
        [f"{k}\n({'FPR' if k == 'different_problem' else 'recall'})" for k in kind_names],
    )
    ax2.set(ylim=(0, 1.05), title="Per pair kind (test)")
    return [_save(fig, out / "per_transform.png")]


def plot_robustness(results: Path, out: Path) -> list[Path]:
    """Recall on the hard positive suites, FPR on the boilerplate suite, F1 on unseen Java."""
    rob = _read_json(results / "robustness.json")
    if not rob:
        return []
    suites = [
        ("hard_obfuscation", "recall"),
        ("hard_equivalent", "recall"),
        ("hard_boilerplate", "fpr"),
        ("unseen_java", "f1"),
        ("unseen_java", "roc_auc"),
    ]
    suites = [(s, k) for s, k in suites if s in rob]
    combiners = list(next(iter(rob.values())))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(combiners)
    for k, cname in enumerate(combiners):
        vals = [rob[s][cname]["at_threshold"][key] for s, key in suites]
        ax.bar(
            [i + k * width for i in range(len(suites))],
            vals,
            width,
            color=COLORS.get(cname),
            label=LABELS.get(cname, cname),
        )
    ax.set_xticks(
        [i + width * (len(combiners) - 1) / 2 for i in range(len(suites))],
        [f"{s}\n{k}" for s, k in suites],
    )
    ax.set(ylim=(0, 1.05), title="Robustness (thresholds fixed on validation)")
    ax.legend(fontsize=8)
    return [_save(fig, out / "robustness.png")]


def plot_training(results: Path, out: Path) -> list[Path]:
    """Loss and ROC-AUC per epoch for train subset, validation and test."""
    hist = _read_json(results / "training_history.json")
    if not hist or not hist.get("history"):
        return []
    epochs = [h["epoch"] for h in hist["history"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for split, color in (("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")):
        if split not in hist["history"][0]:
            continue
        ax1.plot(
            epochs, [h[split]["loss"] for h in hist["history"]], "o-", color=color, label=split
        )
        ax2.plot(
            epochs, [h[split]["roc_auc"] for h in hist["history"]], "o-", color=color, label=split
        )
    ax1.set(xlabel="epoch", ylabel="BCE loss", title="Loss per epoch")
    ax2.set(xlabel="epoch", ylabel="ROC-AUC", title="ROC-AUC per epoch")
    ax2.axvline(hist["best_epoch"], color="gray", ls=":", label="selected (val)")
    ax1.legend()
    ax2.legend()
    return [_save(fig, out / "training_curves.png")]


def plot_scan(results: Path, out: Path) -> list[Path]:
    """Scan time and compared pairs against corpus size."""
    eff = _read_json(results / "efficiency.json")
    if not eff or not eff.get("scan"):
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for prefilter in dict.fromkeys(s["prefilter"] for s in eff["scan"]):
        rows = [s for s in eff["scan"] if s["prefilter"] == prefilter]
        n = [s["files"] for s in rows]
        ax1.plot(n, [s["seconds"] for s in rows], "o-", label=prefilter)
        ax2.plot(n, [s["candidate_pairs"] for s in rows], "o-", label=prefilter)
    ax1.set(xlabel="files", ylabel="seconds", title="Scan wall-clock time")
    ax2.set(xlabel="files", ylabel="pairs compared", title="Pairs compared", yscale="log")
    ax1.legend()
    ax2.legend()
    return [_save(fig, out / "scan_throughput.png")]


def plot_score_distributions(results: Path, out: Path) -> list[Path]:
    """Histogram of the combined score per pair kind on test."""
    rows = _read_csv(results / "scores_test.csv")
    if not rows:
        return []
    key = "score_combined" if "score_combined" in rows[0] else "score_ast_only"
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in sorted({r["kind"] for r in rows}):
        ax.hist([float(r[key]) for r in rows if r["kind"] == kind], bins=40, alpha=0.55, label=kind)
    ax.set(
        xlabel=key.replace("score_", "") + " score",
        ylabel="pairs",
        title="Score distribution (test)",
    )
    ax.legend()
    return [_save(fig, out / "score_distributions.png")]


def make_all_plots(results_dir: Path) -> list[Path]:
    """Regenerate every plot under ``results_dir / 'plots'``.

    Returns:
        Paths of the written figures.
    """
    out = results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fn in (
        plot_curves,
        plot_thresholds,
        plot_confusion,
        plot_per_transform,
        plot_robustness,
        plot_training,
        plot_scan,
        plot_score_distributions,
    ):
        written.extend(fn(results_dir, out))
    return written


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/results")
    for p in make_all_plots(target):
        print(p)
