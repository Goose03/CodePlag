"""End-to-end pipeline on offline fixtures: pairs -> train -> features -> combiner -> report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeplag.core.config import Settings
from codeplag.data.codenet import CodeFile
from codeplag.data.pairs import PairBuilder, write_jsonl
from codeplag.evaluation import efficiency
from codeplag.evaluation.evaluate import run_evaluation
from codeplag.evaluation.features import ast_features, neural_scores
from codeplag.neural.infer import NeuralScorer
from codeplag.neural.train import train_model
from codeplag.similarity.combiner import fit_all, save_combiners
from codeplag.similarity.detector import PlagiarismDetector
from codeplag.similarity.report import VERDICT_FLAGGED
from tests.conftest import FIXTURES, MockTokenizer, make_mock_model

TEMPLATES = [
    "n = int(input())\nprint(sum(range(n + {k})))\n",
    "s = input()\nfor c in s:\n    if c == '{k}':\n        print(c)\n",
    "a = list(map(int, input().split()))\na.sort()\nprint(a[{k} % len(a)])\n",
    "def f(x):\n    return x * {k} + 1\nprint(f(int(input())))\n",
    "x, y = map(int, input().split())\nwhile x > {k}:\n    x -= y\nprint(x)\n",
    "d = {{}}\nfor i in range({k}):\n    d[i] = i * i\nprint(len(d))\n",
]


def _files() -> dict[str, list[CodeFile]]:
    return {
        f"p{p}": [CodeFile(f"p{p}", f"s{i}", "python", t.format(k=i + 2)) for i in range(5)]
        for p, t in enumerate(TEMPLATES)
    }


def _prepare(settings: Settings) -> None:
    proc = settings.paths.processed_dir
    files = _files()
    for offset, (split, n) in enumerate((("train", 40), ("val", 30), ("test", 30))):
        write_jsonl(
            PairBuilder(files, split, seed=offset).balanced(n, 0.5), proc / f"{split}.jsonl"
        )


def test_full_pipeline(settings: Settings) -> None:
    _prepare(settings)
    tokenizer = MockTokenizer()
    model = make_mock_model()
    result = train_model(settings, model=model, tokenizer=tokenizer)
    assert result.checkpoint is not None and result.checkpoint.is_file()

    scorer = NeuralScorer(model, tokenizer, settings.neural)
    val = [json.loads(line) for line in (settings.paths.processed_dir / "val.jsonl").open()]
    ast = ast_features(
        val, settings.ast, workers=1, cache_path=settings.paths.cache_dir / "v.jsonl"
    )
    nn = neural_scores(val, scorer)
    rows = [
        {**{k: ast[r["id"]][k] for k in ("ted", "subtree", "ngram")}, "neural": nn[r["id"]]}
        for r in val
    ]
    combiners = fit_all(rows, [r["label"] for r in val], settings.combiner.max_fpr)
    save_combiners(combiners, settings.paths.combiner_path)
    # Cached features are reused on the second call.
    assert (
        ast_features(val, settings.ast, workers=1, cache_path=settings.paths.cache_dir / "v.jsonl")
        == ast
    )

    detector = PlagiarismDetector(settings, scorer=scorer)
    report, a, b = detector.compare_files(
        FIXTURES / "python" / "original.py", FIXTURES / "python" / "original.py"
    )
    assert report.detector == "combined"
    assert report.neural_score is not None
    assert report.verdict == VERDICT_FLAGGED
    assert report.matches
    assert "matching regions" in report.to_text(a.code, b.code)


def test_evaluation_ast_only_writes_results(settings: Settings) -> None:
    _prepare(settings)
    suite = PairBuilder(_files(), "test", seed=2).collect("heavy_obfuscation", 4)
    write_jsonl(suite, settings.paths.processed_dir / "hard_obfuscation.jsonl")
    summary = run_evaluation(settings, use_neural=False, workers=1, efficiency=False)
    results: Path = settings.paths.results_dir
    assert set(summary["test"]) == {"ast_only"}
    for name in ("metrics.json", "threshold_analysis.csv", "per_kind.csv", "per_transform.csv",
                 "robustness.json", "curves.json", "scores_test.csv", "summary.json"):  # fmt: skip
        assert (results / name).is_file(), name
    assert (results / "plots" / "roc_pr_test.png").is_file()
    assert settings.paths.combiner_path.is_file()
    metrics = json.loads((results / "metrics.json").read_text())
    assert 0.0 <= metrics["combiners"]["ast_only"]["test_at_fpr_threshold"]["f1"] <= 1.0
    robustness = json.loads((results / "robustness.json").read_text())
    assert robustness["hard_obfuscation"]["ast_only"]["at_threshold"]["roc_auc"] is None
    assert "roc_auc" not in summary["robustness"]["hard_obfuscation"]["ast_only"]


def test_evaluation_uses_requested_workers_for_benchmark(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(settings)
    monkeypatch.setattr(efficiency, "SCAN_SIZES", (4,))
    summary = run_evaluation(settings, use_neural=False, workers=1)
    benchmark = summary["efficiency"]
    assert benchmark["scan_workers"] == 1
    assert benchmark["scan_device"] is None
    assert benchmark["scan"][0]["files"] == 4
    assert benchmark["rss_mib_after_scan"] > 0
    assert settings.scan.workers == 4


def test_ast_cache_invalidates_when_source_changes(settings: Settings) -> None:
    cache = settings.paths.cache_dir / "ast.jsonl"
    record = {
        "id": "test-000001",
        "lang_a": "python",
        "lang_b": "python",
        "code_a": "x = 1\n",
        "code_b": "x = 1\n",
    }
    original = ast_features([record], settings.ast, workers=1, cache_path=cache)
    updated = {**record, "code_b": "print('hello')\n"}
    changed = ast_features([updated], settings.ast, workers=1, cache_path=cache)
    assert changed["test-000001"]["ted"] < original["test-000001"]["ted"]
