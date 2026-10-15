"""Tests for metrics, the combiner, reports, prefilters and the detector."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from codeplag.core.config import CombinerConfig, Settings
from codeplag.core.metrics import binary_metrics, threshold_for_f1, threshold_for_fpr
from codeplag.neural.infer import NeuralScorer
from codeplag.similarity.combiner import (
    Combiner,
    fit_all,
    fit_combiner,
    load_combiners,
    save_combiners,
    weighted_combiner,
)
from codeplag.similarity.detector import PlagiarismDetector
from codeplag.similarity.prefilter import (
    MinHasher,
    embedding_candidates,
    estimated_jaccard,
    lsh_candidates,
    shingles,
)
from codeplag.similarity.report import VERDICT_CLEAR, VERDICT_FLAGGED
from tests.conftest import FIXTURES, MockTokenizer, make_mock_model

# --------------------------------------------------------------------------- metrics


def test_binary_metrics_counts() -> None:
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5
    assert m["roc_auc"] == 0.75
    single = binary_metrics([1, 1], [0.2, 0.8])
    assert np.isnan(single["roc_auc"])


def test_threshold_selection() -> None:
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.6, 50), np.linspace(0.4, 1.0, 50)])
    t = threshold_for_fpr(y, s, max_fpr=0.1)
    assert binary_metrics(y, s, t)["fpr"] <= 0.1
    assert threshold_for_fpr(y, s, max_fpr=0.0) >= 0.6
    t_f1 = threshold_for_f1(y, s)
    assert 0.4 <= t_f1 <= 0.6


# --------------------------------------------------------------------------- combiner


def _rows(n: int = 200, seed: int = 0) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(seed)
    labels = [int(i % 2) for i in range(n)]
    rows = [
        {
            "ted": float(np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.1), 0, 1)),
            "subtree": float(np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15), 0, 1)),
            "ngram": float(np.clip(0.5 + 0.2 * y + rng.normal(0, 0.1), 0, 1)),
            "neural": float(np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15), 0, 1)),
        }
        for y in labels
    ]
    return rows, labels


def test_fit_all_and_roundtrip(tmp_path: Path) -> None:
    rows, labels = _rows()
    combiners = fit_all(rows, labels, max_fpr=0.05)
    assert set(combiners) == {"ast_only", "neural_only", "combined"}
    comb = combiners["combined"]
    assert comb.fit_metrics["fpr"] <= 0.05
    assert comb.fit_metrics["roc_auc"] > 0.9
    save_combiners(combiners, tmp_path / "c.json")
    loaded = load_combiners(tmp_path / "c.json")
    assert loaded["combined"].score(rows[0]) == pytest.approx(comb.score(rows[0]))
    assert loaded["combined"].predict(rows[1]) == comb.predict(rows[1])


def test_fit_all_without_neural_feature() -> None:
    rows, labels = _rows()
    for r in rows:
        r.pop("neural")
    assert set(fit_all(rows, labels, 0.05)) == {"ast_only"}


def test_combiner_scores_are_bounded() -> None:
    rows, labels = _rows()
    comb = fit_combiner("x", rows, labels, ["ted", "neural"], 0.05)
    scores = comb.score_many(rows)
    assert ((scores >= 0) & (scores <= 1)).all()
    with pytest.raises(KeyError):
        comb.score({"ted": 0.5})


def test_weighted_combiner() -> None:
    comb = weighted_combiner(CombinerConfig(), {"ted": 0.5, "subtree": 0.25, "ngram": 0.25})
    assert comb.score({"ted": 1, "subtree": 1, "ngram": 1, "neural": 1}) == pytest.approx(1.0)
    assert comb.score({"ted": 0, "subtree": 0, "ngram": 0, "neural": 1}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- prefilter


def test_minhash_estimates_similarity() -> None:
    base = [f"t{i}" for i in range(200)]
    near = base[:180] + [f"x{i}" for i in range(20)]
    far = [f"y{i}" for i in range(200)]
    h = MinHasher(num_perm=128, seed=1)
    sa, sn, sf = (h.signature(shingles(t, 3)) for t in (base, near, far))
    assert estimated_jaccard(sa, sn) > 0.5
    assert estimated_jaccard(sa, sf) < 0.1
    pairs = lsh_candidates([sa, sn, sf, h.signature(set())], bands=32)
    assert (0, 1) in pairs and (0, 2) not in pairs
    with pytest.raises(ValueError):
        lsh_candidates([sa], bands=7)
    assert lsh_candidates([], bands=4) == set()
    assert shingles(["a", "b"], 5) and not shingles([], 5)


def test_embedding_candidates() -> None:
    emb = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]])
    pairs = embedding_candidates(emb, top_k=1)
    assert pairs == {(0, 1), (2, 3)}
    assert embedding_candidates(emb[:1], top_k=3) == set()


# --------------------------------------------------------------------------- detector + report


def test_detector_ast_only_compare(settings: Settings) -> None:
    detector = PlagiarismDetector(settings)
    report, a, b = detector.compare_files(
        FIXTURES / "python" / "original.py", FIXTURES / "python" / "renamed.py"
    )
    assert report.neural_score is None
    assert report.detector == "ast_weighted"
    assert report.verdict == VERDICT_FLAGGED
    assert any("Neural model unavailable" in w for w in report.warnings)
    text = report.to_text(a.code, b.code)
    assert "LIKELY PLAGIARISM" in text and "lines" in text
    data = json.loads(report.to_json())
    assert data["verdict"] == VERDICT_FLAGGED and "disclaimer" in data
    clear, _, _ = detector.compare_files(
        FIXTURES / "python" / "original.py", FIXTURES / "python" / "different.py"
    )
    assert clear.verdict == VERDICT_CLEAR
    assert "unlikely plagiarism" in clear.to_text()


def test_detector_empty_file_scores_zero(settings: Settings) -> None:
    detector = PlagiarismDetector(settings, use_neural=False)
    report, _, _ = detector.compare_files(
        FIXTURES / "python" / "empty.py", FIXTURES / "python" / "empty.py"
    )
    assert report.score == 0.0 and report.verdict == VERDICT_CLEAR


def test_detector_with_neural_and_fitted_combiner(settings: Settings) -> None:
    scorer = NeuralScorer(make_mock_model(), MockTokenizer(), settings.neural)
    rows, labels = _rows()
    detector = PlagiarismDetector(settings, scorer=scorer, combiners=fit_all(rows, labels, 0.05))
    report, _, _ = detector.compare_files(
        FIXTURES / "python" / "original.py", FIXTURES / "python" / "renamed.py"
    )
    assert report.detector == "combined"
    assert report.neural_score is not None and 0 <= report.score <= 1


def test_detector_loads_combiners_from_disk(settings: Settings) -> None:
    combiner = Combiner("ast_only", "logistic", ["ted", "subtree", "ngram"], [5, 5, 5], -7.5, 0.4)
    save_combiners({"ast_only": combiner}, settings.paths.combiner_path)
    detector = PlagiarismDetector(settings)
    assert detector.combiner(with_neural=False).threshold == 0.4
    assert detector.combiner(with_neural=True).name == "weighted"


def test_scan_with_and_without_prefilter(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("original.py", "renamed.py", "reordered.py", "different.py"):
        shutil.copy(FIXTURES / "python" / name, corpus / name)
    (corpus / "notes.txt").write_text("ignored")
    detector = PlagiarismDetector(settings, use_neural=False)
    result = detector.scan(corpus)
    assert result.files == 4 and result.total_pairs == 6 and result.candidate_pairs == 6
    assert result.prefilter == "none"
    top = result.reports[0]
    assert "different" not in top.file_a + top.file_b
    assert len(result.flagged) == 3
    settings.scan.min_candidates_for_prefilter = 2
    pre = PlagiarismDetector(settings, use_neural=False).scan(corpus)
    assert pre.prefilter == "minhash"
    assert pre.candidate_pairs <= 6
    flagged_pairs = {frozenset((Path(r.file_a).name, Path(r.file_b).name)) for r in pre.flagged}
    assert frozenset(("original.py", "reordered.py")) in flagged_pairs
    with pytest.raises(NotADirectoryError):
        detector.scan(corpus / "original.py")


def test_scan_embedding_prefilter(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    for name in ("original.py", "renamed.py", "different.py"):
        shutil.copy(FIXTURES / "python" / name, corpus / name)
    settings.scan.prefilter = "embedding"
    settings.scan.min_candidates_for_prefilter = 2
    settings.scan.top_k = 1
    scorer = NeuralScorer(make_mock_model(), MockTokenizer(), settings.neural)
    result = PlagiarismDetector(settings, scorer=scorer).scan(corpus)
    assert result.prefilter == "minhash+embedding"
    assert result.candidate_pairs >= 1


def test_scan_parallel_matches_serial(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "par"
    corpus.mkdir()
    for name in ("original.py", "renamed.py", "reordered.py", "different.py"):
        shutil.copy(FIXTURES / "python" / name, corpus / name)
    serial = PlagiarismDetector(settings, use_neural=False).scan(corpus)
    settings.scan.workers = 2
    settings.scan.parallel_min_pairs = 1
    parallel = PlagiarismDetector(settings, use_neural=False).scan(corpus)
    assert [round(r.score, 6) for r in parallel.reports] == [
        round(r.score, 6) for r in serial.reports
    ]
