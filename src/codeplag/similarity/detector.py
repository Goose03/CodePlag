"""High-level detector: compare two files or scan a directory.

:class:`PlagiarismDetector` wires the AST analyzer, the neural scorer (when a checkpoint
exists) and the fitted combiner. Without a trained model it falls back to the ``ast_only``
combiner (or an unfitted weighted AST score) and says so in the report warnings.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from codeplag.ast_analysis.normalizer import NormalizedSource
from codeplag.ast_analysis.parser import EXTENSIONS, detect_language
from codeplag.ast_analysis.tree_distance import (
    AstComparison,
    compare_normalized,
    compare_sources,
    normalize_source,
)
from codeplag.core.config import AstConfig, Settings
from codeplag.core.logging import get_logger
from codeplag.neural.infer import NeuralScorer
from codeplag.similarity.combiner import (
    AST_FEATURES,
    Combiner,
    load_combiners,
    weighted_combiner,
)
from codeplag.similarity.prefilter import (
    MinHasher,
    embedding_candidates,
    lsh_candidates,
    shingles,
)
from codeplag.similarity.report import VERDICT_CLEAR, VERDICT_FLAGGED, ComparisonReport

logger = get_logger(__name__)


@dataclass
class SourceFile:
    """A loaded and normalized source file."""

    label: str
    language: str
    code: str
    normalized: NormalizedSource


@dataclass
class ScanResult:
    """Result of scanning a directory.

    Attributes:
        files: Number of supported files found.
        total_pairs: ``n * (n - 1) / 2``.
        candidate_pairs: Pairs that survived the prefilter and were fully compared.
        prefilter: Prefilter used.
        reports: Reports for compared pairs, highest score first.
        seconds: Wall-clock time.
    """

    files: int
    total_pairs: int
    candidate_pairs: int
    prefilter: str
    reports: list[ComparisonReport] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def flagged(self) -> list[ComparisonReport]:
        """Reports whose verdict is likely plagiarism."""
        return [r for r in self.reports if r.verdict == VERDICT_FLAGGED]


def _ast_worker(job: tuple[str, str, str, str, dict[str, Any]]) -> AstComparison:
    code_a, lang_a, code_b, lang_b, cfg = job
    return compare_sources(code_a, code_b, lang_a, lang_b, AstConfig.model_validate(cfg))


def _labels(source: SourceFile) -> list[str]:
    """Pre-order labels of the normalized tree (rename- and order-of-definition invariant)."""
    return [node.label for node in source.normalized.tree.preorder()]


class PlagiarismDetector:
    """Compares source files with the AST analyzer, the neural model and the combiner.

    Args:
        settings: Loaded settings.
        scorer: Neural scorer to use (tests inject one); loaded lazily from the
            checkpoint otherwise.
        combiners: Fitted combiners; loaded from ``paths.combiner_path`` when omitted.
        use_neural: Disable the neural model entirely when false.
    """

    def __init__(
        self,
        settings: Settings,
        scorer: NeuralScorer | None = None,
        combiners: dict[str, Combiner] | None = None,
        use_neural: bool = True,
    ) -> None:
        self.settings = settings
        self._scorer = scorer
        self._scorer_loaded = scorer is not None or not use_neural
        self.use_neural = use_neural
        if combiners is None:
            path = settings.paths.combiner_path
            combiners = load_combiners(path) if path.is_file() else {}
        self.combiners = combiners

    @property
    def scorer(self) -> NeuralScorer | None:
        """The neural scorer, or ``None`` if no checkpoint is available."""
        if not self._scorer_loaded:
            self._scorer_loaded = True
            ckpt = self.settings.paths.checkpoints_dir / self.settings.neural.checkpoint_name
            if ckpt.is_file():
                try:
                    self._scorer = NeuralScorer.from_checkpoint(ckpt, self.settings)
                except Exception as exc:  # noqa: BLE001 - degrade to AST-only, but tell the user
                    logger.warning("Could not load neural model (%s); using AST only", exc)
            else:
                logger.info("No checkpoint at %s; using AST only", ckpt)
        return self._scorer if self.use_neural else None

    def combiner(self, with_neural: bool) -> Combiner:
        """Pick the combiner for the available detectors."""
        name = "combined" if with_neural else "ast_only"
        if name in self.combiners:
            return self.combiners[name]
        if with_neural:
            return weighted_combiner(self.settings.combiner, self.settings.ast.weights)
        weights = self.settings.ast.weights
        return Combiner(
            name="ast_weighted",
            method="weighted",
            features=list(AST_FEATURES),
            coef=[weights.get(f, 0.0) for f in AST_FEATURES],
            threshold=self.settings.combiner.threshold,
        )

    def load(self, path: str | Path, language: str | None = None) -> SourceFile:
        """Read and normalize a file."""
        file_path = Path(path)
        lang = language or detect_language(file_path)
        code = file_path.read_text(encoding="utf-8", errors="replace")
        return self.from_code(code, lang, str(file_path))

    def from_code(self, code: str, language: str, label: str = "<code>") -> SourceFile:
        """Normalize source code given as a string."""
        return SourceFile(
            label, language, code, normalize_source(code, language, self.settings.ast)
        )

    def compare(
        self,
        a: SourceFile,
        b: SourceFile,
        neural_score: float | None = None,
        ast_cmp: AstComparison | None = None,
    ) -> ComparisonReport:
        """Compare two loaded files.

        Args:
            a: File A.
            b: File B.
            neural_score: Precomputed neural score (used by :meth:`scan`).
            ast_cmp: Precomputed structural comparison (used by :meth:`scan`).

        Returns:
            The comparison report.
        """
        if ast_cmp is None:
            ast_cmp = compare_normalized(a.normalized, b.normalized, self.settings.ast)
        warnings = list(ast_cmp.warnings)
        scorer = self.scorer
        if neural_score is None and scorer is not None:
            neural_score = scorer.score(a.code, b.code)
        if scorer is None:
            warnings.append("Neural model unavailable; score uses AST features only.")
        features = dict(ast_cmp.features)
        if neural_score is not None:
            features["neural"] = neural_score
        combiner = self.combiner(neural_score is not None)
        empty = not a.normalized.tree.children or not b.normalized.tree.children
        score = 0.0 if empty else combiner.score(features)
        return ComparisonReport(
            file_a=a.label,
            file_b=b.label,
            language_a=a.language,
            language_b=b.language,
            score=score,
            threshold=combiner.threshold,
            verdict=VERDICT_FLAGGED if score >= combiner.threshold else VERDICT_CLEAR,
            detector=combiner.name,
            ast_score=ast_cmp.score,
            ast_features=ast_cmp.features,
            neural_score=neural_score,
            matches=ast_cmp.matches,
            warnings=warnings,
        )

    def compare_files(
        self, path_a: str | Path, path_b: str | Path
    ) -> tuple[ComparisonReport, SourceFile, SourceFile]:
        """Load and compare two files; also returns the loaded files for rendering."""
        a, b = self.load(path_a), self.load(path_b)
        return self.compare(a, b), a, b

    def candidate_pairs(self, files: list[SourceFile]) -> tuple[set[tuple[int, int]], str]:
        """Prefilter candidate index pairs for a scan."""
        cfg = self.settings.scan
        n = len(files)
        if cfg.prefilter == "none" or n < cfg.min_candidates_for_prefilter:
            return set(combinations(range(n), 2)), "none"
        hasher = MinHasher(cfg.num_perm, self.settings.seed)
        sigs = [hasher.signature(shingles(_labels(f))) for f in files]
        candidates = lsh_candidates(sigs, cfg.bands)
        method = "minhash"
        scorer = self.scorer
        if cfg.prefilter == "embedding" and scorer is not None:
            emb = scorer.embed([f.code for f in files]).numpy()
            candidates |= embedding_candidates(emb, cfg.top_k)
            method = "minhash+embedding"
        return candidates, method

    def _ast_parallel(
        self, files: list[SourceFile], pairs: list[tuple[int, int]]
    ) -> dict[tuple[int, int], AstComparison]:
        """Structural comparisons in worker processes (empty dict when run serially)."""
        workers = self.settings.scan.workers or max(1, (os.cpu_count() or 2) - 1)
        if workers == 1 or len(pairs) < self.settings.scan.parallel_min_pairs:
            return {}
        cfg = self.settings.ast.model_dump()
        jobs = [
            (files[i].code, files[i].language, files[j].code, files[j].language, cfg)
            for i, j in pairs
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_ast_worker, jobs, chunksize=8))
        return dict(zip(pairs, results, strict=True))

    def scan(self, directory: str | Path) -> ScanResult:
        """Compare files under ``directory`` pairwise (after prefiltering).

        Args:
            directory: Directory searched recursively for supported extensions.

        Returns:
            The scan result with reports sorted by score.

        Raises:
            NotADirectoryError: If ``directory`` is not a directory.
        """
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")
        start = time.perf_counter()
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        files = [self.load(p) for p in paths]
        candidates, method = self.candidate_pairs(files)
        pairs = sorted(candidates)
        neural: dict[tuple[int, int], float] = {}
        scorer = self.scorer
        if scorer is not None and pairs:
            scores = scorer.score_pairs([(files[i].code, files[j].code) for i, j in pairs])
            neural = {pair: float(s) for pair, s in zip(pairs, scores, strict=True)}
        ast_results = self._ast_parallel(files, pairs)
        reports = [
            self.compare(files[i], files[j], neural.get((i, j)), ast_results.get((i, j)))
            for i, j in pairs
        ]
        reports.sort(key=lambda r: -r.score)
        n = len(files)
        return ScanResult(
            files=n,
            total_pairs=n * (n - 1) // 2,
            candidate_pairs=len(pairs),
            prefilter=method,
            reports=reports,
            seconds=time.perf_counter() - start,
        )
