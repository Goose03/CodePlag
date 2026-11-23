"""Human-readable and JSON reports for a pair comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from codeplag.ast_analysis.tree_distance import MatchRegion

VERDICT_FLAGGED = "likely_plagiarism"
VERDICT_CLEAR = "unlikely_plagiarism"

DISCLAIMER = (
    "A high score is evidence for human review, not proof of plagiarism. Boilerplate, "
    "templates and standard algorithms can legitimately look alike."
)


@dataclass
class ComparisonReport:
    """Result of comparing two files.

    Attributes:
        file_a: Path or label of file A.
        file_b: Path or label of file B.
        language_a: Language of A.
        language_b: Language of B.
        score: Final score in ``[0, 1]``.
        threshold: Decision threshold applied to ``score``.
        verdict: :data:`VERDICT_FLAGGED` or :data:`VERDICT_CLEAR`.
        detector: Combiner used (``combined``, ``ast_only``, ``weighted``...).
        ast_score: Weighted AST score.
        ast_features: ``ted``, ``subtree`` and ``ngram`` similarities.
        neural_score: Neural similarity, or ``None`` when no model is available.
        matches: Largest shared code regions.
        warnings: Caveats collected during the comparison.
    """

    file_a: str
    file_b: str
    language_a: str
    language_b: str
    score: float
    threshold: float
    verdict: str
    detector: str
    ast_score: float
    ast_features: dict[str, float]
    neural_score: float | None
    matches: list[MatchRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        data = asdict(self)
        data["matches"] = [
            {
                "lines_a": list(m.lines_a),
                "lines_b": list(m.lines_b),
                "size": m.size,
                "node": m.label,
            }
            for m in self.matches
        ]
        data["disclaimer"] = DISCLAIMER
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report as JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(
        self, code_a: str | None = None, code_b: str | None = None, context: int = 4
    ) -> str:
        """Render a plain-text report.

        Args:
            code_a: Source of A, used to print snippets of matching regions.
            code_b: Source of B.
            context: Maximum lines of each snippet.

        Returns:
            The report text.
        """
        flagged = self.verdict == VERDICT_FLAGGED
        lines = [
            f"codeplag report: {self.file_a}  vs  {self.file_b}",
            "=" * 72,
            f"Verdict      : {'LIKELY PLAGIARISM' if flagged else 'unlikely plagiarism'}"
            f"  (score {self.score:.3f}, threshold {self.threshold:.3f}, detector {self.detector})",
            f"AST score    : {self.ast_score:.3f}  "
            + "  ".join(f"{k}={v:.3f}" for k, v in self.ast_features.items()),
            "Neural score : "
            + (f"{self.neural_score:.3f}" if self.neural_score is not None else "n/a (no model)"),
        ]
        if self.matches:
            lines.append("")
            lines.append("Largest matching regions (normalized structure):")
            src_a = code_a.splitlines() if code_a else None
            src_b = code_b.splitlines() if code_b else None
            for i, m in enumerate(self.matches, start=1):
                lines.append(
                    f"  {i}. A lines {m.lines_a[0]}-{m.lines_a[1]}  <->  "
                    f"B lines {m.lines_b[0]}-{m.lines_b[1]}  ({m.label}, {m.size} nodes)"
                )
                if src_a and src_b:
                    for tag, src, (s, e) in (("A", src_a, m.lines_a), ("B", src_b, m.lines_b)):
                        snippet = src[s - 1 : min(e, s - 1 + context)]
                        for offset, text in enumerate(snippet):
                            lines.append(f"       {tag}{s + offset:>5} | {text}")
        if self.warnings:
            lines.append("")
            lines.extend(f"Warning: {w}" for w in self.warnings)
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)
