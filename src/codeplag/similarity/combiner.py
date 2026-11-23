"""Fusion of AST and neural scores into one calibrated similarity.

A :class:`Combiner` maps a feature dictionary to a score in ``[0, 1]``:

* ``logistic``: logistic regression over the selected features, fitted on the validation
  split. Its output is a calibrated probability of the positive class.
* ``weighted``: fixed weighted average (used before any combiner has been fitted).

The decision threshold is chosen on the validation split as the smallest threshold whose
false-positive rate stays within ``max_fpr`` (wrongly accusing someone is the costly error).
Three combiners are fitted and stored together: ``ast_only`` (``ted``, ``subtree``,
``ngram``), ``neural_only`` (``neural``) and ``combined`` (all four).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from codeplag.core.config import CombinerConfig
from codeplag.core.metrics import binary_metrics, threshold_for_fpr

AST_FEATURES: tuple[str, ...] = ("ted", "subtree", "ngram")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "ast_only": AST_FEATURES,
    "neural_only": ("neural",),
    "combined": (*AST_FEATURES, "neural"),
}


@dataclass
class Combiner:
    """Maps detector features to a final score.

    Attributes:
        name: Identifier (``ast_only``, ``neural_only``, ``combined`` or ``weighted``).
        method: ``logistic`` or ``weighted``.
        features: Feature names, in coefficient order.
        coef: Logistic coefficients, or weights for the weighted method.
        intercept: Logistic intercept (unused for ``weighted``).
        threshold: Decision threshold on the final score.
        fit_metrics: Validation metrics recorded when fitting.
    """

    name: str
    method: Literal["logistic", "weighted"]
    features: list[str]
    coef: list[float]
    intercept: float = 0.0
    threshold: float = 0.5
    fit_metrics: dict[str, float] = field(default_factory=dict)

    def score(self, features: Mapping[str, float]) -> float:
        """Final score for one feature dictionary.

        Raises:
            KeyError: If a required feature is missing.
        """
        x = [float(features[name]) for name in self.features]
        if self.method == "weighted":
            total = sum(self.coef) or 1.0
            return float(
                min(1.0, max(0.0, sum(w * v for w, v in zip(self.coef, x, strict=True)) / total))
            )
        z = self.intercept + sum(c * v for c, v in zip(self.coef, x, strict=True))
        return 1.0 / (1.0 + math.exp(-z))

    def score_many(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Scores for many feature dictionaries."""
        return np.array([self.score(r) for r in rows], dtype=float)

    def predict(self, features: Mapping[str, float]) -> bool:
        """Whether the pair is flagged as likely plagiarism."""
        return self.score(features) >= self.threshold


def weighted_combiner(config: CombinerConfig, ast_weights: Mapping[str, float]) -> Combiner:
    """Unfitted fallback: AST features weighted per ``ast_weights`` and blended with neural.

    The AST score receives ``config.weights['ast']`` and the neural score
    ``config.weights['neural']``.
    """
    w_ast = config.weights.get("ast", 0.5)
    w_nn = config.weights.get("neural", 0.5)
    total_ast = sum(ast_weights.get(f, 0.0) for f in AST_FEATURES) or 1.0
    coef = [w_ast * ast_weights.get(f, 0.0) / total_ast for f in AST_FEATURES] + [w_nn]
    return Combiner(
        name="weighted",
        method="weighted",
        features=[*AST_FEATURES, "neural"],
        coef=coef,
        threshold=config.threshold,
    )


def fit_combiner(
    name: str,
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    feature_names: Sequence[str],
    max_fpr: float,
    seed: int = 42,
) -> Combiner:
    """Fit a logistic combiner and its FPR-constrained threshold on validation data.

    Args:
        name: Combiner name.
        rows: Feature dictionaries (validation split).
        labels: Validation labels.
        feature_names: Features to use.
        max_fpr: False-positive-rate budget for the threshold.
        seed: Random state for the solver.

    Returns:
        The fitted combiner.
    """
    x = np.array([[float(r[f]) for f in feature_names] for r in rows], dtype=float)
    y = np.asarray(labels, dtype=int)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(x, y)
    combiner = Combiner(
        name=name,
        method="logistic",
        features=list(feature_names),
        coef=[float(c) for c in model.coef_[0]],
        intercept=float(model.intercept_[0]),
    )
    probs = combiner.score_many(rows)
    combiner.threshold = threshold_for_fpr(y, probs, max_fpr)
    combiner.fit_metrics = binary_metrics(y, probs, combiner.threshold)
    return combiner


def fit_all(
    rows: Sequence[Mapping[str, float]], labels: Sequence[int], max_fpr: float, seed: int = 42
) -> dict[str, Combiner]:
    """Fit the ``ast_only``, ``neural_only`` and ``combined`` combiners.

    Feature sets whose features are absent from ``rows`` are skipped.
    """
    available = set(rows[0]) if rows else set()
    return {
        name: fit_combiner(name, rows, labels, feats, max_fpr, seed)
        for name, feats in FEATURE_SETS.items()
        if set(feats) <= available
    }


def save_combiners(combiners: Mapping[str, Combiner], path: Path) -> None:
    """Persist combiners as JSON (no pickles)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(c) for name, c in combiners.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_combiners(path: Path) -> dict[str, Combiner]:
    """Load combiners saved with :func:`save_combiners`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    payload: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return {name: Combiner(**data) for name, data in payload.items()}
