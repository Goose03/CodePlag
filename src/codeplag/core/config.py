"""Configuration loading.

Settings come from three layers, later layers override earlier ones:

1. Defaults defined on the pydantic models below.
2. A YAML file: ``--config`` on the CLI, ``CODEPLAG_CONFIG``, or ``configs/default.yaml``
   relative to the working directory when it exists.
3. Environment variables: ``CODEPLAG_DEVICE``, ``CODEPLAG_MODEL_NAME``, ``CODEPLAG_LOG_LEVEL``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class PathsConfig(BaseModel):
    """Filesystem locations, relative to the working directory unless absolute."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    checkpoints_dir: Path = Path("models/checkpoints")
    combiner_path: Path = Path("models/checkpoints/combiner.json")
    results_dir: Path = Path("docs/results")
    cache_dir: Path = Path("data/processed/cache")


class AstConfig(BaseModel):
    """Semantic tree analyzer settings."""

    max_nodes: int = Field(
        default=250, description="Size guard: trees are pruned above this many nodes."
    )
    canonicalize_constructs: bool = Field(
        default=True, description="Map equivalent constructs (e.g. for/while) onto one label."
    )
    min_subtree_size: int = Field(
        default=4, description="Smallest subtree considered for hash overlap."
    )
    ngram_size: int = 3
    weights: dict[str, float] = Field(
        default_factory=lambda: {"ted": 0.4, "subtree": 0.3, "ngram": 0.3},
        description="Weights used to fold the three AST features into one AST score.",
    )
    top_matches: int = 5


class NeuralConfig(BaseModel):
    """Siamese model architecture and inference settings."""

    model_name: str = "microsoft/unixcoder-base"
    device: str = "cpu"
    max_length: int = 256
    projection_dim: int = 256
    head: Literal["mlp", "cosine"] = "mlp"
    freeze_layers: int = Field(
        default=6, description="Freeze embeddings plus the first N encoder layers."
    )
    checkpoint_name: str = "best.pt"
    embed_batch_size: int = 32


class TrainConfig(BaseModel):
    """Training loop settings."""

    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    patience: int = 2
    max_train_pairs: int | None = None
    train_eval_pairs: int = 2000
    fp16: bool = True
    num_workers: int = 0


class DataConfig(BaseModel):
    """Dataset preparation and pair generation settings."""

    primary_archive: str = "Project_CodeNet_Python800.tar.gz"
    unseen_archive: str = "Project_CodeNet_Java250.tar.gz"
    archive_base_url: str = (
        "https://huggingface.co/datasets/qiankunmu/Project_CodeNet_Python800_and_Java250"
        "/resolve/main"
    )
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    files_per_problem: int = 40
    max_file_bytes: int = 6000
    min_file_lines: int = 3
    pairs: dict[str, int] = Field(
        default_factory=lambda: {"train": 24000, "val": 4000, "test": 4000},
        description="Total pairs per split; half positive (same-problem/synthetic), half negative.",
    )
    synthetic_fraction: float = Field(
        default=0.5, description="Share of positives that are synthetic."
    )
    hard_pairs_per_suite: int = 500
    unseen_pairs: int = 2000
    unseen_problems: int = 100


class CombinerConfig(BaseModel):
    """Score fusion settings."""

    method: Literal["logistic", "weighted"] = "logistic"
    weights: dict[str, float] = Field(default_factory=lambda: {"ast": 0.5, "neural": 0.5})
    threshold: float = 0.5
    max_fpr: float = Field(
        default=0.05, description="FPR budget used when picking the threshold on val."
    )


class ScanConfig(BaseModel):
    """Directory scan settings."""

    prefilter: Literal["minhash", "embedding", "none"] = "minhash"
    num_perm: int = 64
    bands: int = 32
    top_k: int = 10
    min_candidates_for_prefilter: int = 20
    workers: int = Field(default=0, description="Processes for AST comparisons (0 = cores - 1).")
    parallel_min_pairs: int = 64


class Settings(BaseModel):
    """Root settings object."""

    seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ast: AstConfig = Field(default_factory=AstConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    combiner: CombinerConfig = Field(default_factory=CombinerConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)


def _apply_env(raw: dict[str, Any]) -> dict[str, Any]:
    env_map = {
        "CODEPLAG_DEVICE": ("neural", "device"),
        "CODEPLAG_MODEL_NAME": ("neural", "model_name"),
    }
    for var, (section, key) in env_map.items():
        value = os.environ.get(var)
        if value:
            raw.setdefault(section, {})[key] = value
    if os.environ.get("CODEPLAG_LOG_LEVEL"):
        raw["log_level"] = os.environ["CODEPLAG_LOG_LEVEL"]
    return raw


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from YAML plus environment overrides.

    Args:
        path: Explicit YAML file. Falls back to ``CODEPLAG_CONFIG`` and then
            ``configs/default.yaml`` when present.

    Returns:
        A validated :class:`Settings` instance.

    Raises:
        FileNotFoundError: If an explicitly requested file does not exist.
    """
    explicit = path or os.environ.get("CODEPLAG_CONFIG")
    raw: dict[str, Any] = {}
    if explicit:
        cfg_path = Path(explicit)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    elif DEFAULT_CONFIG_PATH.is_file():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_apply_env(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings loaded with :func:`load_settings` defaults."""
    return load_settings()
