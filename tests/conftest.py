"""Shared test fixtures: paths, isolated settings and a tiny mock encoder/tokenizer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from codeplag.core.config import NeuralConfig, Settings

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(rel: str) -> str:
    """Read a fixture file as text."""
    return (FIXTURES / rel).read_text(encoding="utf-8")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose every output path lives under ``tmp_path``."""
    s = Settings()
    s.paths.raw_dir = tmp_path / "raw"
    s.paths.processed_dir = tmp_path / "processed"
    s.paths.checkpoints_dir = tmp_path / "checkpoints"
    s.paths.combiner_path = tmp_path / "checkpoints" / "combiner.json"
    s.paths.results_dir = tmp_path / "results"
    s.paths.cache_dir = tmp_path / "cache"
    s.neural.device = "cpu"
    s.neural.projection_dim = 16
    s.neural.max_length = 64
    s.train.epochs = 1
    s.train.batch_size = 4
    s.train.fp16 = False
    s.train.train_eval_pairs = 8
    return s


class MockTokenizer:
    """Character-level tokenizer with the Hugging Face call signature."""

    vocab_size = 128

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        ids = [[1] + [ord(c) % 120 + 2 for c in t][: max_length - 1] for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class MockEncoder(nn.Module):
    """Tiny RoBERTa-shaped encoder: ``embeddings`` + ``encoder.layer``."""

    def __init__(self, hidden: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(MockTokenizer.vocab_size, hidden)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        h = self.embeddings(input_ids)
        for layer in self.encoder.layer:  # type: ignore[union-attr]
            h = torch.tanh(layer(h))
        return SimpleNamespace(last_hidden_state=h)


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


def make_mock_model(config: NeuralConfig | None = None, head: str = "mlp") -> Any:
    """A :class:`SiameseCodeModel` around :class:`MockEncoder`."""
    from codeplag.neural.model import build_model

    cfg = (config or NeuralConfig()).model_copy(
        update={"projection_dim": 16, "freeze_layers": 1, "head": head}
    )
    torch.manual_seed(0)
    return build_model(cfg, encoder=MockEncoder(), hidden_size=32)
