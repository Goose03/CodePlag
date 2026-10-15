"""Tests for the siamese model, dataset, training loop and inference (mocked encoder)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from codeplag.core.config import NeuralConfig, Settings
from codeplag.neural.dataset import PairCollator, PairDataset
from codeplag.neural.infer import NeuralScorer
from codeplag.neural.model import (
    SiameseCodeModel,
    build_model,
    freeze_lower_layers,
    mean_pool,
    resolve_device,
)
from codeplag.neural.train import predict, set_seed, train_model
from tests.conftest import MockEncoder, MockTokenizer, make_mock_model, read_fixture


def _records(n: int = 24) -> list[dict[str, Any]]:
    a = read_fixture("python/original.py")
    b = read_fixture("python/renamed.py")
    c = read_fixture("python/different.py")
    out = []
    for i in range(n):
        positive = i % 2 == 0
        out.append(
            {"id": f"x-{i}", "code_a": a, "code_b": b if positive else c, "label": int(positive)}
        )
    return out


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


@pytest.mark.parametrize("head", ["mlp", "cosine"])
def test_model_is_symmetric_and_shaped(head: str, mock_tokenizer: MockTokenizer) -> None:
    model = make_mock_model(head=head).eval()
    collate = PairCollator(mock_tokenizer, 64)
    ids_a, mask_a = collate.encode(["def f(): pass", "x = 1"])
    ids_b, mask_b = collate.encode(["def g(): pass", "while True: y = 2"])
    with torch.no_grad():
        ab = model(ids_a, mask_a, ids_b, mask_b)
        ba = model(ids_b, mask_b, ids_a, mask_a)
    assert ab.shape == (2,)
    assert torch.allclose(ab, ba, atol=1e-6)


def test_unknown_head_rejected() -> None:
    with pytest.raises(ValueError):
        SiameseCodeModel(MockEncoder(), 32, 16, head="nope")


def test_freeze_lower_layers() -> None:
    encoder = MockEncoder(layers=3)
    frozen = freeze_lower_layers(encoder, 2)
    assert frozen > 0
    assert not any(p.requires_grad for p in encoder.embeddings.parameters())
    layers = encoder.encoder.layer  # type: ignore[union-attr]
    assert not any(p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
    assert freeze_lower_layers(MockEncoder(), 0) == 0


def test_build_model_requires_hidden_size() -> None:
    with pytest.raises(ValueError):
        build_model(NeuralConfig(), encoder=MockEncoder())


def test_resolve_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda").type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_dataset_and_collator(tmp_path: Path, mock_tokenizer: MockTokenizer) -> None:
    path = _write(tmp_path / "pairs.jsonl", _records(6))
    ds = PairDataset.from_jsonl(path, limit=4)
    assert len(ds) == 4
    assert ds.labels == [1, 0, 1, 0]
    batch = PairCollator(mock_tokenizer, max_length=16)([ds[0], ds[1]])
    assert batch["ids_a"].shape[1] <= 16
    assert batch["labels"].tolist() == [1.0, 0.0]


def test_train_model_end_to_end(settings: Settings, mock_tokenizer: MockTokenizer) -> None:
    proc = settings.paths.processed_dir
    _write(proc / "train.jsonl", _records(24))
    _write(proc / "val.jsonl", _records(8))
    _write(proc / "test.jsonl", _records(8))
    settings.train.epochs = 3
    settings.train.head_learning_rate = 5e-3
    settings.train.patience = 5
    result = train_model(settings, model=make_mock_model(), tokenizer=mock_tokenizer)
    assert result.checkpoint is not None and result.checkpoint.is_file()
    assert len(result.history) == 3
    assert {"train", "val", "test"} <= set(result.history[0])
    assert result.best_val["roc_auc"] >= 0.5
    saved = json.loads((settings.paths.results_dir / "training_history.json").read_text())
    assert saved["best_epoch"] == result.best_epoch
    payload = torch.load(result.checkpoint, weights_only=False)
    assert "state_dict" in payload and payload["neural_config"]["projection_dim"] == 16


def test_train_model_requires_data(settings: Settings, mock_tokenizer: MockTokenizer) -> None:
    with pytest.raises(FileNotFoundError):
        train_model(settings, model=make_mock_model(), tokenizer=mock_tokenizer)


def test_train_model_requires_tokenizer_with_model(settings: Settings) -> None:
    proc = settings.paths.processed_dir
    _write(proc / "train.jsonl", _records(4))
    _write(proc / "val.jsonl", _records(4))
    with pytest.raises(ValueError):
        train_model(settings, model=make_mock_model())


def test_predict_returns_probabilities(mock_tokenizer: MockTokenizer) -> None:
    from torch.utils.data import DataLoader

    set_seed(0)
    loader = DataLoader(
        PairDataset(_records(6)), batch_size=4, collate_fn=PairCollator(mock_tokenizer, 32)
    )
    y, p, loss = predict(make_mock_model(), loader, torch.device("cpu"))
    assert y.tolist() == [1, 0, 1, 0, 1, 0]
    assert ((p >= 0) & (p <= 1)).all() and loss > 0


def test_scorer_caches_embeddings(tmp_path: Path, mock_tokenizer: MockTokenizer) -> None:
    config = NeuralConfig(projection_dim=16, max_length=32, embed_batch_size=2)
    scorer = NeuralScorer(make_mock_model(), mock_tokenizer, config, cache_dir=tmp_path / "emb")
    calls = 0
    original = scorer.model.embed

    def counting(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += ids.shape[0]
        return original(ids, mask)

    scorer.model.embed = counting  # type: ignore[method-assign]
    s1 = scorer.score("a = 1", "b = 2")
    s2 = scorer.score("b = 2", "a = 1")
    assert calls == 2  # each unique file embedded once
    assert s1 == pytest.approx(s2)
    assert 0.0 <= s1 <= 1.0
    assert len(list((tmp_path / "emb").glob("*.npy"))) == 2
    fresh = NeuralScorer(make_mock_model(), mock_tokenizer, config, cache_dir=tmp_path / "emb")
    assert torch.allclose(fresh.embed(["a = 1"]), scorer.embed(["a = 1"]))
    assert scorer.score_pairs([]).shape == (0,)
    assert scorer.embed([]).shape == (0, 16)


def test_from_checkpoint_missing_file(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        NeuralScorer.from_checkpoint(settings.paths.checkpoints_dir / "nope.pt", settings)
