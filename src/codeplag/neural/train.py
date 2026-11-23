"""Training loop for the siamese model.

Binary cross-entropy on pair labels, AdamW with a linear warmup/decay schedule, mixed
precision on CUDA, early stopping on validation ROC-AUC and per-epoch metrics for a fixed
subset of training pairs, the validation split and the test split. Test metrics are logged
only to draw overfitting curves; they are never used for model selection.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from codeplag.core.config import Settings
from codeplag.core.logging import get_logger
from codeplag.core.metrics import binary_metrics
from codeplag.neural.dataset import PairCollator, PairDataset, Tokenizer
from codeplag.neural.model import (
    SiameseCodeModel,
    build_model,
    checkpoint_payload,
    resolve_device,
)

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    """Summary of a training run.

    Attributes:
        best_epoch: Epoch (1-based) with the best validation ROC-AUC.
        best_val: Validation metrics at the best epoch.
        history: Per-epoch metrics.
        checkpoint: Path to the best checkpoint.
        seconds: Wall-clock training time.
    """

    best_epoch: int
    best_val: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: Path | None = None
    seconds: float = 0.0


@torch.no_grad()
def predict(
    model: SiameseCodeModel, loader: DataLoader[Any], device: torch.device, amp: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Score every pair in ``loader``.

    Returns:
        Labels, probabilities and mean BCE loss.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    labels, probs, total = [], [], 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(batch["ids_a"], batch["mask_a"], batch["ids_b"], batch["mask_b"])
        logits = logits.float()
        total += float(loss_fn(logits, batch["labels"]))
        labels.append(batch["labels"].cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(labels) if labels else np.zeros(0)
    p = np.concatenate(probs) if probs else np.zeros(0)
    return y.astype(int), p, total / max(len(y), 1)


def _metrics(
    model: SiameseCodeModel, loader: DataLoader[Any], device: torch.device, amp: bool
) -> dict[str, float]:
    y, p, loss = predict(model, loader, device, amp)
    out = binary_metrics(y, p, 0.5)
    out["loss"] = loss
    return out


def train_model(
    settings: Settings,
    *,
    model: SiameseCodeModel | None = None,
    tokenizer: Tokenizer | None = None,
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
) -> TrainResult:
    """Train the siamese model and save the best checkpoint.

    Args:
        settings: Loaded settings.
        model: Pre-built model (tests inject a mock-encoder model).
        tokenizer: Tokenizer matching the model (required when ``model`` is given).
        train_path: Override for ``<processed>/train.jsonl``.
        val_path: Override for ``<processed>/val.jsonl``.
        test_path: Override for ``<processed>/test.jsonl`` (monitoring only; optional).

    Returns:
        A :class:`TrainResult`.

    Raises:
        FileNotFoundError: If the training or validation pairs are missing.
    """
    cfg, ncfg, paths = settings.train, settings.neural, settings.paths
    set_seed(settings.seed)
    device = resolve_device(ncfg.device)
    amp = cfg.fp16 and device.type == "cuda"

    train_path = train_path or paths.processed_dir / "train.jsonl"
    val_path = val_path or paths.processed_dir / "val.jsonl"
    test_path = test_path if test_path is not None else paths.processed_dir / "test.jsonl"
    for p in (train_path, val_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found; run `codeplag prepare-data` first.")

    if model is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ncfg.model_name)
        model = build_model(ncfg)
    if tokenizer is None:
        raise ValueError("tokenizer is required when passing a model")
    model.to(device)

    train_ds = PairDataset.from_jsonl(train_path, limit=cfg.max_train_pairs)
    val_ds = PairDataset.from_jsonl(val_path)
    test_ds = PairDataset.from_jsonl(test_path) if test_path.is_file() else None
    rng = random.Random(settings.seed)
    train_eval_ds = PairDataset(
        rng.sample(train_ds.records, k=min(cfg.train_eval_pairs, len(train_ds)))
    )
    collate = PairCollator(tokenizer, ncfg.max_length)
    generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        generator=generator,
    )
    eval_bs = cfg.batch_size * 2

    def loader(ds: PairDataset) -> DataLoader[Any]:
        return DataLoader(ds, batch_size=eval_bs, shuffle=False, collate_fn=collate)

    eval_loaders = {"train": loader(train_eval_ds), "val": loader(val_ds)}
    if test_ds is not None:
        eval_loaders["test"] = loader(test_ds)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.learning_rate},
            {"params": head_params, "lr": cfg.head_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()

    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = paths.checkpoints_dir / ncfg.checkpoint_name
    logger.info(
        "Training on %s: %d pairs, %d val, %d steps, amp=%s",
        device,
        len(train_ds),
        len(val_ds),
        total_steps,
        amp,
    )

    history: list[dict[str, Any]] = []
    best_auc, best_epoch, best_val, bad_epochs = -1.0, 0, {}, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        epoch_start = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False, mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(batch["ids_a"], batch["mask_a"], batch["ids_b"], batch["mask_b"])
            loss = loss_fn(logits.float(), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach()) * len(batch["labels"])
            seen += len(batch["labels"])

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss_running": running / max(seen, 1),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        for name, eval_loader in eval_loaders.items():
            record[name] = _metrics(model, eval_loader, device, amp)
        history.append(record)
        val = record["val"]
        logger.info(
            "epoch %d: loss %.4f | train auc %.4f | val auc %.4f f1 %.4f loss %.4f",
            epoch,
            record["train_loss_running"],
            record["train"]["roc_auc"],
            val["roc_auc"],
            val["f1"],
            val["loss"],
        )

        if val["roc_auc"] > best_auc:
            best_auc, best_epoch, best_val, bad_epochs = val["roc_auc"], epoch, val, 0
            torch.save(
                checkpoint_payload(model, ncfg, epoch=epoch, val_metrics=val, seed=settings.seed),
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    elapsed = time.perf_counter() - start
    summary = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "seconds": elapsed,
        "device": str(device),
        "train_pairs": len(train_ds),
        "history": history,
    }
    (paths.checkpoints_dir / "history.json").write_text(json.dumps(summary, indent=2))
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "training_history.json").write_text(json.dumps(summary, indent=2))
    return TrainResult(best_epoch, best_val, history, ckpt_path, elapsed)
