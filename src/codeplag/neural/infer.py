"""Inference with a trained siamese checkpoint.

:class:`NeuralScorer` embeds source files once and caches the embeddings by content hash
(in memory, and optionally on disk as ``.npy`` files), so scanning ``n`` files costs ``n``
encoder passes plus cheap head evaluations for the candidate pairs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from codeplag.core.config import NeuralConfig, Settings
from codeplag.core.logging import get_logger
from codeplag.neural.dataset import PairCollator, Tokenizer
from codeplag.neural.model import SiameseCodeModel, build_model, resolve_device

logger = get_logger(__name__)


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()


class NeuralScorer:
    """Scores code pairs with a trained :class:`SiameseCodeModel`.

    Args:
        model: Trained model.
        tokenizer: Matching tokenizer.
        config: Neural settings (``max_length``, ``embed_batch_size``).
        device: Torch device.
        cache_dir: Optional directory for persistent embedding cache files.
    """

    def __init__(
        self,
        model: SiameseCodeModel,
        tokenizer: Tokenizer,
        config: NeuralConfig,
        device: torch.device | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.collator = PairCollator(tokenizer, config.max_length)
        self.cache: dict[str, Tensor] = {}
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, settings: Settings, *, use_disk_cache: bool = False
    ) -> NeuralScorer:
        """Load a checkpoint written by :func:`codeplag.neural.train.train_model`.

        The encoder architecture is rebuilt from its Hugging Face config (no pretrained
        weights are loaded) and the checkpoint weights are applied on top.

        Args:
            path: Checkpoint file.
            settings: Settings; ``neural.device`` selects the device.
            use_disk_cache: Persist embeddings under ``paths.cache_dir``.

        Returns:
            A ready-to-use scorer.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        from transformers import AutoTokenizer

        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = NeuralConfig.model_validate(
            {**payload.get("neural_config", {}), "device": settings.neural.device}
        )
        model = build_model(config.model_copy(update={"freeze_layers": 0}), pretrained=False)
        model.load_state_dict(payload["state_dict"])
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        cache_dir = settings.paths.cache_dir / "embeddings" if use_disk_cache else None
        return cls(model, tokenizer, config, resolve_device(config.device), cache_dir)

    def _disk_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.npy" if self.cache_dir is not None else None

    @torch.no_grad()
    def embed(self, codes: Sequence[str]) -> Tensor:
        """Embed source files, reusing cached embeddings.

        Args:
            codes: Source texts.

        Returns:
            ``[len(codes), projection_dim]`` tensor on CPU.
        """
        keys = [_digest(c) for c in codes]
        missing: dict[str, str] = {}
        for key, code in zip(keys, codes, strict=True):
            if key in self.cache or key in missing:
                continue
            disk = self._disk_path(key)
            if disk is not None and disk.is_file():
                self.cache[key] = torch.from_numpy(np.load(disk))
            else:
                missing[key] = code
        items = list(missing.items())
        bs = self.config.embed_batch_size
        for start in range(0, len(items), bs):
            chunk = items[start : start + bs]
            ids, mask = self.collator.encode([code for _, code in chunk])
            emb = self.model.embed(ids.to(self.device), mask.to(self.device)).float().cpu()
            for (key, _), vector in zip(chunk, emb, strict=True):
                self.cache[key] = vector
                disk = self._disk_path(key)
                if disk is not None:
                    np.save(disk, vector.numpy())
        if not keys:
            return torch.zeros(0, self.config.projection_dim)
        return torch.stack([self.cache[k] for k in keys])

    @torch.no_grad()
    def score_embeddings(self, u: Tensor, v: Tensor) -> np.ndarray:
        """Similarity probabilities for row-aligned embedding matrices."""
        logits = self.model.pair_logits(u.to(self.device), v.to(self.device))
        return torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Score many ``(code_a, code_b)`` pairs, embedding each unique file once."""
        if not pairs:
            return np.zeros(0)
        unique = list(dict.fromkeys([c for pair in pairs for c in pair]))
        index = {code: i for i, code in enumerate(unique)}
        emb = self.embed(unique)
        out = []
        for start in range(0, len(pairs), 1024):
            chunk = pairs[start : start + 1024]
            u = emb[[index[a] for a, _ in chunk]]
            v = emb[[index[b] for _, b in chunk]]
            out.append(self.score_embeddings(u, v))
        return np.concatenate(out)

    def score(self, code_a: str, code_b: str) -> float:
        """Similarity probability for one pair."""
        return float(self.score_pairs([(code_a, code_b)])[0])
