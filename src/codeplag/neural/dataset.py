"""PyTorch dataset over the processed pair files.

Records are kept as raw strings; tokenization happens per batch in :class:`PairCollator`,
which lets a fast tokenizer process the whole batch at once and pad to the longest sequence
in the batch (bounded by ``max_length``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from codeplag.data.pairs import read_jsonl

Tokenizer = Callable[..., Mapping[str, Any]]


class PairDataset(Dataset[dict[str, Any]]):
    """Code pairs with labels.

    Args:
        records: Dicts with at least ``code_a``, ``code_b`` and ``label``.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: int | None = None) -> PairDataset:
        """Load pairs from a JSONL file, optionally keeping only the first ``limit``."""
        records: list[Mapping[str, Any]] = []
        for record in read_jsonl(Path(path)):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "code_a": str(record["code_a"]),
            "code_b": str(record["code_b"]),
            "label": float(record["label"]),
        }

    @property
    def labels(self) -> list[int]:
        """Labels of all records, in order."""
        return [int(r["label"]) for r in self.records]


class PairCollator:
    """Tokenizes and pads a batch of pairs.

    Args:
        tokenizer: A Hugging Face tokenizer (or compatible callable).
        max_length: Truncation length in tokens.
    """

    def __init__(self, tokenizer: Tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenize ``texts`` into padded ``input_ids`` and ``attention_mask`` tensors."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        ids_a, mask_a = self.encode([item["code_a"] for item in batch])
        ids_b, mask_b = self.encode([item["code_b"] for item in batch])
        return {
            "ids_a": ids_a,
            "mask_a": mask_a,
            "ids_b": ids_b,
            "mask_b": mask_b,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        }
