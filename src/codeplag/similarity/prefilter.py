"""Cheap candidate generation for directory scans.

Comparing every pair of ``n`` files with tree edit distance costs ``O(n^2)`` expensive
comparisons. Two prefilters reduce the candidate set:

* **MinHash LSH** over shingles (k-grams) of the pre-order labels of the normalized tree
  (identifiers and literals already canonicalized, so renaming does not change them; the
  alpha-renamed token stream is not used because it depends on definition order).
  Signatures cost ``O(n * |shingles| * num_perm)``; banding with ``bands`` bands of
  ``rows = num_perm / bands`` rows makes a pair a candidate if any band collides. The
  collision probability for Jaccard ``s`` is ``1 - (1 - s^rows)^bands``; with 64
  permutations and 32 bands the 50% point is ``s ~= 0.15``.
* **Embedding nearest neighbors** (when a neural model is available): each file's ``top_k``
  most cosine-similar files.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from itertools import combinations

import numpy as np

_PRIME = (1 << 31) - 1


def shingles(tokens: Sequence[str], k: int = 5) -> set[int]:
    """Hash every k-gram of ``tokens`` with CRC32 (stable across processes)."""
    if len(tokens) < k:
        return {zlib.crc32(" ".join(tokens).encode())} if tokens else set()
    return {zlib.crc32(" ".join(tokens[i : i + k]).encode()) for i in range(len(tokens) - k + 1)}


class MinHasher:
    """Seeded MinHash signatures with universal hashing ``(a * x + b) mod p``.

    Args:
        num_perm: Signature length.
        seed: Random seed for the hash coefficients.
    """

    def __init__(self, num_perm: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _PRIME, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, _PRIME, size=num_perm, dtype=np.int64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        """MinHash signature of a shingle set (all ``_PRIME`` for an empty set)."""
        if not shingle_set:
            return np.full(self.num_perm, _PRIME, dtype=np.int64)
        x = np.fromiter(shingle_set, dtype=np.int64) % _PRIME
        hashed = (np.outer(x, self.a) + self.b) % _PRIME
        return np.asarray(hashed.min(axis=0), dtype=np.int64)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of equal signature positions (unbiased Jaccard estimate)."""
    return float(np.mean(sig_a == sig_b))


def lsh_candidates(signatures: Sequence[np.ndarray], bands: int) -> set[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j``, that collide in at least one LSH band.

    Raises:
        ValueError: If the signature length is not divisible by ``bands``.
    """
    if not signatures:
        return set()
    num_perm = len(signatures[0])
    if num_perm % bands:
        raise ValueError(f"num_perm={num_perm} is not divisible by bands={bands}")
    rows = num_perm // bands
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for idx, sig in enumerate(signatures):
            if int(sig[0]) == _PRIME:  # empty file
                continue
            key = sig[band * rows : (band + 1) * rows].tobytes()
            buckets.setdefault(key, []).append(idx)
        for members in buckets.values():
            if len(members) > 1:
                candidates.update(combinations(members, 2))
    return candidates


def embedding_candidates(embeddings: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    """Pairs linking each row to its ``top_k`` most cosine-similar rows."""
    n = len(embeddings)
    if n < 2:
        return set()
    norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9, None)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)
    k = min(top_k, n - 1)
    neighbors = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    out: set[tuple[int, int]] = set()
    for i, row in enumerate(neighbors):
        for j in row:
            a, b = (i, int(j)) if i < j else (int(j), i)
            out.add((a, b))
    return out
