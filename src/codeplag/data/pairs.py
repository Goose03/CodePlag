"""Labeled pair generation and the end-to-end ``prepare-data`` pipeline.

Pair kinds (``kind`` field):

* ``same_problem`` (label 1): two accepted solutions to the same problem (type-4 clones).
* ``synthetic`` (label 1): a file and a transformed copy (``transforms`` lists what changed).
* ``different_problem`` (label 0): solutions to two different problems.

Hard suites, all drawn from the test split:

* ``hard_obfuscation`` (label 1): every available transformation stacked (at least 4 applied).
* ``hard_equivalent`` (label 1): same-problem pairs with the *lowest* lexical token overlap
  among sampled candidates, a detector-independent proxy for "different but equivalent
  algorithm".
* ``hard_boilerplate`` (label 0): different-problem pairs that share an identical template
  header (imports, fast-I/O helpers, constants) prepended to both files.

The unseen-language set uses CodeNet Java250 with the same pair kinds.

Every file and problem belongs to exactly one split, so no pair crosses split boundaries.
"""

from __future__ import annotations

import io
import json
import random
import statistics
import tarfile
import tokenize
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from codeplag.core.config import Settings
from codeplag.core.logging import get_logger
from codeplag.data.codenet import (
    CodeFile,
    LoadStats,
    download_archive,
    load_archive,
    split_problems,
)
from codeplag.data.transforms import apply_transforms, available_transforms

logger = get_logger(__name__)

BOILERPLATE_PY = """import sys
import math
from collections import defaultdict, deque

sys.setrecursionlimit(10 ** 7)
INF = float("inf")
MOD = 10 ** 9 + 7


def read_int():
    return int(sys.stdin.readline())


def read_ints():
    return list(map(int, sys.stdin.readline().split()))


"""


@dataclass
class Pair:
    """A labeled code pair as stored in the processed JSONL files."""

    id: str
    split: str
    label: int
    kind: str
    lang_a: str
    lang_b: str
    file_a: str
    file_b: str
    code_a: str
    code_b: str
    transforms: list[str] = field(default_factory=list)


def lexical_tokens(code: str, language: str) -> list[str]:
    """Raw lexical tokens, used for detector-independent overlap measures."""
    if language == "python":
        try:
            return [
                t.string
                for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type
                not in (
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.COMMENT,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENDMARKER,
                )
            ]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
    return code.split()


def token_jaccard(a: str, b: str, language: str) -> float:
    """Jaccard overlap of the sets of lexical tokens."""
    sa, sb = set(lexical_tokens(a, language)), set(lexical_tokens(b, language))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class PairBuilder:
    """Builds labeled pairs from files grouped by problem.

    Args:
        files: Mapping problem id -> files (one split only).
        split: Split name stored in each pair.
        seed: Seed for the builder's random generator.
    """

    def __init__(self, files: dict[str, list[CodeFile]], split: str, seed: int) -> None:
        self.files = {p: fs for p, fs in files.items() if len(fs) >= 2}
        self.problems = sorted(self.files)
        self.split = split
        self.rng = random.Random(seed)
        self.counter = 0
        self.seen: set[tuple[str, ...]] = set()

    def _make(
        self, a: CodeFile, b: CodeFile, code_b: str, label: int, kind: str, transforms: list[str]
    ) -> Pair | None:
        key = (kind, *sorted((a.uid, b.uid if kind != "synthetic" else b.uid + "*")))
        if kind != "synthetic" and key in self.seen:
            return None
        self.seen.add(key)
        self.counter += 1
        return Pair(
            id=f"{self.split}-{self.counter:06d}",
            split=self.split,
            label=label,
            kind=kind,
            lang_a=a.language,
            lang_b=b.language,
            file_a=a.uid,
            file_b=b.uid if kind != "synthetic" else f"{a.uid}#synthetic",
            code_a=a.code,
            code_b=code_b,
            transforms=transforms,
        )

    def same_problem(self) -> Pair | None:
        """A positive pair of two solutions to one problem."""
        if not self.problems:
            return None
        a, b = self.rng.sample(self.files[self.rng.choice(self.problems)], 2)
        return self._make(a, b, b.code, 1, "same_problem", [])

    def different_problem(self) -> Pair | None:
        """A negative pair from two different problems."""
        if len(self.problems) < 2:
            return None
        pa, pb = self.rng.sample(self.problems, 2)
        a, b = self.rng.choice(self.files[pa]), self.rng.choice(self.files[pb])
        return self._make(a, b, b.code, 0, "different_problem", [])

    def synthetic(self, k_min: int = 1, k_max: int = 2, min_applied: int = 1) -> Pair | None:
        """A positive pair made of a file and a transformed copy of it."""
        if not self.problems:
            return None
        a = self.rng.choice(self.files[self.rng.choice(self.problems)])
        names = list(available_transforms(a.language))
        chosen = self.rng.sample(names, k=min(len(names), self.rng.randint(k_min, k_max)))
        try:
            result = apply_transforms(a.code, a.language, chosen, self.rng)
        except (SyntaxError, ValueError, RecursionError) as exc:
            logger.debug("Transformation failed for %s: %s", a.uid, exc)
            return None
        if len(result.applied) < min_applied or result.code.strip() == a.code.strip():
            return None
        return self._make(a, a, result.code, 1, "synthetic", result.applied)

    def heavy_obfuscation(self) -> Pair | None:
        """A positive pair with every transformation stacked."""
        names = len(available_transforms("python"))
        return self.synthetic(k_min=names, k_max=names, min_applied=4)

    def boilerplate_negative(self) -> Pair | None:
        """A negative pair whose files share an identical template header."""
        pair = self.different_problem()
        if pair is None:
            return None
        pair.kind = "hard_boilerplate"
        pair.code_a = BOILERPLATE_PY + pair.code_a
        pair.code_b = BOILERPLATE_PY + pair.code_b
        return pair

    def equivalent_candidates(self, n: int, pool_factor: int = 10) -> list[Pair]:
        """Same-problem pairs with the lowest lexical overlap among ``n * pool_factor``."""
        pool: list[tuple[float, Pair]] = []
        attempts = 0
        while len(pool) < n * pool_factor and attempts < n * pool_factor * 3:
            attempts += 1
            pair = self.same_problem()
            if pair is not None:
                pool.append((token_jaccard(pair.code_a, pair.code_b, pair.lang_a), pair))
        pool.sort(key=lambda item: item[0])
        chosen = [p for _, p in pool[:n]]
        for p in chosen:
            p.kind = "hard_equivalent"
        return chosen

    def collect(self, maker: str, n: int, **kwargs: int) -> list[Pair]:
        """Call the generator method ``maker`` until ``n`` pairs are produced."""
        fn = getattr(self, maker)
        out: list[Pair] = []
        attempts = 0
        while len(out) < n and attempts < n * 20:
            attempts += 1
            pair = fn(**kwargs)
            if pair is not None:
                out.append(pair)
        if len(out) < n:
            logger.warning("%s/%s: produced %d of %d pairs", self.split, maker, len(out), n)
        return out

    def balanced(self, total: int, synthetic_fraction: float) -> list[Pair]:
        """A class-balanced mix of same-problem, synthetic and negative pairs."""
        n_pos = total // 2
        n_syn = int(round(n_pos * synthetic_fraction))
        pairs = (
            self.collect("same_problem", n_pos - n_syn)
            + self.collect("synthetic", n_syn)
            + self.collect("different_problem", total - n_pos)
        )
        self.rng.shuffle(pairs)
        return pairs


def write_jsonl(pairs: list[Pair], path: Path) -> None:
    """Write pairs as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Iterate over the records of a JSONL file."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _pair_summary(pairs: list[Pair]) -> dict[str, object]:
    kinds: dict[str, int] = {}
    transforms: dict[str, int] = {}
    for p in pairs:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
        for t in p.transforms:
            transforms[t] = transforms.get(t, 0) + 1
    positives = sum(p.label for p in pairs)
    return {
        "pairs": len(pairs),
        "positives": positives,
        "negatives": len(pairs) - positives,
        "kinds": kinds,
        "transforms": transforms,
    }


def _load_summary(stats: LoadStats) -> dict[str, object]:
    return {
        "files_in_archive": stats.total_files,
        "problems_loaded": stats.problems,
        "files_kept_after_cleaning": stats.kept,
        "dropped": dict(stats.dropped),
        "files_sampled": stats.sampled,
        "bytes_median": statistics.median(stats.bytes_kept) if stats.bytes_kept else 0,
        "lines_median": statistics.median(stats.lines_kept) if stats.lines_kept else 0,
        "lines_mean": round(statistics.fmean(stats.lines_kept), 2) if stats.lines_kept else 0,
    }


def prepare_dataset(settings: Settings, *, download: bool = True) -> dict[str, object]:
    """Download (if needed), clean, split and pair the datasets.

    Writes ``train/val/test.jsonl``, ``hard_*.jsonl``, ``unseen_java.jsonl`` and
    ``splits.json`` to ``paths.processed_dir`` and returns a statistics dictionary (also
    written to ``dataset_stats.json`` in the processed and results directories).

    Args:
        settings: Loaded settings.
        download: Download missing archives.

    Returns:
        Dataset statistics.

    Raises:
        FileNotFoundError: If an archive is missing and ``download`` is false.
    """
    cfg, paths, seed = settings.data, settings.paths, settings.seed
    archives = {}
    for key, name in (("primary", cfg.primary_archive), ("unseen", cfg.unseen_archive)):
        local = paths.raw_dir / name
        if download:
            local = download_archive(cfg.archive_base_url, name, paths.raw_dir)
        elif not local.is_file():
            raise FileNotFoundError(f"Missing archive {local}; run prepare-data with download.")
        archives[key] = local

    logger.info("Loading %s", archives["primary"])
    files, load_stats = load_archive(
        archives["primary"],
        files_per_problem=cfg.files_per_problem,
        max_file_bytes=cfg.max_file_bytes,
        min_file_lines=cfg.min_file_lines,
        seed=seed,
    )
    splits = split_problems(list(files), cfg.split_ratios, seed)
    stats: dict[str, object] = {
        "primary": {"archive": cfg.primary_archive, **_load_summary(load_stats)},
        "split_problems": {k: len(v) for k, v in splits.items()},
    }
    out_dir = paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    pair_stats: dict[str, object] = {}
    for offset, split in enumerate(("train", "val", "test")):
        builder = PairBuilder({p: files[p] for p in splits[split]}, split, seed + 101 * offset)
        pairs = builder.balanced(cfg.pairs[split], cfg.synthetic_fraction)
        write_jsonl(pairs, out_dir / f"{split}.jsonl")
        pair_stats[split] = _pair_summary(pairs)
        logger.info("%s: %d pairs", split, len(pairs))

    test_files = {p: files[p] for p in splits["test"]}
    n_hard = cfg.hard_pairs_per_suite
    hard = PairBuilder(test_files, "hard", seed + 7)
    suites = {
        "hard_obfuscation": hard.collect("heavy_obfuscation", n_hard),
        "hard_equivalent": hard.equivalent_candidates(n_hard),
        "hard_boilerplate": hard.collect("boilerplate_negative", n_hard),
    }
    for name, pairs in suites.items():
        if name == "hard_obfuscation":
            for p in pairs:
                p.kind = name
        write_jsonl(pairs, out_dir / f"{name}.jsonl")
        pair_stats[name] = _pair_summary(pairs)

    logger.info("Loading %s", archives["unseen"])
    java_problem_ids = _list_problems(archives["unseen"])
    rng = random.Random(seed + 13)
    chosen = set(rng.sample(java_problem_ids, k=min(cfg.unseen_problems, len(java_problem_ids))))
    java_files, java_stats = load_archive(
        archives["unseen"],
        files_per_problem=cfg.files_per_problem,
        max_file_bytes=cfg.max_file_bytes,
        min_file_lines=cfg.min_file_lines,
        seed=seed,
        problems=chosen,
    )
    unseen = PairBuilder(java_files, "unseen_java", seed + 17).balanced(
        cfg.unseen_pairs, cfg.synthetic_fraction
    )
    write_jsonl(unseen, out_dir / "unseen_java.jsonl")
    pair_stats["unseen_java"] = _pair_summary(unseen)
    stats["unseen"] = {
        "archive": cfg.unseen_archive,
        "problems_total": len(java_problem_ids),
        **_load_summary(java_stats),
    }
    stats["pairs"] = pair_stats

    text = json.dumps(stats, indent=2)
    (out_dir / "dataset_stats.json").write_text(text, encoding="utf-8")
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "dataset_stats.json").write_text(text, encoding="utf-8")
    return stats


def _list_problems(archive: Path) -> list[str]:
    problems: set[str] = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            parts = member.name.split("/")
            if member.isfile() and len(parts) >= 3:
                problems.add(parts[-2])
    return sorted(problems)
