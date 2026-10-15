"""Loading, cleaning and splitting Project CodeNet benchmark archives.

The archives (``Project_CodeNet_Python800.tar.gz``, ``Project_CodeNet_Java250.tar.gz``) are
read directly with :mod:`tarfile`; they are never extracted, which avoids writing ~300k tiny
files to disk. Each member path has the form ``<archive>/<problem_id>/<submission_id>.<ext>``.

Cleaning steps, applied per file:

1. Decode as UTF-8 (undecodable bytes replaced) and normalize line endings to ``\\n``.
2. Drop files larger than ``max_file_bytes`` or with fewer than ``min_file_lines`` non-blank
   lines.
3. Drop files that do not parse: Python files must parse with Python 3's ``ast`` (this
   removes Python 2 submissions); Java files must parse with tree-sitter without errors.
4. Drop exact duplicates (same content after stripping whitespace) within a problem.
"""

from __future__ import annotations

import ast
import hashlib
import random
import tarfile
import urllib.request
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from codeplag.ast_analysis.parser import parse_source
from codeplag.core.logging import get_logger

logger = get_logger(__name__)

_EXT_LANG = {".py": "python", ".java": "java"}


@dataclass(frozen=True)
class CodeFile:
    """One cleaned source file.

    Attributes:
        problem: CodeNet problem id (e.g. ``p02709``).
        submission: Submission id (e.g. ``s078445571``).
        language: ``python`` or ``java``.
        code: Cleaned source text.
    """

    problem: str
    submission: str
    language: str
    code: str

    @property
    def uid(self) -> str:
        """Unique id ``<problem>/<submission>``."""
        return f"{self.problem}/{self.submission}"


@dataclass
class LoadStats:
    """Counters collected while loading an archive."""

    total_files: int = 0
    problems: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    kept: int = 0
    sampled: int = 0
    bytes_kept: list[int] = field(default_factory=list)
    lines_kept: list[int] = field(default_factory=list)


def download_archive(base_url: str, name: str, dest_dir: Path) -> Path:
    """Download ``name`` from ``base_url`` into ``dest_dir`` unless it already exists.

    Args:
        base_url: URL prefix of the archive.
        name: Archive file name.
        dest_dir: Target directory.

    Returns:
        Path to the local archive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    if target.is_file() and target.stat().st_size > 0:
        logger.info("Using existing archive %s", target)
        return target
    url = f"{base_url.rstrip('/')}/{name}"
    logger.info("Downloading %s", url)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(target)
    return target


def _is_valid(code: str, language: str) -> bool:
    if language == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(code)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return False
        return True
    return not parse_source(code, language).has_errors


def load_archive(
    archive: Path,
    *,
    files_per_problem: int,
    max_file_bytes: int,
    min_file_lines: int,
    seed: int,
    problems: set[str] | None = None,
) -> tuple[dict[str, list[CodeFile]], LoadStats]:
    """Read, clean and subsample a CodeNet benchmark archive.

    Files are grouped by problem; within each problem a seeded random sample of
    ``files_per_problem`` cleaned files is kept.

    Args:
        archive: Path to the ``.tar.gz`` archive.
        files_per_problem: Number of cleaned files to keep per problem.
        max_file_bytes: Size limit (bytes).
        min_file_lines: Minimum number of non-blank lines.
        seed: Seed for the per-problem sampling.
        problems: Restrict loading to these problem ids.

    Returns:
        Mapping problem id -> cleaned files, and loading statistics.
    """
    stats = LoadStats()
    raw: dict[str, list[tuple[str, str, bytes]]] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            problem, filename = parts[-2], parts[-1]
            suffix = Path(filename).suffix
            if suffix not in _EXT_LANG:
                continue
            stats.total_files += 1
            if problems is not None and problem not in problems:
                continue
            if member.size > max_file_bytes:
                stats.dropped["too_large"] += 1
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            raw.setdefault(problem, []).append(
                (Path(filename).stem, _EXT_LANG[suffix], handle.read())
            )

    stats.problems = len(raw)
    rng = random.Random(seed)
    result: dict[str, list[CodeFile]] = {}
    for problem in sorted(raw):
        entries = sorted(raw[problem])
        rng.shuffle(entries)
        seen: set[str] = set()
        kept: list[CodeFile] = []
        for submission, language, data in entries:
            code = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            lines = [ln for ln in code.splitlines() if ln.strip()]
            if len(lines) < min_file_lines:
                stats.dropped["too_short"] += 1
                continue
            digest = hashlib.sha1("".join(code.split()).encode()).hexdigest()
            if digest in seen:
                stats.dropped["duplicate"] += 1
                continue
            if not _is_valid(code, language):
                stats.dropped["parse_error"] += 1
                continue
            seen.add(digest)
            stats.kept += 1
            if len(kept) < files_per_problem:
                kept.append(CodeFile(problem, submission, language, code))
                stats.bytes_kept.append(len(code.encode()))
                stats.lines_kept.append(len(lines))
        if len(kept) >= 2:
            result[problem] = kept
            stats.sampled += len(kept)
        else:
            stats.dropped["problem_too_small"] += 1
    return result, stats


def split_problems(
    problems: list[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[str]]:
    """Split problem ids into train/val/test so no problem appears in two splits.

    Args:
        problems: Problem ids.
        ratios: Train, validation and test fractions (normalized if they do not sum to 1).
        seed: Shuffle seed.

    Returns:
        Mapping ``train``/``val``/``test`` -> sorted problem ids.
    """
    ordered = sorted(problems)
    random.Random(seed).shuffle(ordered)
    total = sum(ratios)
    n = len(ordered)
    n_train = int(round(n * ratios[0] / total))
    n_val = int(round(n * ratios[1] / total))
    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }
