"""Tests for plagiarism transformations, CodeNet loading, splitting and pair building."""

from __future__ import annotations

import ast
import contextlib
import io
import random
import tarfile
from pathlib import Path

import pytest

from codeplag.core.config import Settings
from codeplag.data.codenet import CodeFile, load_archive, split_problems
from codeplag.data.pairs import (
    BOILERPLATE_PY,
    PairBuilder,
    prepare_dataset,
    read_jsonl,
    token_jaccard,
    write_jsonl,
)
from codeplag.data.transforms import (
    JAVA_TRANSFORMS,
    PYTHON_TRANSFORMS,
    apply_transforms,
    available_transforms,
    transform_python,
)
from tests.conftest import read_fixture

PROGRAM = """
def solve(values, limit):
    total = 0
    count = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            total += values[i] - values[i - 1]
        else:
            count += 1
    for j in range(limit):
        total -= 1
    return total, count


data = [3, 1, 4, 1, 5, 9, 2, 6]
a = 10
b = 20
print(solve(data, 2), a + b)
"""


def _run(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<test>", "exec"), {"__name__": "__main__"})  # noqa: S102
    return out.getvalue()


@pytest.mark.parametrize("name", PYTHON_TRANSFORMS)
def test_each_python_transform_preserves_behaviour(name: str) -> None:
    result = transform_python(PROGRAM, [name], random.Random(3))
    assert name in result.applied
    ast.parse(result.code)
    assert result.code.strip() != PROGRAM.strip()
    assert _run(result.code) == _run(PROGRAM)


def test_all_transforms_stacked_preserve_behaviour() -> None:
    for seed in range(10):
        result = transform_python(PROGRAM, list(PYTHON_TRANSFORMS), random.Random(seed))
        assert len(result.applied) >= 4
        assert _run(result.code) == _run(PROGRAM)


def test_transforms_are_deterministic_for_a_seed() -> None:
    a = transform_python(PROGRAM, list(PYTHON_TRANSFORMS), random.Random(7))
    b = transform_python(PROGRAM, list(PYTHON_TRANSFORMS), random.Random(7))
    c = transform_python(PROGRAM, list(PYTHON_TRANSFORMS), random.Random(8))
    assert a == b
    assert a.code != c.code


def test_rename_skips_keyword_arguments_and_imports() -> None:
    code = "import math\ndef f(sep):\n    return math.sqrt(4)\nprint(f(1), sep='')\n"
    result = transform_python(code, ["rename"], random.Random(0))
    assert "sep=''" in result.code and "math.sqrt" in result.code and "def f(" not in result.code


def test_loop_swap_skips_unsafe_loops() -> None:
    uses_var_after = "for i in range(3):\n    pass\nprint(i)\n"
    has_continue = "for i in range(3):\n    if i:\n        continue\n    print(i)\n"
    for code in (uses_var_after, has_continue):
        assert "loop_swap" not in transform_python(code, ["loop_swap"], random.Random(0)).applied


def test_unknown_transform_raises() -> None:
    with pytest.raises(ValueError):
        transform_python(PROGRAM, ["nope"], random.Random(0))
    with pytest.raises(ValueError):
        apply_transforms("class A {}", "java", ["nope"], random.Random(0))


def test_java_transforms_keep_code_parseable() -> None:
    from codeplag.ast_analysis.parser import parse_source

    code = read_fixture("java/Original.java")
    result = apply_transforms(code, "java", list(JAVA_TRANSFORMS), random.Random(1))
    assert set(result.applied) == set(JAVA_TRANSFORMS)
    assert "sumSquares" not in result.code  # method renamed
    assert "System.out.println" in result.code and "main(String[] args)" in result.code
    assert not parse_source(result.code, "java").has_errors
    assert available_transforms("java") == JAVA_TRANSFORMS
    assert available_transforms("python") == PYTHON_TRANSFORMS


def _make_archive(path: Path, problems: dict[str, dict[str, str]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for problem, files in problems.items():
            for name, code in files.items():
                data = code.encode()
                info = tarfile.TarInfo(f"Project_CodeNet_Test/{problem}/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


def _python_problem(seed: int, n: int) -> dict[str, str]:
    return {
        f"s{seed}{i:03d}.py": f"x = int(input())\ny = x * {seed} + {i}\nfor k in range({i + 1}):\n"
        f"    y += k\nprint(y)\n"
        for i in range(n)
    }


def test_load_archive_cleans_and_samples(tmp_path: Path) -> None:
    files = _python_problem(1, 5)
    files["s_py2.py"] = "print 'python 2'\nx = 1\ny = 2\n"
    files["s_short.py"] = "print(1)\n"
    files["s_dup.py"] = files["s1000.py"].replace("\n", "\n\n")
    files["s_big.py"] = "x = 1\n" * 5000
    archive = tmp_path / "a.tar.gz"
    _make_archive(archive, {"p00001": files, "p00002": _python_problem(2, 4)})
    loaded, stats = load_archive(
        archive, files_per_problem=3, max_file_bytes=2000, min_file_lines=3, seed=0
    )
    assert set(loaded) == {"p00001", "p00002"}
    assert all(len(v) == 3 for v in loaded.values())
    assert stats.total_files == 13
    assert stats.dropped["parse_error"] == 1
    assert stats.dropped["too_short"] == 1
    assert stats.dropped["duplicate"] == 1
    assert stats.dropped["too_large"] == 1


def test_split_problems_is_disjoint_and_deterministic() -> None:
    problems = [f"p{i:05d}" for i in range(100)]
    a = split_problems(problems, (0.7, 0.15, 0.15), seed=1)
    b = split_problems(problems, (0.7, 0.15, 0.15), seed=1)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (70, 15, 15)
    assert not set(a["train"]) & set(a["val"]) and not set(a["val"]) & set(a["test"])
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(problems)


def _files(n_problems: int = 6, per: int = 5) -> dict[str, list[CodeFile]]:
    return {
        f"p{p}": [
            CodeFile(f"p{p}", f"s{i}", "python", code)
            for i, code in enumerate(_python_problem(p + 1, per).values())
        ]
        for p in range(n_problems)
    }


def test_pair_builder_balanced_and_labeled() -> None:
    pairs = PairBuilder(_files(), "train", seed=0).balanced(40, synthetic_fraction=0.5)
    assert len(pairs) == 40
    assert sum(p.label for p in pairs) == 20
    kinds = {p.kind for p in pairs}
    assert kinds == {"same_problem", "synthetic", "different_problem"}
    for p in pairs:
        pa, pb = p.file_a.split("/")[0], p.file_b.split("/")[0]
        if p.kind == "different_problem":
            assert p.label == 0 and pa != pb
        else:
            assert p.label == 1 and pa == pb
        if p.kind == "synthetic":
            assert p.transforms and p.code_a != p.code_b


def test_hard_suites() -> None:
    builder = PairBuilder(_files(), "hard", seed=1)
    boiler = builder.collect("boilerplate_negative", 5)
    assert all(p.label == 0 and p.code_a.startswith(BOILERPLATE_PY) for p in boiler)
    equivalent = builder.equivalent_candidates(3, pool_factor=3)
    assert len(equivalent) == 3 and all(p.kind == "hard_equivalent" for p in equivalent)
    heavy = builder.collect("heavy_obfuscation", 3)
    assert all(len(p.transforms) >= 4 for p in heavy)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    pairs = PairBuilder(_files(), "val", seed=2).balanced(10, 0.5)
    write_jsonl(pairs, tmp_path / "x.jsonl")
    records = list(read_jsonl(tmp_path / "x.jsonl"))
    assert [r["id"] for r in records] == [p.id for p in pairs]


def test_token_jaccard() -> None:
    assert token_jaccard("a = b + c", "a = b + c", "python") == 1.0
    assert token_jaccard("a = 1", "while True: pass", "python") < 0.2
    assert token_jaccard("", "", "python") == 0.0


def test_prepare_dataset_end_to_end(settings: Settings) -> None:
    raw = settings.paths.raw_dir
    raw.mkdir(parents=True)
    _make_archive(
        raw / settings.data.primary_archive,
        {f"p{i:05d}": _python_problem(i + 1, 6) for i in range(20)},
    )
    java = read_fixture("java/Original.java")
    _make_archive(
        raw / settings.data.unseen_archive,
        {
            f"p{i:05d}": {f"s{j}.java": java.replace("1;", f"{j + i};") for j in range(4)}
            for i in range(4)
        },
    )
    settings.data.pairs = {"train": 20, "val": 10, "test": 10}
    settings.data.hard_pairs_per_suite = 4
    settings.data.unseen_pairs = 8
    settings.data.unseen_problems = 4
    settings.data.files_per_problem = 5
    stats = prepare_dataset(settings, download=False)
    out = settings.paths.processed_dir
    for name in ("train", "val", "test", "hard_obfuscation", "unseen_java"):
        assert (out / f"{name}.jsonl").is_file()
    assert stats["split_problems"] == {"train": 14, "val": 3, "test": 3}
    assert (settings.paths.results_dir / "dataset_stats.json").is_file()


def test_prepare_dataset_missing_archive(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_dataset(settings, download=False)
