"""Tests for parsing, normalization and tree comparison."""

from __future__ import annotations

import pytest

from codeplag.ast_analysis.normalizer import NormNode, normalize
from codeplag.ast_analysis.parser import (
    CodeParseError,
    UnsupportedLanguageError,
    detect_language,
    parse_file,
    parse_source,
)
from codeplag.ast_analysis.tree_distance import (
    compare_sources,
    find_matches,
    label_ngrams,
    normalize_source,
    prune_to_size,
    subtree_overlap,
    ted_similarity,
    tree_edit_distance,
)
from codeplag.core.config import AstConfig
from tests.conftest import FIXTURES, read_fixture

# --------------------------------------------------------------------------- parser


@pytest.mark.parametrize(
    ("name", "lang"),
    [("a.py", "python"), ("B.JAVA", "java"), ("x.js", "javascript"), ("m.mjs", "javascript")],
)
def test_detect_language(name: str, lang: str) -> None:
    assert detect_language(name) == lang


def test_detect_language_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedLanguageError):
        detect_language("main.cpp")


def test_parse_source_rejects_unknown_language() -> None:
    with pytest.raises(UnsupportedLanguageError):
        parse_source("int main() {}", "cpp")


def test_syntax_error_is_recovered_and_counted() -> None:
    parsed = parse_file(FIXTURES / "python" / "syntax_error.py")
    assert parsed.has_errors
    assert parsed.error_count >= 1


def test_syntax_error_raises_in_strict_mode() -> None:
    with pytest.raises(CodeParseError):
        parse_file(FIXTURES / "python" / "syntax_error.py", strict=True)


def test_parse_file_missing_raises() -> None:
    with pytest.raises(CodeParseError):
        parse_file(FIXTURES / "python" / "does_not_exist.py")


def test_empty_file_is_empty() -> None:
    parsed = parse_file(FIXTURES / "python" / "empty.py")
    assert parsed.is_empty
    assert not parsed.has_errors


# --------------------------------------------------------------------------- normalizer


def _labels(code: str, language: str = "python", **kwargs: bool) -> list[str]:
    return [n.label for n in normalize(parse_source(code, language), **kwargs).tree.preorder()]


def test_comments_and_docstrings_are_removed() -> None:
    with_noise = 'def f(a):\n    """Doc."""\n    # comment\n    return a  # trailing\n'
    clean = "def f(a):\n    return a\n"
    assert _labels(with_noise) == _labels(clean)


def test_identifiers_and_literals_are_canonicalized() -> None:
    assert _labels("total = count + 1\n") == _labels("x = y + 99\n")
    assert _labels('s = "hello"\n') == _labels("s = 'bye'\n")
    assert "STR" in _labels('s = "a"\n') and "NUM" in _labels("n = 3.5\n")


def test_library_names_are_kept() -> None:
    labels = _labels("print(len(items))\n")
    assert "ID:print" in labels and "ID:len" in labels


def test_alpha_renaming_tokens_follow_first_occurrence() -> None:
    a = normalize_source("def f(a, b):\n    c = a + b\n    return c\n", "python")
    b = normalize_source("def g(x, y):\n    z = x + y\n    return z\n", "python")
    assert a.tokens == b.tokens
    assert a.tokens[:3] == ["v1", "v2", "v3"]


def test_scopes_resolve_outer_names() -> None:
    norm = normalize_source("n = 1\ndef f():\n    return n\n", "python")
    assert norm.tokens.count("v1") == 2  # `n` resolved to the outer definition


def test_loops_and_compound_assignments_are_canonicalized() -> None:
    for_loop = "i = 0\nwhile i < 3:\n    s += i\n    i += 1\n"
    other = "i = 0\nwhile i < 3:\n    s = s + i\n    i = i + 1\n"
    assert _labels(for_loop) == _labels(other)
    assert "LOOP" in _labels("for x in y:\n    pass\n")
    assert "LOOP" not in _labels("for x in y:\n    pass\n", canonicalize_constructs=False)


def test_java_increment_matches_explicit_addition() -> None:
    a = "class A { void f() { int i = 0; i++; } }"
    b = "class A { void f() { int k = 0; k = k + 1; } }"
    assert _labels(a, "java") == _labels(b, "java")


def test_parentheses_are_hoisted() -> None:
    assert _labels("x = (a + b)\n") == _labels("x = a + b\n")


def test_normnode_helpers() -> None:
    tree = NormNode("r", [NormNode("a", [NormNode("c")]), NormNode("b")])
    assert tree.size() == 4
    assert tree.depth() == 3
    assert [n.label for n in tree.preorder()] == ["r", "a", "c", "b"]


# --------------------------------------------------------------------------- tree distance


def test_identical_files_score_one() -> None:
    code = read_fixture("python/original.py")
    result = compare_sources(code, code, "python")
    assert result.features == {"ted": 1.0, "subtree": 1.0, "ngram": 1.0}
    assert result.score == pytest.approx(1.0)


def test_renamed_copy_scores_high_and_unrelated_scores_low() -> None:
    original = read_fixture("python/original.py")
    renamed = compare_sources(original, read_fixture("python/renamed.py"), "python")
    different = compare_sources(original, read_fixture("python/different.py"), "python")
    assert renamed.score > 0.9
    assert different.score < 0.5
    assert renamed.score > different.score + 0.4


def test_reordered_functions_keep_subtree_overlap() -> None:
    original = read_fixture("python/original.py")
    result = compare_sources(original, read_fixture("python/reordered.py"), "python")
    assert result.features["subtree"] > 0.9
    assert result.matches and result.matches[0].label == "function_definition"


def test_java_and_javascript_renames() -> None:
    java = compare_sources(
        read_fixture("java/Original.java"), read_fixture("java/Renamed.java"), "java"
    )
    js = compare_sources(
        read_fixture("javascript/a.js"), read_fixture("javascript/b.js"), "javascript"
    )
    assert java.score > 0.6  # rename + for->while rewrite
    assert js.score > 0.95


def test_empty_file_gives_zero_and_warning() -> None:
    result = compare_sources("", read_fixture("python/original.py"), "python")
    assert result.score == 0.0
    assert any("empty" in w for w in result.warnings)


def test_syntax_errors_and_language_mismatch_are_reported() -> None:
    result = compare_sources(
        read_fixture("python/syntax_error.py"), read_fixture("java/Original.java"), "python", "java"
    )
    assert any("different languages" in w for w in result.warnings)
    assert any("syntax error" in w for w in result.warnings)
    assert 0.0 <= result.score <= 1.0


def test_tree_edit_distance_basics() -> None:
    a = NormNode("f", [NormNode("a"), NormNode("b")])
    b = NormNode("f", [NormNode("a"), NormNode("c")])
    assert tree_edit_distance(a, a) == 0
    assert tree_edit_distance(a, b) == 1
    sim, approx = ted_similarity(a, b, max_nodes=100)
    assert sim == pytest.approx(1 - 1 / 3)
    assert not approx


def test_size_guard_prunes_large_trees() -> None:
    deep = NormNode(
        "root", [NormNode("n", [NormNode("leaf") for _ in range(50)]) for _ in range(50)]
    )
    pruned, cut = prune_to_size(deep, max_nodes=60)
    assert cut
    assert pruned.size() <= 60
    small, cut_small = prune_to_size(NormNode("x"), max_nodes=10)
    assert not cut_small and small.size() == 1


def test_large_comparison_is_flagged_approximate() -> None:
    body = "\n".join(f"x{i} = x{i - 1} + {i}" for i in range(1, 200))
    result = compare_sources(body, body, "python", config=AstConfig(max_nodes=100))
    assert result.approximate
    assert any("approximated" in w for w in result.warnings)
    assert result.score == pytest.approx(1.0)


def test_subtree_overlap_and_ngrams() -> None:
    a = normalize_source(read_fixture("python/original.py"), "python").tree
    b = normalize_source(read_fixture("python/different.py"), "python").tree
    assert subtree_overlap(a, a, 4) == 1.0
    assert subtree_overlap(a, b, 4) < 0.3
    assert sum(label_ngrams(NormNode("x"), 3).values()) == 1


def test_find_matches_reports_disjoint_regions() -> None:
    a = normalize_source(read_fixture("python/original.py"), "python").tree
    b = normalize_source(read_fixture("python/reordered.py"), "python").tree
    matches = find_matches(a, b, min_size=4, top_k=5)
    assert matches
    assert matches == sorted(matches, key=lambda m: -m.size)
    for m in matches:
        assert m.lines_a[0] <= m.lines_a[1] and m.lines_b[0] <= m.lines_b[1]


def test_fast_ted_matches_zss_reference() -> None:
    import random

    from codeplag.ast_analysis.tree_distance import zss_reference_distance

    rng = random.Random(0)

    def random_tree(n: int) -> NormNode:
        nodes = [NormNode(rng.choice("abc"))]
        for _ in range(n - 1):
            child = NormNode(rng.choice("abc"))
            rng.choice(nodes).children.append(child)
            nodes.append(child)
        return nodes[0]

    for _ in range(100):
        a, b = random_tree(rng.randint(1, 20)), random_tree(rng.randint(1, 20))
        assert tree_edit_distance(a, b) == zss_reference_distance(a, b)
