"""Structural similarity between normalized trees.

Three features are computed for a pair of normalized trees, each in ``[0, 1]``:

* ``ted``: tree edit distance (Zhang-Shasha, via ``zss``) with unit costs, converted to a
  similarity as ``1 - d / max(n1, n2)`` and clipped to ``[0, 1]``.
* ``subtree``: multiset Dice overlap of hashes of all subtrees with at least
  ``min_subtree_size`` nodes. Robust to reordering of functions and statements.
* ``ngram``: multiset Dice overlap of label n-grams taken from the pre-order traversal.

Complexity: Zhang-Shasha runs in ``O(n1 * n2 * min(d1, l1) * min(d2, l2))`` time and
``O(n1 * n2)`` memory (``n`` nodes, ``d`` depth, ``l`` leaves); in pure Python it becomes slow
above a few thousand nodes. The size guard therefore prunes any tree larger than
``max_nodes`` by cutting it at the largest depth for which the pruned tree still fits (cut
subtrees are replaced by a leaf with the subtree root's label plus ``~``). The comparison is
then flagged ``approximate``. Subtree hashing and n-grams are linear and always exact.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import zss

from codeplag.ast_analysis.normalizer import NormalizedSource, NormNode, normalize
from codeplag.ast_analysis.parser import ParsedSource, parse_source
from codeplag.core.config import AstConfig


@dataclass(frozen=True)
class MatchRegion:
    """A subtree present in both files.

    Attributes:
        lines_a: Inclusive (start, end) line range in file A.
        lines_b: Inclusive (start, end) line range in file B.
        size: Number of normalized nodes in the shared subtree.
        label: Label of the subtree root (e.g. ``function_definition``).
    """

    lines_a: tuple[int, int]
    lines_b: tuple[int, int]
    size: int
    label: str


@dataclass
class AstComparison:
    """Result of comparing two files structurally.

    Attributes:
        features: ``ted``, ``subtree`` and ``ngram`` similarities.
        score: Weighted combination of the features.
        matches: Largest shared subtrees, biggest first.
        approximate: Whether the size guard pruned a tree before tree edit distance.
        sizes: Node counts of the normalized trees (A, B).
        warnings: Human-readable caveats (syntax errors, empty input, language mismatch).
    """

    features: dict[str, float]
    score: float
    matches: list[MatchRegion] = field(default_factory=list)
    approximate: bool = False
    sizes: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)


def _label_dist(a: str, b: str) -> int:
    return 0 if a == b else 1


def _children(node: NormNode) -> list[NormNode]:
    return node.children


def _label(node: NormNode) -> str:
    return node.label


def prune_to_size(tree: NormNode, max_nodes: int) -> tuple[NormNode, bool]:
    """Depth-prune ``tree`` until it has at most ``max_nodes`` nodes.

    Args:
        tree: Normalized tree.
        max_nodes: Node budget (at least 1).

    Returns:
        The (possibly) pruned copy and whether pruning happened.
    """
    if tree.size() <= max_nodes:
        return tree, False

    # Count nodes per depth level to find the deepest cut that fits the budget.
    per_level: Counter[int] = Counter()
    stack = [(tree, 1)]
    while stack:
        node, d = stack.pop()
        per_level[d] += 1
        stack.extend((c, d + 1) for c in node.children)
    total, keep_depth = 0, 1
    for d in sorted(per_level):
        if total + per_level[d] > max_nodes:
            break
        total += per_level[d]
        keep_depth = d

    def copy(node: NormNode, d: int) -> NormNode:
        if d >= keep_depth and node.children:
            return NormNode(node.label + "~", [], node.start_line, node.end_line)
        return NormNode(
            node.label, [copy(c, d + 1) for c in node.children], node.start_line, node.end_line
        )

    return copy(tree, 1), True


def tree_edit_distance(a: NormNode, b: NormNode) -> int:
    """Unit-cost Zhang-Shasha tree edit distance.

    The trees are annotated with ``zss.AnnotatedTree`` (post-order, leftmost leaf
    descendants and keyroots); the dynamic program is the Zhang-Shasha recurrence used by
    ``zss.distance``, minus the per-cell edit-script bookkeeping that ``zss`` 1.2 always
    performs (it copies an operation list into every cell, which made it orders of magnitude
    slower). Results are identical to ``zss.simple_distance`` with unit costs; the unit tests
    check this.
    """
    ta, tb = zss.AnnotatedTree(a, _children), zss.AnnotatedTree(b, _children)
    la = [n.label for n in ta.nodes]
    lb = [n.label for n in tb.nodes]
    al, bl = ta.lmds, tb.lmds
    td = [[0] * len(lb) for _ in range(len(la))]

    for i in ta.keyroots:
        ali = al[i]
        ioff = ali - 1
        m = i - ali + 2
        for j in tb.keyroots:
            blj = bl[j]
            joff = blj - 1
            n = j - blj + 2
            fd = [[0] * n for _ in range(m)]
            for y in range(1, n):
                fd[0][y] = y
            for x in range(1, m):
                xi = x + ioff
                fdx, fdp = fd[x], fd[x - 1]
                fdx[0] = x
                lx, alx, tdx = la[xi], al[xi], td[xi]
                same_root_x = alx == ali
                p = alx - 1 - ioff
                for y in range(1, n):
                    yj = y + joff
                    remove, insert = fdp[y] + 1, fdx[y - 1] + 1
                    if same_root_x and bl[yj] == blj:
                        update = fdp[y - 1] + (lx != lb[yj])
                        best = remove if remove < insert else insert
                        best = update if update < best else best
                        fdx[y] = best
                        tdx[yj] = best
                    else:
                        sub = fd[p][bl[yj] - 1 - joff] + tdx[yj]
                        best = remove if remove < insert else insert
                        fdx[y] = sub if sub < best else best
    return td[-1][-1]


def zss_reference_distance(a: NormNode, b: NormNode) -> int:
    """Unit-cost distance computed directly with ``zss.simple_distance`` (slow; for tests)."""
    return int(zss.simple_distance(a, b, _children, _label, _label_dist))


def ted_similarity(a: NormNode, b: NormNode, max_nodes: int) -> tuple[float, bool]:
    """Tree edit distance converted to a similarity in ``[0, 1]``.

    Args:
        a: First tree.
        b: Second tree.
        max_nodes: Size guard passed to :func:`prune_to_size`.

    Returns:
        The similarity and whether either tree was pruned.
    """
    pa, cut_a = prune_to_size(a, max_nodes)
    pb, cut_b = prune_to_size(b, max_nodes)
    na, nb = pa.size(), pb.size()
    distance = tree_edit_distance(pa, pb)
    sim = 1.0 - distance / max(na, nb, 1)
    return min(1.0, max(0.0, sim)), cut_a or cut_b


def subtree_hashes(tree: NormNode) -> dict[int, tuple[str, int]]:
    """Compute a stable structural hash for every subtree.

    Args:
        tree: Normalized tree.

    Returns:
        Mapping from ``id(node)`` to ``(hash, size)``.
    """
    result: dict[int, tuple[str, int]] = {}
    # Iterative post-order.
    stack: list[tuple[NormNode, bool]] = [(tree, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            stack.extend((c, False) for c in node.children)
            continue
        h = hashlib.blake2b(node.label.encode(), digest_size=8)
        size = 1
        for child in node.children:
            child_hash, child_size = result[id(child)]
            h.update(child_hash.encode())
            size += child_size
        result[id(node)] = (h.hexdigest(), size)
    return result


def _dice(a: Counter[str], b: Counter[str]) -> float:
    total = sum(a.values()) + sum(b.values())
    if total == 0:
        return 0.0
    inter = sum((a & b).values())
    return 2.0 * inter / total


def subtree_overlap(a: NormNode, b: NormNode, min_size: int) -> float:
    """Multiset Dice overlap of subtree hashes for subtrees with ``>= min_size`` nodes."""
    ha = Counter(h for h, s in subtree_hashes(a).values() if s >= min_size)
    hb = Counter(h for h, s in subtree_hashes(b).values() if s >= min_size)
    return _dice(ha, hb)


def label_ngrams(tree: NormNode, n: int) -> Counter[str]:
    """Count label n-grams over the pre-order traversal of ``tree``."""
    labels = [node.label for node in tree.preorder()]
    if len(labels) < n:
        return Counter(["|".join(labels)]) if labels else Counter()
    return Counter("|".join(labels[i : i + n]) for i in range(len(labels) - n + 1))


def ngram_overlap(a: NormNode, b: NormNode, n: int) -> float:
    """Multiset Dice overlap of label n-grams."""
    return _dice(label_ngrams(a, n), label_ngrams(b, n))


def _descendant_ids(node: NormNode) -> Iterable[int]:
    return (id(n) for n in node.preorder())


def find_matches(a: NormNode, b: NormNode, min_size: int, top_k: int) -> list[MatchRegion]:
    """Find the largest maximal subtrees shared by ``a`` and ``b``.

    A subtree is reported only if no already-reported larger subtree contains it, on either
    side.

    Args:
        a: First tree.
        b: Second tree.
        min_size: Ignore shared subtrees smaller than this.
        top_k: Maximum number of regions to return.

    Returns:
        Matching regions sorted by size, largest first.
    """
    hashes_a, hashes_b = subtree_hashes(a), subtree_hashes(b)
    index_b: dict[str, list[NormNode]] = {}
    for node in b.preorder():
        h, s = hashes_b[id(node)]
        if s >= min_size:
            index_b.setdefault(h, []).append(node)

    candidates = [
        (hashes_a[id(node)][1], node) for node in a.preorder() if hashes_a[id(node)][0] in index_b
    ]
    candidates = [(s, n) for s, n in candidates if s >= min_size]
    candidates.sort(key=lambda item: -item[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    regions: list[MatchRegion] = []
    for size, node_a in candidates:
        if id(node_a) in used_a:
            continue
        partner = next((n for n in index_b[hashes_a[id(node_a)][0]] if id(n) not in used_b), None)
        if partner is None:
            continue
        used_a.update(_descendant_ids(node_a))
        used_b.update(_descendant_ids(partner))
        regions.append(
            MatchRegion(
                lines_a=(node_a.start_line, node_a.end_line),
                lines_b=(partner.start_line, partner.end_line),
                size=size,
                label=node_a.label,
            )
        )
        if len(regions) >= top_k:
            break
    return regions


def combine_features(features: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average of AST features using ``weights`` (missing keys count as 0)."""
    total = sum(weights.get(k, 0.0) for k in features)
    if total <= 0:
        return sum(features.values()) / max(len(features), 1)
    return sum(features[k] * weights.get(k, 0.0) for k in features) / total


def compare_normalized(
    a: NormalizedSource, b: NormalizedSource, config: AstConfig | None = None
) -> AstComparison:
    """Compare two normalized sources.

    Args:
        a: First normalized source.
        b: Second normalized source.
        config: AST settings; defaults to :class:`AstConfig` defaults.

    Returns:
        Structural features, AST score and matching regions.
    """
    cfg = config or AstConfig()
    warnings: list[str] = []
    if a.language != b.language:
        warnings.append(f"Comparing different languages ({a.language} vs {b.language}).")
    for name, src in (("A", a), ("B", b)):
        if src.error_count:
            warnings.append(f"File {name} has {src.error_count} syntax error node(s).")
    sizes = (a.tree.size(), b.tree.size())
    if not a.tree.children or not b.tree.children:
        warnings.append("At least one file is empty; structural similarity set to 0.")
        zeros = {"ted": 0.0, "subtree": 0.0, "ngram": 0.0}
        return AstComparison(features=zeros, score=0.0, sizes=sizes, warnings=warnings)

    ted, approximate = ted_similarity(a.tree, b.tree, cfg.max_nodes)
    if approximate:
        warnings.append(f"Tree larger than {cfg.max_nodes} nodes; tree edit distance approximated.")
    features = {
        "ted": ted,
        "subtree": subtree_overlap(a.tree, b.tree, cfg.min_subtree_size),
        "ngram": ngram_overlap(a.tree, b.tree, cfg.ngram_size),
    }
    return AstComparison(
        features=features,
        score=combine_features(features, cfg.weights),
        matches=find_matches(a.tree, b.tree, cfg.min_subtree_size, cfg.top_matches),
        approximate=approximate,
        sizes=sizes,
        warnings=warnings,
    )


def normalize_source(
    code: str | bytes, language: str, config: AstConfig | None = None
) -> NormalizedSource:
    """Parse and normalize source code in one step."""
    cfg = config or AstConfig()
    parsed: ParsedSource = parse_source(code, language)
    return normalize(parsed, canonicalize_constructs=cfg.canonicalize_constructs)


def compare_sources(
    code_a: str | bytes,
    code_b: str | bytes,
    language_a: str,
    language_b: str | None = None,
    config: AstConfig | None = None,
) -> AstComparison:
    """Parse, normalize and compare two source strings.

    Args:
        code_a: Source of file A.
        code_b: Source of file B.
        language_a: Language of file A.
        language_b: Language of file B (defaults to ``language_a``).
        config: AST settings.

    Returns:
        The structural comparison.
    """
    cfg = config or AstConfig()
    na = normalize_source(code_a, language_a, cfg)
    nb = normalize_source(code_b, language_b or language_a, cfg)
    return compare_normalized(na, nb, cfg)
