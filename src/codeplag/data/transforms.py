"""Seeded plagiarism transformations used to build synthetic positive pairs.

Python transformations operate on the standard-library ``ast`` and are written back with
``ast.unparse`` (which also discards comments and re-formats the code, as many plagiarists'
editors would). Each one is conservative: when it cannot prove a rewrite is safe it skips
that site, and it reports whether it changed anything.

=====================  ==================================================================
Name                   What it does (Python)
=====================  ==================================================================
``rename``             Renames user-defined variables, functions and parameters
                       consistently. Skips builtins, imports, dunder names and any name
                       used as a keyword argument.
``comments``           Inserts comment lines, trailing comments and blank lines.
``reorder``            Shuffles runs of adjacent top-level function/class definitions and
                       swaps adjacent simple statements that share no names and make no
                       calls.
``dead_code``          Inserts unused assignments, ``if False:`` blocks and loops that run
                       zero times.
``loop_swap``          Rewrites ``for v in range(...)`` into an equivalent ``while`` loop
                       when the body has no ``continue``/``else`` and ``v`` is not used
                       after the loop.
``expr_rewrite``       ``x op= c`` -> ``x = x op c``, ``a < b`` -> ``b > a`` (no calls),
                       ``if c: A else: B`` -> ``if not c: B else: A``.
=====================  ==================================================================

Java (and JavaScript) sources get a smaller, text-level set via tree-sitter: ``rename``,
``comments`` and ``dead_code``.
"""

from __future__ import annotations

import ast
import builtins
import io
import random
import tokenize
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tree_sitter import Node

from codeplag.ast_analysis.parser import parse_source

PYTHON_TRANSFORMS: tuple[str, ...] = (
    "rename",
    "comments",
    "reorder",
    "dead_code",
    "loop_swap",
    "expr_rewrite",
)
JAVA_TRANSFORMS: tuple[str, ...] = ("rename", "comments", "dead_code")

_NAME_STEMS = (
    "val",
    "tmp",
    "data",
    "res",
    "cnt",
    "arr",
    "num",
    "idx",
    "acc",
    "item",
    "elem",
    "buf",
    "total",
    "cur",
    "nxt",
    "prev",
    "left",
    "right",
    "key",
    "flag",
    "seq",
    "grid",
    "memo",
)
_COMMENTS = (
    "compute the answer",
    "read input",
    "helper",
    "edge case",
    "main logic",
    "TODO: optimize",
    "update state",
    "loop over values",
    "check condition",
    "store result",
)
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass
class TransformResult:
    """Output of applying a sequence of transformations.

    Attributes:
        code: Transformed source.
        applied: Names of the transformations that changed the code, in order.
    """

    code: str
    applied: list[str] = field(default_factory=list)


class _NamePool:
    """Generates fresh identifiers that do not collide with existing names."""

    def __init__(self, taken: set[str], rng: random.Random) -> None:
        self.taken = set(taken) | _BUILTIN_NAMES
        self.rng = rng

    def fresh(self) -> str:
        while True:
            name = f"{self.rng.choice(_NAME_STEMS)}{self.rng.randint(0, 99)}"
            if name not in self.taken:
                self.taken.add(name)
                return name


def _all_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
    return names


# ---------------------------------------------------------------------------- python


def _py_rename(tree: ast.Module, rng: random.Random) -> bool:
    imported: set[str] = set()
    keyword_names: set[str] = set()
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.alias):
            imported.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.keyword) and node.arg:
            keyword_names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined.add(node.name)
    candidates = sorted(
        n
        for n in defined
        if n not in imported
        and n not in keyword_names
        and n not in _BUILTIN_NAMES
        and not (n.startswith("__") and n.endswith("__"))
        and n not in ("self", "cls", "_")
    )
    if not candidates:
        return False
    pool = _NamePool(_all_names(tree), rng)
    mapping = {name: pool.fresh() for name in candidates}

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            node.id = mapping[node.id]
        elif isinstance(node, ast.arg) and node.arg in mapping:
            node.arg = mapping[node.arg]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in mapping:
            node.name = mapping[node.name]
        elif isinstance(node, ast.Global | ast.Nonlocal):
            node.names = [mapping.get(n, n) for n in node.names]
    return True


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _has_call(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call | ast.Await | ast.Yield | ast.YieldFrom | ast.NamedExpr)
        for n in ast.walk(node)
    )


def _bodies(tree: ast.AST) -> list[list[ast.stmt]]:
    out: list[list[ast.stmt]] = []
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            value = getattr(node, attr, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                out.append(value)
    return out


def _py_reorder(tree: ast.Module, rng: random.Random) -> bool:
    changed = False
    # Shuffle runs of adjacent top-level definitions.
    body = tree.body
    i = 0
    while i < len(body):
        j = i
        while j < len(body) and isinstance(
            body[j], ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            if getattr(body[j], "decorator_list", None):
                break
            j += 1
        if j - i >= 2:
            run = body[i:j]
            original = list(run)
            rng.shuffle(run)
            if run != original:
                body[i:j] = run
                changed = True
        i = max(j, i + 1)
    # Swap adjacent independent simple statements.
    simple = (ast.Assign, ast.AugAssign, ast.AnnAssign)
    for stmts in _bodies(tree):
        for k in range(len(stmts) - 1):
            a, b = stmts[k], stmts[k + 1]
            if not (isinstance(a, simple) and isinstance(b, simple)):
                continue
            if _has_call(a) or _has_call(b) or (_names_in(a) & _names_in(b)):
                continue
            if rng.random() < 0.7:
                stmts[k], stmts[k + 1] = b, a
                changed = True
    return changed


def _dead_stmt(pool: _NamePool, rng: random.Random) -> ast.stmt:
    name = pool.fresh()
    choice = rng.randrange(3)
    if choice == 0:
        src = f"{name} = {rng.randint(0, 1000)}"
    elif choice == 1:
        src = f"if False:\n    {name} = {rng.randint(0, 9)} * {rng.randint(0, 9)}"
    else:
        src = f"for {name} in range(0):\n    pass"
    return ast.parse(src).body[0]


def _py_dead_code(tree: ast.Module, rng: random.Random) -> bool:
    pool = _NamePool(_all_names(tree), rng)
    bodies = _bodies(tree)
    if not bodies:
        return False
    for _ in range(rng.randint(1, 3)):
        stmts = rng.choice(bodies)
        # Keep docstrings and `from __future__` imports first.
        start = 1 if stmts and isinstance(stmts[0], ast.ImportFrom | ast.Expr) else 0
        stmts.insert(rng.randint(start, len(stmts)), _dead_stmt(pool, rng))
    if rng.random() < 0.5:
        fname, arg = pool.fresh(), pool.fresh()
        helper = ast.parse(f"def {fname}({arg}):\n    return {arg} * 2").body[0]
        tree.body.insert(rng.randint(0, len(tree.body)), helper)
    return True


def _contains_own_continue(stmts: Sequence[ast.stmt]) -> bool:
    """Whether ``continue`` appears at this loop level (ignoring nested loops/functions)."""
    stack: list[ast.AST] = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Continue):
            return True
        if isinstance(
            node, ast.For | ast.While | ast.AsyncFor | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _py_loop_swap(tree: ast.Module, rng: random.Random) -> bool:
    pool = _NamePool(_all_names(tree), rng)
    changed = False
    for stmts in _bodies(tree):
        k = 0
        while k < len(stmts):
            node = stmts[k]
            replacement = _for_to_while(node, stmts[k + 1 :], pool)
            if replacement is not None and rng.random() < 0.8:
                stmts[k : k + 1] = replacement
                k += len(replacement)
                changed = True
            else:
                k += 1
    return changed


def _for_to_while(
    node: ast.stmt, following: Sequence[ast.stmt], pool: _NamePool
) -> list[ast.stmt] | None:
    if not isinstance(node, ast.For) or node.orelse or not isinstance(node.target, ast.Name):
        return None
    call = node.iter
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and not call.keywords
        and 1 <= len(call.args) <= 3
    ):
        return None
    var = node.target.id
    if _contains_own_continue(node.body):
        return None
    assigned_in_body = {
        n.id
        for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store | ast.Del)
    }
    if var in assigned_in_body:
        return None
    if any(var in _names_in(stmt) for stmt in following):
        return None
    args = call.args
    if len(args) == 1:
        start: ast.expr = ast.Constant(0)
        stop, step = args[0], 1
    else:
        start, stop = args[0], args[1]
        step = 1
        if len(args) == 3:
            if not (isinstance(args[2], ast.Constant) and isinstance(args[2].value, int)):
                return None
            step = args[2].value
            if step <= 0:
                return None
    pre: list[ast.stmt] = []
    if not isinstance(stop, ast.Constant):
        bound = pool.fresh()
        pre.append(ast.Assign(targets=[ast.Name(bound, ast.Store())], value=stop, lineno=0))
        stop = ast.Name(bound, ast.Load())
    pre.append(ast.Assign(targets=[ast.Name(var, ast.Store())], value=start, lineno=0))
    increment = ast.AugAssign(
        target=ast.Name(var, ast.Store()), op=ast.Add(), value=ast.Constant(step)
    )
    loop = ast.While(
        test=ast.Compare(left=ast.Name(var, ast.Load()), ops=[ast.Lt()], comparators=[stop]),
        body=[*node.body, increment],
        orelse=[],
    )
    return [*pre, loop]


_FLIP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
}
_SAFE_AUG = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


class _ExprRewriter(ast.NodeTransformer):
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.changed = False

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        self.generic_visit(node)
        numeric_add = isinstance(node.op, ast.Add) and isinstance(node.value, ast.Constant)
        if (
            isinstance(node.target, ast.Name)
            and (isinstance(node.op, _SAFE_AUG) or numeric_add)
            and self.rng.random() < 0.8
        ):
            self.changed = True
            return ast.Assign(
                targets=[ast.Name(node.target.id, ast.Store())],
                value=ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, node.value),
                lineno=0,
            )
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        if (
            len(node.ops) == 1
            and type(node.ops[0]) in _FLIP
            and not _has_call(node)
            and self.rng.random() < 0.6
        ):
            self.changed = True
            return ast.Compare(
                left=node.comparators[0], ops=[_FLIP[type(node.ops[0])]()], comparators=[node.left]
            )
        return node

    def visit_If(self, node: ast.If) -> ast.stmt:
        self.generic_visit(node)
        if node.orelse and self.rng.random() < 0.5:
            self.changed = True
            return ast.If(
                test=ast.UnaryOp(ast.Not(), node.test), body=node.orelse, orelse=node.body
            )
        return node


def _py_expr_rewrite(tree: ast.Module, rng: random.Random) -> bool:
    rewriter = _ExprRewriter(rng)
    rewriter.visit(tree)
    return rewriter.changed


def _string_lines(code: str) -> set[int]:
    """1-based line numbers that are inside (or continue) a multi-line token."""
    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.start[0] != tok.end[0]:
                lines.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set(range(1, code.count("\n") + 2))
    return lines


def _comments_text(code: str, rng: random.Random, prefix: str = "#") -> str:
    protected = _string_lines(code) if prefix == "#" else set()
    out: list[str] = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if lineno not in protected and stripped and rng.random() < 0.25:
            out.append(f"{indent}{prefix} {rng.choice(_COMMENTS)}")
        if (
            lineno not in protected
            and stripped
            and not stripped.endswith("\\")
            and (lineno + 1) not in protected
            and rng.random() < 0.15
        ):
            line = f"{line}  {prefix} {rng.choice(_COMMENTS)}"
        out.append(line)
        if lineno not in protected and (lineno + 1) not in protected and rng.random() < 0.1:
            out.append("")
    return "\n".join(out) + "\n"


_AST_TRANSFORMS: dict[str, Callable[[ast.Module, random.Random], bool]] = {
    "rename": _py_rename,
    "reorder": _py_reorder,
    "dead_code": _py_dead_code,
    "loop_swap": _py_loop_swap,
    "expr_rewrite": _py_expr_rewrite,
}


def transform_python(code: str, names: Sequence[str], rng: random.Random) -> TransformResult:
    """Apply Python transformations in order.

    Args:
        code: Valid Python 3 source.
        names: Transformation names from :data:`PYTHON_TRANSFORMS`.
        rng: Seeded random generator (the only source of randomness).

    Returns:
        The transformed code and the names of the transformations that changed it.

    Raises:
        SyntaxError: If ``code`` is not valid Python 3.
        ValueError: If a transformation name is unknown.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code)
    applied: list[str] = []
    want_comments = False
    for name in names:
        if name == "comments":
            want_comments = True
            continue
        if name not in _AST_TRANSFORMS:
            raise ValueError(f"Unknown Python transformation {name!r}")
        if _AST_TRANSFORMS[name](tree, rng):
            applied.append(name)
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + "\n"
    if want_comments:
        result = _comments_text(result, rng, "#")
        applied.append("comments")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        ast.parse(result)  # Transformations must keep the code valid.
    return TransformResult(code=result, applied=applied)


# ---------------------------------------------------------------------------- java / js


def _ts_nodes(root: Node) -> list[Node]:
    out: list[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


def _java_rename(code: str, language: str, rng: random.Random) -> tuple[str, bool]:
    parsed = parse_source(code, language)
    source = parsed.source
    nodes = _ts_nodes(parsed.root)

    def text(n: Node) -> str:
        return source[n.start_byte : n.end_byte].decode("utf-8", errors="replace")

    declared: set[str] = set()
    for node in nodes:
        name_node = None
        if node.type in ("variable_declarator", "formal_parameter", "catch_formal_parameter"):
            name_node = node.child_by_field_name("name")
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
        if name_node is not None and name_node.type == "identifier":
            declared.add(text(name_node))
    declared -= {"main", "args", "length"}
    if not declared:
        return code, False
    taken = {text(n) for n in nodes if n.type == "identifier"}
    pool = _NamePool(taken, rng)
    mapping = {name: pool.fresh() for name in sorted(declared)}

    edits: list[tuple[int, int, str]] = []
    for node in nodes:
        if node.type != "identifier" or text(node) not in mapping:
            continue
        parent = node.parent
        if (
            parent is not None
            and parent.type == "field_access"
            and parent.child_by_field_name("field") == node
        ):
            obj = parent.child_by_field_name("object")
            if obj is None or obj.type != "this":
                continue
        edits.append((node.start_byte, node.end_byte, mapping[text(node)]))
    for start, end, new in sorted(edits, reverse=True):
        source = source[:start] + new.encode() + source[end:]
    return source.decode("utf-8", errors="replace"), bool(edits)


def _java_dead_code(code: str, language: str, rng: random.Random) -> tuple[str, bool]:
    parsed = parse_source(code, language)
    source = parsed.source
    targets = ("method_declaration", "constructor_declaration", "function_declaration")
    bodies: list[Node] = []
    for n in _ts_nodes(parsed.root):
        body = n.child_by_field_name("body") if n.type in targets else None
        if body is not None and body.child_count > 0:
            bodies.append(body)
    if not bodies:
        return code, False
    pool = _NamePool(set(), rng)
    keyword = "int" if language == "java" else "let"
    inserts = []
    for body in rng.sample(bodies, k=min(len(bodies), rng.randint(1, 2))):
        stmt = f" {keyword} {pool.fresh()} = {rng.randint(0, 1000)};"
        inserts.append((body.start_byte + 1, stmt))
    for pos, stmt in sorted(inserts, reverse=True):
        source = source[:pos] + stmt.encode() + source[pos:]
    return source.decode("utf-8", errors="replace"), True


def transform_generic(
    code: str, language: str, names: Sequence[str], rng: random.Random
) -> TransformResult:
    """Apply the text-level transformations available for Java and JavaScript.

    Args:
        code: Source code.
        language: ``java`` or ``javascript``.
        names: Transformation names from :data:`JAVA_TRANSFORMS`.
        rng: Seeded random generator.

    Returns:
        The transformed code and the names of the transformations that changed it.

    Raises:
        ValueError: If a transformation name is unknown.
    """
    applied: list[str] = []
    for name in names:
        if name == "rename":
            code, changed = _java_rename(code, language, rng)
        elif name == "dead_code":
            code, changed = _java_dead_code(code, language, rng)
        elif name == "comments":
            code, changed = _comments_text(code, rng, "//"), True
        else:
            raise ValueError(f"Unknown {language} transformation {name!r}")
        if changed:
            applied.append(name)
    return TransformResult(code=code, applied=applied)


def available_transforms(language: str) -> tuple[str, ...]:
    """Transformation names supported for ``language``."""
    return PYTHON_TRANSFORMS if language == "python" else JAVA_TRANSFORMS


def apply_transforms(
    code: str, language: str, names: Sequence[str], rng: random.Random
) -> TransformResult:
    """Dispatch to :func:`transform_python` or :func:`transform_generic`."""
    if language == "python":
        return transform_python(code, names, rng)
    return transform_generic(code, language, names, rng)
