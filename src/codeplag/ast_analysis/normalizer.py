"""Normalization of tree-sitter trees into compact, rename-invariant trees.

The normalized tree keeps the *shape* of the program and drops what a plagiarist can change
cheaply. Each rule below is applied in a single pre-order pass:

1. **Comments are removed** (``comment``, ``line_comment``, ``block_comment``).
2. **Docstrings are removed** (Python): a string expression statement that is the first
   statement of a module, class or function body.
3. **Punctuation and keyword tokens are removed.** Anonymous tokens such as ``(``, ``:``,
   ``def`` add nothing that the parent node type does not already encode. Operator tokens
   (``+``, ``<``, ``and``, ...) are kept because they carry meaning.
4. **Identifiers are canonicalized.** User-defined names get the label ``ID``; names that a
   plagiarist rarely renames (attribute/method names and well-known builtins or library
   names) keep their text as ``ID:<name>`` so unrelated programs stay distinguishable.
   Independently, every user identifier gets an alpha-renamed canonical name (``v1``,
   ``v2``, ...) assigned in order of first occurrence, resolved through a scope stack where
   functions, methods, classes and lambdas open a new scope. Canonical names appear in the
   canonical token stream, not in tree labels, so tree comparison does not depend on
   declaration order.
5. **Literals are canonicalized**: strings become ``STR``, numbers ``NUM``, booleans
   ``BOOL`` and null-like values ``NULL``. String internals are dropped.
6. **Equivalent constructs are canonicalized** (optional, ``canonicalize_constructs``):
   every loop form (``for``, ``while``, ``do``, enhanced ``for``) becomes ``LOOP``; plain,
   augmented and update assignments become ``ASSIGN``, with ``x op= y`` expanded to
   ``x = x op y`` and ``x++`` to ``x = x + 1``; binary and boolean operations become
   ``BINOP``; parentheses are removed and their contents hoisted into the parent.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field

from tree_sitter import Node

from codeplag.ast_analysis.parser import ParsedSource

COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment", "html_comment"})

IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
        "field_identifier",
    }
)

STRING_TYPES = frozenset(
    {
        "string",
        "concatenated_string",
        "string_literal",
        "character_literal",
        "text_block",
        "template_string",
        "regex",
    }
)
NUMBER_TYPES = frozenset(
    {
        "integer",
        "float",
        "number",
        "decimal_integer_literal",
        "hex_integer_literal",
        "octal_integer_literal",
        "binary_integer_literal",
        "decimal_floating_point_literal",
        "hex_floating_point_literal",
    }
)
BOOL_TYPES = frozenset({"true", "false"})
NULL_TYPES = frozenset({"none", "null_literal", "null", "undefined"})

LOOP_TYPES = frozenset(
    {
        "for_statement",
        "while_statement",
        "do_statement",
        "enhanced_for_statement",
        "for_in_statement",
    }
)
ASSIGN_TYPES = frozenset(
    {
        "assignment",
        "augmented_assignment",
        "assignment_expression",
        "augmented_assignment_expression",
        "update_expression",
    }
)
BINOP_TYPES = frozenset({"binary_operator", "binary_expression", "boolean_operator"})
COMPOUND_OPS = frozenset(
    {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=", ">>>="}
)
PAREN_TYPES = frozenset({"parenthesized_expression"})

SCOPE_TYPES = frozenset(
    {
        "function_definition",
        "class_definition",
        "lambda",
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
        "lambda_expression",
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function_declaration",
    }
)

OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "@",
        "<",
        ">",
        "<=",
        ">=",
        "==",
        "!=",
        "===",
        "!==",
        "<>",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "//=",
        "%=",
        "**=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        ">>>=",
        "&&=",
        "||=",
        "??=",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
        ">>>",
        "!",
        "&&",
        "||",
        "??",
        "++",
        "--",
        "and",
        "or",
        "not",
        "in",
        "is",
        "instanceof",
        "typeof",
        "new",
        "delete",
        "void",
        "return",
        "break",
        "continue",
        "yield",
        "await",
        "throw",
        "raise",
    }
)

# Names that plagiarists rarely rename: keeping them makes unrelated programs distinguishable.
_KEEP_NAMES = frozenset(dir(builtins)) | frozenset(
    {
        # Python stdlib modules and idioms common in competitive programming.
        "sys",
        "math",
        "collections",
        "itertools",
        "heapq",
        "bisect",
        "functools",
        "re",
        "stdin",
        "stdout",
        "readline",
        "deque",
        "defaultdict",
        "Counter",
        "self",
        "cls",
        # Java / JavaScript library names.
        "System",
        "out",
        "in",
        "println",
        "print",
        "printf",
        "Math",
        "String",
        "Integer",
        "Long",
        "Double",
        "Scanner",
        "BufferedReader",
        "InputStreamReader",
        "List",
        "ArrayList",
        "Map",
        "HashMap",
        "Set",
        "HashSet",
        "Arrays",
        "Collections",
        "main",
        "args",
        "console",
        "log",
        "require",
        "process",
        "parseInt",
        "length",
        "push",
    }
)


@dataclass
class NormNode:
    """A node of the normalized tree.

    Attributes:
        label: Canonical label (node type or canonical token).
        children: Ordered child nodes.
        start_line: First source line covered (1-based).
        end_line: Last source line covered (1-based).
    """

    label: str
    children: list[NormNode] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0

    def size(self) -> int:
        """Number of nodes in this subtree."""
        total = 0
        stack = [self]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children)
        return total

    def depth(self) -> int:
        """Height of the subtree (a leaf has depth 1)."""
        best = 0
        stack = [(self, 1)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            stack.extend((child, d + 1) for child in node.children)
        return best

    def preorder(self) -> list[NormNode]:
        """Nodes in pre-order."""
        out: list[NormNode] = []
        stack = [self]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out


@dataclass
class NormalizedSource:
    """Result of normalizing one parsed source.

    Attributes:
        tree: Root of the normalized tree.
        tokens: Canonical token stream (alpha-renamed identifiers, canonical literals).
        language: Source language.
        error_count: Syntax error nodes reported by the parser.
    """

    tree: NormNode
    tokens: list[str]
    language: str
    error_count: int = 0


class _Normalizer:
    def __init__(self, source: bytes, canonicalize_constructs: bool, max_depth: int) -> None:
        self.source = source
        self.canonicalize = canonicalize_constructs
        self.max_depth = max_depth
        self.scopes: list[dict[str, str]] = [{}]
        self.counter = 0
        self.tokens: list[str] = []

    def text(self, node: Node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def canonical_name(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        self.counter += 1
        canon = f"v{self.counter}"
        self.scopes[-1][name] = canon
        return canon

    @staticmethod
    def is_docstring(node: Node) -> bool:
        if node.type != "expression_statement" or node.named_child_count != 1:
            return False
        child = node.named_children[0]
        if child.type not in ("string", "concatenated_string"):
            return False
        parent = node.parent
        if parent is None:
            return False
        if parent.type == "module" or (
            parent.type == "block"
            and parent.parent is not None
            and parent.parent.type in ("function_definition", "class_definition")
        ):
            first = parent.named_children[0] if parent.named_child_count else None
            return first is not None and first.id == node.id
        return False

    @staticmethod
    def is_member_name(node: Node) -> bool:
        parent = node.parent
        if parent is None:
            return False
        if node.type in ("property_identifier", "field_identifier"):
            return True
        if parent.type == "attribute":
            return parent.child_by_field_name("attribute") == node
        if parent.type in ("field_access", "method_invocation"):
            return parent.child_by_field_name("field") == node or (
                parent.type == "method_invocation" and parent.child_by_field_name("name") == node
            )
        return False

    def identifier_label(self, node: Node) -> str:
        name = self.text(node)
        if self.is_member_name(node) or name in _KEEP_NAMES:
            self.tokens.append(name)
            return f"ID:{name}"
        self.tokens.append(self.canonical_name(name))
        return "ID"

    def visit(self, node: Node, depth: int) -> list[NormNode]:
        """Return the normalized nodes for ``node`` (zero, one, or hoisted children)."""
        ntype = node.type
        if ntype in COMMENT_TYPES or self.is_docstring(node):
            return []
        start, end = node.start_point[0] + 1, node.end_point[0] + 1

        if not node.is_named:
            if ntype in OPERATORS:
                self.tokens.append(ntype)
                return [NormNode(ntype, [], start, end)]
            return []

        if ntype in IDENTIFIER_TYPES:
            return [NormNode(self.identifier_label(node), [], start, end)]
        if ntype in STRING_TYPES:
            self.tokens.append("STR")
            return [NormNode("STR", [], start, end)]
        if ntype in NUMBER_TYPES:
            self.tokens.append("NUM")
            return [NormNode("NUM", [], start, end)]
        if ntype in BOOL_TYPES:
            self.tokens.append("BOOL")
            return [NormNode("BOOL", [], start, end)]
        if ntype in NULL_TYPES:
            self.tokens.append("NULL")
            return [NormNode("NULL", [], start, end)]
        if depth >= self.max_depth:
            return [NormNode("DEEP", [], start, end)]

        if self.canonicalize and ntype in PAREN_TYPES:
            hoisted: list[NormNode] = []
            for child in node.children:
                hoisted.extend(self.visit(child, depth + 1))
            return hoisted

        opens_scope = ntype in SCOPE_TYPES
        if opens_scope:
            self.scopes.append({})
        children: list[NormNode] = []
        for child in node.children:
            children.extend(self.visit(child, depth + 1))
        if opens_scope:
            self.scopes.pop()

        label = ntype
        if self.canonicalize:
            if ntype in LOOP_TYPES:
                label = "LOOP"
            elif ntype in BINOP_TYPES:
                label = "BINOP"
            elif ntype in ASSIGN_TYPES:
                label = "ASSIGN"
                children = _expand_compound(children, start, end)
        if not children and node.child_count == 0:
            # A named leaf that is not an identifier or literal (e.g. `pass`, `this`).
            self.tokens.append(label)
        return [NormNode(label, children, start, end)]


def _clone(node: NormNode) -> NormNode:
    return NormNode(node.label, [_clone(c) for c in node.children], node.start_line, node.end_line)


def _expand_compound(children: list[NormNode], start: int, end: int) -> list[NormNode]:
    """Rewrite ``x op= y`` as ``x = x op y`` and ``x++``/``++x`` as ``x = x + 1``."""
    ops = [i for i, c in enumerate(children) if c.label in COMPOUND_OPS or c.label in ("++", "--")]
    if len(ops) != 1:
        return children
    i = ops[0]
    op = children[i].label
    if op in ("++", "--"):
        targets = [c for j, c in enumerate(children) if j != i]
        if len(targets) != 1:
            return children
        target = targets[0]
        value = NormNode("NUM", [], start, end)
        binop = NormNode(
            "BINOP", [_clone(target), NormNode(op[0], [], start, end), value], start, end
        )
        return [target, NormNode("=", [], start, end), binop]
    left, right = children[:i], children[i + 1 :]
    if len(left) != 1 or not right:
        return children
    binop = NormNode(
        "BINOP", [_clone(left[0]), NormNode(op[:-1], [], start, end), *right], start, end
    )
    return [left[0], NormNode("=", [], start, end), binop]


def normalize(
    parsed: ParsedSource, *, canonicalize_constructs: bool = True, max_depth: int = 400
) -> NormalizedSource:
    """Normalize a parsed source into a :class:`NormalizedSource`.

    Args:
        parsed: Output of :func:`codeplag.ast_analysis.parser.parse_source`.
        canonicalize_constructs: Apply rule 6 (loops, assignments, parentheses).
        max_depth: Depth guard; deeper subtrees collapse into a ``DEEP`` leaf.

    Returns:
        The normalized tree and canonical token stream.
    """
    normalizer = _Normalizer(parsed.source, canonicalize_constructs, max_depth)
    nodes = normalizer.visit(parsed.root, 0)
    root = nodes[0] if len(nodes) == 1 else NormNode("module", nodes, 1, 1)
    return NormalizedSource(
        tree=root,
        tokens=normalizer.tokens,
        language=parsed.language,
        error_count=parsed.error_count,
    )
