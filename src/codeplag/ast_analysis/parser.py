"""Language detection and tree-sitter parsing.

tree-sitter is error tolerant: a file with syntax errors still yields a tree in which the
broken regions are ``ERROR`` or ``MISSING`` nodes. :func:`parse_source` records how much of
the tree is affected so callers can decide whether to trust it; ``strict=True`` turns any
syntax error into :class:`CodeParseError`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree

from codeplag.core.logging import get_logger

logger = get_logger(__name__)

_GRAMMARS: dict[str, Callable[[], object]] = {
    "python": tree_sitter_python.language,
    "java": tree_sitter_java.language,
    "javascript": tree_sitter_javascript.language,
}

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".java": "java",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(_GRAMMARS)


class UnsupportedLanguageError(ValueError):
    """Raised when a file extension or language name is not supported."""


class CodeParseError(ValueError):
    """Raised when source code cannot be parsed (strict mode or unreadable input)."""


@dataclass(frozen=True)
class ParsedSource:
    """A parsed source file.

    Attributes:
        tree: The tree-sitter syntax tree.
        source: The UTF-8 encoded source the tree refers to.
        language: Language name, one of :data:`SUPPORTED_LANGUAGES`.
        error_count: Number of ``ERROR``/``MISSING`` nodes in the tree.
    """

    tree: Tree
    source: bytes
    language: str
    error_count: int

    @property
    def root(self) -> Node:
        """Root node of the syntax tree."""
        return self.tree.root_node

    @property
    def has_errors(self) -> bool:
        """Whether tree-sitter had to recover from syntax errors."""
        return self.error_count > 0

    @property
    def is_empty(self) -> bool:
        """Whether the file contains no code at all."""
        return self.root.named_child_count == 0


def detect_language(path: str | Path) -> str:
    """Detect the language of a file from its extension.

    Args:
        path: File path. Only the suffix is inspected.

    Returns:
        A language name from :data:`SUPPORTED_LANGUAGES`.

    Raises:
        UnsupportedLanguageError: If the extension is unknown.
    """
    suffix = Path(path).suffix.lower()
    try:
        return EXTENSIONS[suffix]
    except KeyError:
        raise UnsupportedLanguageError(
            f"Unsupported file extension {suffix!r} for {path}; supported: "
            + ", ".join(sorted(EXTENSIONS))
        ) from None


@cache
def get_parser(language: str) -> Parser:
    """Return a cached tree-sitter parser for ``language``.

    Raises:
        UnsupportedLanguageError: If the language is not supported.
    """
    if language not in _GRAMMARS:
        raise UnsupportedLanguageError(
            f"Unsupported language {language!r}; supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return Parser(Language(_GRAMMARS[language]()))


def _count_errors(node: Node) -> int:
    if not node.has_error:
        return 0
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_error or current.is_missing:
            count += 1
        stack.extend(child for child in current.children if child.has_error or child.is_missing)
    return count


def parse_source(code: str | bytes, language: str, *, strict: bool = False) -> ParsedSource:
    """Parse source code with tree-sitter.

    Args:
        code: Source text (``str``) or UTF-8 bytes.
        language: Language name.
        strict: Raise instead of returning a partially recovered tree.

    Returns:
        The parsed source.

    Raises:
        UnsupportedLanguageError: If the language is not supported.
        CodeParseError: In strict mode, if the code contains syntax errors.
    """
    source = code.encode("utf-8", errors="replace") if isinstance(code, str) else code
    tree = get_parser(language).parse(source)
    errors = _count_errors(tree.root_node)
    if errors:
        message = f"{errors} syntax error node(s) while parsing {language} source"
        if strict:
            raise CodeParseError(message)
        logger.debug(message)
    return ParsedSource(tree=tree, source=source, language=language, error_count=errors)


def parse_file(
    path: str | Path, language: str | None = None, *, strict: bool = False
) -> ParsedSource:
    """Read and parse a file, detecting the language from its extension when not given.

    Args:
        path: File to parse.
        language: Override for the detected language.
        strict: Raise on syntax errors.

    Returns:
        The parsed source.

    Raises:
        CodeParseError: If the file cannot be read, or in strict mode on syntax errors.
        UnsupportedLanguageError: If the language cannot be determined.
    """
    file_path = Path(path)
    lang = language or detect_language(file_path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise CodeParseError(f"Cannot read {file_path}: {exc}") from exc
    return parse_source(data, lang, strict=strict)
