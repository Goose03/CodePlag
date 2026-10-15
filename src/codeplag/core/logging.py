"""Logging setup shared by the library and the CLI.

Library modules call :func:`get_logger` and never configure handlers themselves; the CLI
(or a script) calls :func:`setup_logging` once at startup.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT_LOGGER_NAME = "codeplag"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure the ``codeplag`` logger hierarchy.

    Args:
        level: Logging level name or number. Defaults to ``CODEPLAG_LOG_LEVEL`` or ``INFO``.
    """
    global _configured
    if level is None:
        level = os.environ.get("CODEPLAG_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``codeplag`` hierarchy.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` whose name starts with ``codeplag``.
    """
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
