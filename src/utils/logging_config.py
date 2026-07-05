"""Shared logging configuration.

Every module should call `get_logger(__name__)` instead of using
`print()`, so log verbosity/format is controlled in one place and
output can be redirected (e.g. to a file) without touching business
logic.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Usually `__name__` of the calling module.
        level: Logging level, defaults to INFO.

    Returns:
        A `logging.Logger` with a single stdout handler attached.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
