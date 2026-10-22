"""Centralized logging configuration.

Uses ``rich`` for pretty console output when available, falling back to the
standard library formatter otherwise. Call :func:`get_logger` everywhere instead
of configuring logging ad-hoc.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_DEFAULT_LEVEL = os.environ.get("FP_LOG_LEVEL", "INFO").upper()


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
        datefmt = "[%X]"
    except Exception:  # pragma: no cover - rich is optional
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=_DEFAULT_LEVEL,
        format=fmt,
        datefmt=datefmt,
        handlers=[handler],
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name; typically ``__name__`` of the calling module.
    """
    _configure_root()
    return logging.getLogger(name if name else "football_predictor")


def set_level(level: str) -> None:
    """Set the level on the root football_predictor logger and handlers."""
    _configure_root()
    logging.getLogger().setLevel(level.upper())
