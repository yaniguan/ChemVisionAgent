"""Structured logging for ChemVisionAgent.

Replaces all bare print() calls with structured logging.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with consistent format.

    First call configures the root chemvision logger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger("chemvision")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True

    return logging.getLogger(f"chemvision.{name}")
