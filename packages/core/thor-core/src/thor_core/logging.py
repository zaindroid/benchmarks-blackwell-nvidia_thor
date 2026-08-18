"""Structured logging for ThorAI platform.

Wraps structlog; falls back to the stdlib logging module if structlog
is not installed so that core utilities remain importable anywhere.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:  # pragma: no cover - structlog is a hard dependency
    import structlog

    _STRUCTLOG_AVAILABLE = True
except Exception:  # pragma: no cover
    _STRUCTLOG_AVAILABLE = False

_DEFAULT_LEVEL = logging.INFO


def configure_logging(level: int | str = _DEFAULT_LEVEL, json: bool = False) -> None:
    """Configure global logging for the platform."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if _STRUCTLOG_AVAILABLE:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        if json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        )
    else:  # pragma: no cover
        logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")


def get_logger(name: str) -> Any:
    """Return a logger for the given module name."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)  # pragma: no cover


configure_logging()
