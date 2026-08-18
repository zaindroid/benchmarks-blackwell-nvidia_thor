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


def configure_logging(level: int | str = _DEFAULT_LEVEL, json: bool | None = None) -> None:
    """Configure global logging for the platform.

    On deployed environments (``APP_ENV`` set) logs go to stdout as
    JSON with the level from ``LOG_LEVEL``, matching the zorc platform
    contract (structured JSON logs to stdout, never files).
    """
    import os

    if json is None:
        json = bool(os.getenv("APP_ENV"))
    if level is _DEFAULT_LEVEL and os.getenv("LOG_LEVEL"):
        level = os.getenv("LOG_LEVEL", "INFO")
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
        stream = sys.stdout if os.getenv("APP_ENV") else sys.stderr
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(stream),
        )
    else:  # pragma: no cover
        logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")


def get_logger(name: str) -> Any:
    """Return a logger for the given module name."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)  # pragma: no cover


configure_logging()
