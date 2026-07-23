"""Structured JSON logging via structlog."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(level: str = "INFO", json_file: str | None = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=lvl,
    )
    if json_file:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(json_file, encoding="utf-8")
        file_h.setLevel(lvl)
        file_h.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_h)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
