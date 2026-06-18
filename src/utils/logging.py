"""Structured logging configuration."""

import logging
import sys
from pathlib import Path

from src.config import settings


def setup_logging(log_file: str | None = None, level: str | None = None) -> None:
    """
    Configure application logging.

    Args:
        log_file: Path to log file (default from settings)
        level: Log level (default from settings)
    """
    log_file = log_file or settings.logging.file
    level = level or settings.logging.level
    fmt = settings.logging.format

    # Create log directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module."""
    return logging.getLogger(name)
