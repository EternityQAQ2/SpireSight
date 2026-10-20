from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path


LOGGER_NAME = "sts2_advisor"


def _base_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(directory: Path, retention_days: int = 3) -> Path:
    """Create a per-run UTF-8 log and remove logs older than the retention window."""
    if retention_days < 1:
        raise ValueError("log-retention-days 必须至少为 1")
    directory.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in directory.glob("advisor-*.log"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if modified < cutoff:
                path.unlink()
        except OSError:
            # Logging must never prevent the advisor from starting.
            pass

    path = directory / datetime.now().strftime("advisor-%Y%m%d-%H%M%S.log")
    logger = _base_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.info("logging_started retention_days=%s", retention_days)
    return path


def close_logging() -> None:
    logger = _base_logger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    _base_logger()
    suffix = f".{name}" if name else ""
    return logging.getLogger(LOGGER_NAME + suffix)
