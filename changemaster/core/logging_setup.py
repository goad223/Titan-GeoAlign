"""Logging setup: rotating UTF-8 file logs (5 MB x 3) plus console output.

The file handler always writes UTF-8 so Arabic log messages are preserved on
Windows. The console handler degrades gracefully when the terminal cannot
render non-ASCII characters.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from changemaster.core.config import get_config_dir

LOGGER_NAME = "changemaster"
LOG_FILE_NAME = "changemaster.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _SafeConsoleHandler(logging.StreamHandler):
    """Console handler that never crashes on un-encodable characters.

    When the active console encoding cannot represent a character (common
    with Arabic text on legacy Windows code pages), the message is re-encoded
    with backslash escapes instead of raising ``UnicodeEncodeError``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the record, replacing un-encodable characters if needed."""
        try:
            msg = self.format(record)
            stream = self.stream
            encoding = getattr(stream, "encoding", None) or "utf-8"
            try:
                msg.encode(encoding)
            except UnicodeEncodeError:
                msg = msg.encode(encoding, errors="backslashreplace").decode(encoding)
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - mirror logging.Handler contract
            self.handleError(record)


def get_log_dir() -> Path:
    """Return the directory where rotating log files are stored."""
    return get_config_dir() / "logs"


def setup_logging(
    level: int | str = logging.INFO,
    log_dir: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return the application root logger.

    Idempotent: calling it again replaces existing ChangeMaster handlers
    instead of duplicating them.

    Args:
        level: Logging level (name or numeric).
        log_dir: Directory for the rotating log file; defaults to the
            per-user config directory's ``logs`` subfolder.
        console: Whether to also log to stderr.

    Returns:
        The configured ``changemaster`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)

    target_dir = log_dir or get_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        target_dir / LOG_FILE_NAME,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = _SafeConsoleHandler(stream=sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the ChangeMaster root logger.

    Args:
        name: Optional sub-logger name (e.g. ``"io_engine"``).

    Returns:
        ``changemaster`` logger or ``changemaster.<name>`` child logger.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
