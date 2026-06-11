"""Tests for the rotating UTF-8 logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from changemaster.core.logging_setup import (
    BACKUP_COUNT,
    LOG_FILE_NAME,
    MAX_BYTES,
    get_log_dir,
    get_logger,
    setup_logging,
)


def test_setup_creates_log_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = setup_logging(log_dir=log_dir, console=False)
    logger.info("hello world")
    assert (log_dir / LOG_FILE_NAME).exists()
    content = (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "hello world" in content


def test_arabic_messages_survive_utf8(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = setup_logging(log_dir=log_dir, console=False)
    logger.warning("تحذير: مساحة القرص منخفضة")
    content = (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "تحذير: مساحة القرص منخفضة" in content


def test_setup_is_idempotent(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir, console=True)
    logger = setup_logging(log_dir=log_dir, console=True)
    assert len(logger.handlers) == 2  # one file + one console, no duplicates


def test_rotation_configuration(tmp_path: Path) -> None:
    logger = setup_logging(log_dir=tmp_path / "logs", console=False)
    handler = logger.handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == MAX_BYTES == 5 * 1024 * 1024
    assert handler.backupCount == BACKUP_COUNT == 3


def test_level_accepts_string(tmp_path: Path) -> None:
    logger = setup_logging(level="debug", log_dir=tmp_path / "logs", console=False)
    assert logger.level == logging.DEBUG


def test_console_handler_handles_unencodable(tmp_path: Path, capsys) -> None:
    logger = setup_logging(log_dir=tmp_path / "logs", console=True)
    logger.error("خطأ فادح")  # must not raise even on narrow consoles
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_get_logger_children() -> None:
    root = get_logger()
    child = get_logger("io_engine")
    assert root.name == "changemaster"
    assert child.name == "changemaster.io_engine"


def test_get_log_dir_under_config_dir(isolated_config_dir: Path) -> None:
    assert get_log_dir() == isolated_config_dir / "logs"
