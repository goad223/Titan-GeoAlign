"""Persistent application configuration stored as JSON.

On Windows the config lives in ``%APPDATA%/ChangeMaster/config.json``; on
Linux/macOS it falls back to ``~/.config/ChangeMaster/config.json``. All path
handling uses :mod:`pathlib` so the same code runs everywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, ClassVar

from changemaster.core.exceptions import ConfigError

APP_DIR_NAME = "ChangeMaster"
CONFIG_FILE_NAME = "config.json"


def get_config_dir() -> Path:
    """Return the per-user configuration directory (created on demand).

    Uses ``%APPDATA%`` on Windows and ``~/.config`` elsewhere. The
    ``CHANGEMASTER_CONFIG_DIR`` environment variable overrides both, which
    is also how tests isolate themselves from the real user profile.
    """
    override = os.environ.get("CHANGEMASTER_CONFIG_DIR")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_DIR_NAME
    return Path.home() / ".config" / APP_DIR_NAME


@dataclass
class AppConfig:
    """User-tunable application settings with safe defaults.

    Attributes:
        language: UI language code, ``"en"`` or ``"ar"``.
        theme: UI theme name (``"dark"`` or ``"light"``).
        max_workers: Parallel worker count; 0 means auto from hardware.
        tile_size: Tile edge in pixels; 0 means auto from hardware.
        cache_dir: Directory for temporary caches ("" means default temp).
        last_open_dir: Last directory used in file dialogs.
        log_level: Logging level name (e.g. ``"INFO"``).
        recent_files: Most recently opened files (newest first, max 10).
    """

    language: str = "en"
    theme: str = "dark"
    max_workers: int = 0
    tile_size: int = 0
    cache_dir: str = ""
    last_open_dir: str = ""
    log_level: str = "INFO"
    recent_files: list[str] = field(default_factory=list)

    MAX_RECENT: ClassVar[int] = 10

    def add_recent_file(self, path: str | Path) -> None:
        """Push ``path`` to the front of the recent-files list (deduplicated)."""
        text = str(path)
        if text in self.recent_files:
            self.recent_files.remove(text)
        self.recent_files.insert(0, text)
        del self.recent_files[self.MAX_RECENT:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all settings to a JSON-compatible dictionary."""
        data = asdict(self)
        data.pop("MAX_RECENT", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Parsed JSON dictionary (possibly from an older version).

        Returns:
            A config where known keys override defaults.
        """
        valid = {f.name for f in fields(cls) if f.name != "MAX_RECENT"}
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)


class ConfigManager:
    """Loads, validates and persists :class:`AppConfig` as JSON on disk."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Create a manager bound to a config directory.

        Args:
            config_dir: Custom directory; defaults to :func:`get_config_dir`.
        """
        self._dir: Path = config_dir or get_config_dir()
        self._path: Path = self._dir / CONFIG_FILE_NAME
        self._config: AppConfig | None = None

    @property
    def path(self) -> Path:
        """Full path of the JSON config file."""
        return self._path

    @property
    def config(self) -> AppConfig:
        """The active config; loads from disk on first access."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(self) -> AppConfig:
        """Load config from disk; return defaults when no file exists.

        Returns:
            The loaded (or default) configuration.

        Raises:
            ConfigError: If the file exists but contains invalid JSON.
        """
        if not self._path.exists():
            self._config = AppConfig()
            return self._config
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"Cannot read config file {self._path}: {exc}",
                f"تعذر قراءة ملف الإعدادات {self._path}: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Config file {self._path} must contain a JSON object.",
                f"ملف الإعدادات {self._path} يجب أن يحتوي على كائن JSON.",
            )
        self._config = AppConfig.from_dict(data)
        return self._config

    def save(self, config: AppConfig | None = None) -> Path:
        """Persist the config atomically (write temp file, then replace).

        Args:
            config: Config to save; defaults to the active config.

        Returns:
            The path the config was written to.

        Raises:
            ConfigError: When the file cannot be written.
        """
        cfg = config or self.config
        self._config = cfg
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:
            raise ConfigError(
                f"Cannot write config file {self._path}: {exc}",
                f"تعذر كتابة ملف الإعدادات {self._path}: {exc}",
            ) from exc
        return self._path

    def update(self, **changes: Any) -> AppConfig:
        """Apply keyword changes to the config and save immediately.

        Args:
            **changes: Attribute names and new values.

        Returns:
            The updated configuration.

        Raises:
            ConfigError: If an unknown setting name is supplied.
        """
        cfg = self.config
        valid = {f.name for f in fields(AppConfig) if f.name != "MAX_RECENT"}
        for key, value in changes.items():
            if key not in valid:
                raise ConfigError(
                    f"Unknown setting: {key}",
                    f"إعداد غير معروف: {key}",
                )
            setattr(cfg, key, value)
        self.save(cfg)
        return cfg

    def reset(self) -> AppConfig:
        """Restore defaults and persist them."""
        self._config = AppConfig()
        self.save(self._config)
        return self._config
