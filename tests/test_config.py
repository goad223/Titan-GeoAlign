"""Tests for JSON-persistent application configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changemaster.core.config import AppConfig, ConfigManager, get_config_dir
from changemaster.core.exceptions import ConfigError


def test_get_config_dir_honors_env_override(isolated_config_dir: Path) -> None:
    assert get_config_dir() == isolated_config_dir


def test_get_config_dir_uses_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CHANGEMASTER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    assert get_config_dir() == tmp_path / "AppData" / "ChangeMaster"


def test_get_config_dir_fallback_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHANGEMASTER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    assert get_config_dir() == Path.home() / ".config" / "ChangeMaster"


def test_load_returns_defaults_when_missing() -> None:
    manager = ConfigManager()
    cfg = manager.load()
    assert cfg == AppConfig()
    assert cfg.language == "en"
    assert cfg.log_level == "INFO"


def test_save_and_reload_roundtrip() -> None:
    manager = ConfigManager()
    cfg = manager.config
    cfg.language = "ar"
    cfg.tile_size = 2048
    path = manager.save()
    assert path.exists()
    fresh = ConfigManager().load()
    assert fresh.language == "ar"
    assert fresh.tile_size == 2048


def test_saved_file_is_utf8_json() -> None:
    manager = ConfigManager()
    manager.update(last_open_dir="مجلد_الصور")
    raw = manager.path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["last_open_dir"] == "مجلد_الصور"


def test_update_rejects_unknown_key() -> None:
    manager = ConfigManager()
    with pytest.raises(ConfigError):
        manager.update(nonexistent_setting=1)


def test_update_persists_immediately() -> None:
    manager = ConfigManager()
    manager.update(theme="light")
    assert ConfigManager().load().theme == "light"


def test_load_invalid_json_raises() -> None:
    manager = ConfigManager()
    manager.path.parent.mkdir(parents=True, exist_ok=True)
    manager.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        ConfigManager().load()


def test_load_non_object_json_raises() -> None:
    manager = ConfigManager()
    manager.path.parent.mkdir(parents=True, exist_ok=True)
    manager.path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError):
        ConfigManager().load()


def test_from_dict_ignores_unknown_keys() -> None:
    cfg = AppConfig.from_dict({"language": "ar", "obsolete_key": True})
    assert cfg.language == "ar"


def test_recent_files_dedup_and_cap() -> None:
    cfg = AppConfig()
    for i in range(15):
        cfg.add_recent_file(f"/data/img{i}.tif")
    cfg.add_recent_file("/data/img14.tif")
    assert len(cfg.recent_files) == AppConfig.MAX_RECENT
    assert cfg.recent_files[0] == "/data/img14.tif"
    assert cfg.recent_files.count("/data/img14.tif") == 1


def test_reset_restores_defaults() -> None:
    manager = ConfigManager()
    manager.update(language="ar", tile_size=4096)
    cfg = manager.reset()
    assert cfg == AppConfig()
    assert ConfigManager().load() == AppConfig()
