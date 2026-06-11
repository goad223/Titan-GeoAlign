"""Tests for the titan_info and titan_inspect CLI scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.titan_info import format_table_text, main as info_main
from scripts.titan_inspect import inspect, main as inspect_main


class TestTitanInfo:
    """titan_info CLI behavior."""

    def test_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert info_main([]) == 0
        out = capsys.readouterr().out
        assert "Hardware Report" in out
        assert "تقرير العتاد" in out
        assert "Supported Formats" in out
        assert "PNG/JPEG/BMP" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert info_main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "hardware" in payload
        assert "formats" in payload
        assert payload["hardware"]["cpu_count_logical"] >= 1
        assert any(f["format"] == "PNG/JPEG/BMP" for f in payload["formats"])

    def test_format_table_text_alignment(self) -> None:
        rows = [
            {"format": "X", "extensions": ".x", "available": True, "requires": "pkg"},
            {"format": "Y", "extensions": ".y", "available": False, "requires": "other"},
        ]
        text = format_table_text(rows)
        assert "yes | نعم" in text
        assert "no | لا" in text


class TestTitanInspect:
    """titan_inspect CLI behavior."""

    def test_inspect_png(self, png_file: Path) -> None:
        result = inspect(png_file)
        meta = result["metadata"]
        assert isinstance(meta, dict)
        assert meta["band_count"] == 3
        profile = result["sensor_profile"]
        assert isinstance(profile, dict)
        assert profile["name"] == "generic"

    def test_main_text_output(self, png_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert inspect_main([str(png_file)]) == 0
        out = capsys.readouterr().out
        assert "الملف" in out
        assert "Detected profile" in out

    def test_main_json_output(self, png_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert inspect_main([str(png_file), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["metadata"]["width"] > 0
        assert "summary" not in payload

    def test_main_missing_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert inspect_main(["/no/such/file.png"]) == 1
        err = capsys.readouterr().err
        assert "Error" in err
        assert "خطأ" in err

    def test_inspect_detects_sentinel2_from_safe(self, safe_product: Path) -> None:
        pytest.importorskip("rasterio")
        result = inspect(safe_product)
        profile = result["sensor_profile"]
        assert isinstance(profile, dict)
        assert profile["name"] == "sentinel2"
