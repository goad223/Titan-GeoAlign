"""Tests for hardware detection and tier classification."""

from __future__ import annotations

from pathlib import Path

from changemaster.core.hardware import (
    GPUInfo,
    HardwareProfile,
    HardwareTier,
    classify_tier,
    detect_hardware,
    hardware_report,
)


def test_detect_hardware_returns_sane_profile() -> None:
    profile = detect_hardware()
    assert profile.cpu_count_logical >= 1
    assert profile.cpu_count_physical >= 1
    assert profile.total_ram_mb > 0
    assert profile.available_ram_mb > 0
    assert profile.free_disk_mb >= 0
    assert profile.os_name
    assert isinstance(profile.tier, HardwareTier)


def test_detect_hardware_custom_disk_path(tmp_path: Path) -> None:
    profile = detect_hardware(disk_path=tmp_path)
    assert profile.free_disk_mb > 0


def test_classify_tier_boundaries() -> None:
    assert classify_tier(2, 4_000, False) is HardwareTier.LOW
    assert classify_tier(4, 8_000, False) is HardwareTier.MEDIUM
    assert classify_tier(8, 16_000, False) is HardwareTier.HIGH
    assert classify_tier(16, 32_000, True) is HardwareTier.WORKSTATION
    assert classify_tier(16, 32_000, False) is HardwareTier.HIGH
    assert classify_tier(32, 4_000, True) is HardwareTier.LOW


def _profile(tier: HardwareTier, gpus: tuple[GPUInfo, ...] = ()) -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_version="6",
        cpu_count_physical=4,
        cpu_count_logical=8,
        total_ram_mb=16_000,
        available_ram_mb=8_000,
        free_disk_mb=100_000,
        gpus=gpus,
        tier=tier,
    )


def test_recommended_tile_size_per_tier() -> None:
    assert _profile(HardwareTier.LOW).recommended_tile_size() == 512
    assert _profile(HardwareTier.MEDIUM).recommended_tile_size() == 1024
    assert _profile(HardwareTier.HIGH).recommended_tile_size() == 2048
    assert _profile(HardwareTier.WORKSTATION).recommended_tile_size() == 4096


def test_recommended_workers_leaves_one_core() -> None:
    assert _profile(HardwareTier.HIGH).recommended_workers() == 7


def test_profile_to_dict_is_json_friendly() -> None:
    gpu = GPUInfo(name="Test GPU", memory_mb=8192, vendor="nvidia")
    profile = _profile(HardwareTier.WORKSTATION, gpus=(gpu,))
    data = profile.to_dict()
    assert data["tier"] == "workstation"
    assert data["gpus"][0]["name"] == "Test GPU"
    assert profile.has_gpu


def test_hardware_report_is_bilingual() -> None:
    report = hardware_report()
    assert "Hardware Report" in report
    assert "تقرير العتاد" in report
    assert "CPU" in report


def test_hardware_report_with_gpu_profile() -> None:
    gpu = GPUInfo(name="RTX", memory_mb=4096, vendor="nvidia")
    report = hardware_report(_profile(HardwareTier.WORKSTATION, gpus=(gpu,)))
    assert "RTX" in report
