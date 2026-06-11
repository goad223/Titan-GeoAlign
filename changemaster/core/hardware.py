"""Hardware detection and tier classification for ChangeMaster Ultimate.

Detects CPU, RAM, disk and GPU capabilities and classifies the machine into
a performance tier so the application can adapt its processing strategy
(tile sizes, worker counts, caching) to any hardware, fully offline.

``psutil`` is used when available but everything degrades gracefully to the
standard library, so this module never fails to import.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


def _try_import_psutil() -> Any | None:
    """Return the ``psutil`` module if installed, else ``None`` (lazy import)."""
    try:
        import psutil  # noqa: PLC0415

        return psutil
    except ImportError:
        return None


class HardwareTier(str, Enum):
    """Coarse machine performance classes used to tune processing defaults."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    WORKSTATION = "workstation"


@dataclass(frozen=True)
class GPUInfo:
    """Information about a detected GPU.

    Attributes:
        name: GPU model name.
        memory_mb: Dedicated memory in megabytes (0 if unknown).
        vendor: GPU vendor string (e.g. ``nvidia``).
    """

    name: str
    memory_mb: int = 0
    vendor: str = "unknown"


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of the machine's capabilities.

    Attributes:
        os_name: Operating system name (e.g. ``Windows``, ``Linux``).
        os_version: Operating system release string.
        cpu_count_physical: Number of physical CPU cores.
        cpu_count_logical: Number of logical CPU threads.
        total_ram_mb: Total system memory in megabytes.
        available_ram_mb: Currently available memory in megabytes.
        free_disk_mb: Free space on the working drive in megabytes.
        gpus: Detected GPUs (may be empty).
        tier: Classified performance tier.
    """

    os_name: str
    os_version: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_mb: int
    available_ram_mb: int
    free_disk_mb: int
    gpus: tuple[GPUInfo, ...] = field(default_factory=tuple)
    tier: HardwareTier = HardwareTier.LOW

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a plain dictionary (JSON-friendly)."""
        data = asdict(self)
        data["tier"] = self.tier.value
        data["gpus"] = [asdict(g) for g in self.gpus]
        return data

    @property
    def has_gpu(self) -> bool:
        """Whether at least one GPU was detected."""
        return len(self.gpus) > 0

    def recommended_workers(self) -> int:
        """Return a sensible parallel worker count for this machine."""
        return max(1, self.cpu_count_logical - 1)

    def recommended_tile_size(self) -> int:
        """Return a tile edge size (pixels) suited to available memory."""
        if self.tier is HardwareTier.WORKSTATION:
            return 4096
        if self.tier is HardwareTier.HIGH:
            return 2048
        if self.tier is HardwareTier.MEDIUM:
            return 1024
        return 512


def _detect_cpu() -> tuple[int, int]:
    """Detect physical and logical CPU counts with stdlib fallback."""
    psutil = _try_import_psutil()
    logical = os.cpu_count() or 1
    physical = logical
    if psutil is not None:
        physical = psutil.cpu_count(logical=False) or logical
        logical = psutil.cpu_count(logical=True) or logical
    return int(physical), int(logical)


def _detect_memory() -> tuple[int, int]:
    """Detect total and available RAM in MB with stdlib fallback."""
    psutil = _try_import_psutil()
    if psutil is not None:
        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page_size
        avail = os.sysconf("SC_AVPHYS_PAGES") * page_size
        return int(total // (1024 * 1024)), int(avail // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 4096, 2048


def _detect_disk(path: Path | None = None) -> int:
    """Detect free disk space in MB at ``path`` (default: home directory)."""
    target = path or Path.home()
    try:
        usage = shutil.disk_usage(str(target))
        return int(usage.free // (1024 * 1024))
    except OSError:
        return 0


def _detect_gpus() -> tuple[GPUInfo, ...]:
    """Detect NVIDIA GPUs through ``nvidia-smi`` if present (offline-safe)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return ()
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return ()
        gpus: list[GPUInfo] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            mem = 0
            if len(parts) > 1:
                try:
                    mem = int(float(parts[1]))
                except ValueError:
                    mem = 0
            gpus.append(GPUInfo(name=parts[0], memory_mb=mem, vendor="nvidia"))
        return tuple(gpus)
    except (OSError, subprocess.SubprocessError):
        return ()


def classify_tier(
    cpu_logical: int,
    total_ram_mb: int,
    has_gpu: bool,
) -> HardwareTier:
    """Classify a machine into a :class:`HardwareTier`.

    Args:
        cpu_logical: Logical CPU thread count.
        total_ram_mb: Total RAM in megabytes.
        has_gpu: Whether a discrete GPU was detected.

    Returns:
        The performance tier for the supplied capabilities.
    """
    if cpu_logical >= 16 and total_ram_mb >= 32_000 and has_gpu:
        return HardwareTier.WORKSTATION
    if cpu_logical >= 8 and total_ram_mb >= 16_000:
        return HardwareTier.HIGH
    if cpu_logical >= 4 and total_ram_mb >= 8_000:
        return HardwareTier.MEDIUM
    return HardwareTier.LOW


def detect_hardware(disk_path: Path | None = None) -> HardwareProfile:
    """Probe the current machine and return a :class:`HardwareProfile`.

    Args:
        disk_path: Directory whose drive's free space should be measured;
            defaults to the user's home directory.

    Returns:
        A fully populated, classified hardware profile. Never raises for
        missing optional tools; every probe has a safe fallback.
    """
    physical, logical = _detect_cpu()
    total_ram, avail_ram = _detect_memory()
    free_disk = _detect_disk(disk_path)
    gpus = _detect_gpus()
    tier = classify_tier(logical, total_ram, bool(gpus))
    return HardwareProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_count_physical=physical,
        cpu_count_logical=logical,
        total_ram_mb=total_ram,
        available_ram_mb=avail_ram,
        free_disk_mb=free_disk,
        gpus=gpus,
        tier=tier,
    )


def hardware_report(profile: HardwareProfile | None = None) -> str:
    """Render a human-readable bilingual hardware report.

    Args:
        profile: Optional pre-computed profile; detected fresh when ``None``.

    Returns:
        Multi-line report text (English labels with Arabic counterparts).
    """
    prof = profile or detect_hardware()
    lines = [
        "=== ChangeMaster Hardware Report | تقرير العتاد ===",
        f"OS | نظام التشغيل: {prof.os_name} {prof.os_version}",
        f"CPU cores (physical/logical) | أنوية المعالج: {prof.cpu_count_physical}/{prof.cpu_count_logical}",
        f"RAM total/available (MB) | الذاكرة: {prof.total_ram_mb}/{prof.available_ram_mb}",
        f"Free disk (MB) | المساحة الحرة: {prof.free_disk_mb}",
        f"Tier | التصنيف: {prof.tier.value}",
    ]
    if prof.gpus:
        for gpu in prof.gpus:
            lines.append(f"GPU | كرت الشاشة: {gpu.name} ({gpu.memory_mb} MB)")
    else:
        lines.append("GPU | كرت الشاشة: none detected | لم يتم الكشف")
    lines.append(f"Recommended workers | عدد العمليات الموصى به: {prof.recommended_workers()}")
    lines.append(f"Recommended tile size | حجم البلاطة الموصى به: {prof.recommended_tile_size()}")
    return "\n".join(lines)
