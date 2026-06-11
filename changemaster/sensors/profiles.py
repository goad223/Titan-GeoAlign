"""Sensor profiles: dataclass model, JSON-backed registry and auto-detection.

Each supported satellite/sensor is described by a JSON file in
``changemaster/sensors/definitions``. Profiles declare bands (with center
wavelengths and common aliases like ``red``/``nir``) and filename regex
patterns used to auto-detect the sensor from a product's file name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from changemaster.core.exceptions import SensorProfileError
from changemaster.io_engine.metadata import ImageMetadata

DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"


@dataclass(frozen=True)
class BandDefinition:
    """A single spectral band of a sensor.

    Attributes:
        name: Native band identifier (e.g. ``B04``).
        wavelength_nm: Center wavelength in nanometres (0 when N/A, e.g. SAR).
        alias: Common semantic alias (``red``, ``nir``, ``swir1``...).
        resolution_m: Ground sample distance in metres.
    """

    name: str
    wavelength_nm: float = 0.0
    alias: str = ""
    resolution_m: float = 0.0


@dataclass(frozen=True)
class SensorProfile:
    """Full description of a satellite sensor.

    Attributes:
        name: Unique profile key (e.g. ``sentinel2``).
        display_name: Human-readable name shown in the UI.
        platform: Mission/platform family (e.g. ``Sentinel-2``).
        sensor_type: ``optical`` or ``sar``.
        resolution_m: Best native resolution in metres.
        bands: Spectral band definitions.
        filename_patterns: Regex patterns that match product file names.
        description_en: English description.
        description_ar: Arabic description.
    """

    name: str
    display_name: str
    platform: str
    sensor_type: str = "optical"
    resolution_m: float = 0.0
    bands: tuple[BandDefinition, ...] = field(default_factory=tuple)
    filename_patterns: tuple[str, ...] = field(default_factory=tuple)
    description_en: str = ""
    description_ar: str = ""

    def band_by_alias(self, alias: str) -> BandDefinition | None:
        """Return the band matching a semantic alias, or ``None``.

        Args:
            alias: Alias such as ``red``, ``green``, ``nir``.
        """
        wanted = alias.lower()
        for band in self.bands:
            if band.alias.lower() == wanted:
                return band
        return None

    def band_index(self, name: str) -> int | None:
        """Return the 1-based index of a band by native name, or ``None``."""
        wanted = name.upper()
        for i, band in enumerate(self.bands, start=1):
            if band.name.upper() == wanted:
                return i
        return None

    def matches_filename(self, filename: str) -> bool:
        """Whether any of this profile's regex patterns match ``filename``."""
        return any(re.search(p, filename, re.IGNORECASE) for p in self.filename_patterns)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorProfile":
        """Build a profile from parsed JSON, validating required keys.

        Raises:
            SensorProfileError: If required keys are missing.
        """
        for key in ("name", "display_name", "platform"):
            if key not in data:
                raise SensorProfileError(
                    f"Sensor profile missing required key '{key}': {data}",
                    f"بروفايل المستشعر يفتقد المفتاح المطلوب '{key}': {data}",
                )
        bands = tuple(
            BandDefinition(
                name=str(b["name"]),
                wavelength_nm=float(b.get("wavelength_nm", 0.0)),
                alias=str(b.get("alias", "")),
                resolution_m=float(b.get("resolution_m", 0.0)),
            )
            for b in data.get("bands", [])
        )
        return cls(
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            platform=str(data["platform"]),
            sensor_type=str(data.get("sensor_type", "optical")),
            resolution_m=float(data.get("resolution_m", 0.0)),
            bands=bands,
            filename_patterns=tuple(str(p) for p in data.get("filename_patterns", [])),
            description_en=str(data.get("description_en", "")),
            description_ar=str(data.get("description_ar", "")),
        )


class SensorRegistry:
    """Loads, caches and queries every JSON sensor profile."""

    _profiles: dict[str, SensorProfile] | None = None

    @classmethod
    def _load(cls) -> dict[str, SensorProfile]:
        """Load all JSON definitions once and cache them.

        Raises:
            SensorProfileError: If a definition file contains invalid JSON.
        """
        if cls._profiles is not None:
            return cls._profiles
        profiles: dict[str, SensorProfile] = {}
        for json_path in sorted(DEFINITIONS_DIR.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SensorProfileError(
                    f"Cannot load sensor profile {json_path}: {exc}",
                    f"تعذر تحميل بروفايل المستشعر {json_path}: {exc}",
                ) from exc
            profile = SensorProfile.from_dict(data)
            profiles[profile.name] = profile
        cls._profiles = profiles
        return profiles

    @classmethod
    def reload(cls) -> None:
        """Drop the cache so definitions are re-read on next access."""
        cls._profiles = None

    @classmethod
    def all_profiles(cls) -> list[SensorProfile]:
        """Every registered profile, sorted by name."""
        return [cls._load()[k] for k in sorted(cls._load())]

    @classmethod
    def get(cls, name: str) -> SensorProfile:
        """Look up a profile by its unique key.

        Raises:
            SensorProfileError: If no profile with that name exists.
        """
        profiles = cls._load()
        key = name.lower()
        if key not in profiles:
            raise SensorProfileError(
                f"Unknown sensor profile '{name}'. Available: {sorted(profiles)}",
                f"بروفايل المستشعر '{name}' غير معروف. المتاح: {sorted(profiles)}",
            )
        return profiles[key]

    @classmethod
    def detect_from_filename(cls, filename: str | Path) -> SensorProfile:
        """Auto-detect the sensor from a file or product name.

        Args:
            filename: File name, directory name or full path.

        Returns:
            The first matching profile; the ``generic`` profile when no
            specific pattern matches.
        """
        name = Path(filename).name
        generic: SensorProfile | None = None
        for profile in cls.all_profiles():
            if profile.name == "generic":
                generic = profile
                continue
            if profile.matches_filename(name):
                return profile
        if generic is None:
            raise SensorProfileError(
                "Generic sensor profile is missing from definitions.",
                "بروفايل المستشعر العام مفقود من التعريفات.",
            )
        return generic

    @classmethod
    def detect_from_metadata(cls, metadata: ImageMetadata) -> SensorProfile:
        """Auto-detect the sensor from unified image metadata.

        Tries the explicit ``sensor`` field first (matching platform and
        profile names), then falls back to filename detection.

        Args:
            metadata: Metadata produced by any reader.

        Returns:
            The best matching profile (``generic`` as a last resort).
        """
        if metadata.sensor:
            sensor_text = metadata.sensor.lower().replace("_", "").replace("-", "")
            for profile in cls.all_profiles():
                if profile.name == "generic":
                    continue
                keys = {
                    profile.name.replace("_", ""),
                    profile.platform.lower().replace("-", "").replace(" ", ""),
                }
                if any(k and k in sensor_text for k in keys):
                    return profile
        return cls.detect_from_filename(metadata.path)
