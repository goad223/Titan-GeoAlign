"""Meta-learner: learns which matcher works best for each image pair type."""
from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Rule-based routing table: (source_sensor, target_sensor) → recommended matcher
_ROUTING_RULES: dict[tuple[str, str], str] = {
    ("SENTINEL1_SAR", "SENTINEL2"): "xoftr",
    ("SENTINEL1_SAR", "LANDSAT8"): "xoftr",
    ("SENTINEL1_SAR", "LANDSAT9"): "xoftr",
    ("SENTINEL1_SAR", "PLANETSCOPE"): "rift2",
    ("ENMAP", "SENTINEL2"): "gim_dkmv3",
    ("PRISMA", "SENTINEL2"): "gim_dkmv3",
}

_DEFAULT_MATCHER = "roma_v2"


class MetaLearner:
    """Learns and recommends matchers based on historical performance."""

    def __init__(self) -> None:
        self.failure_history: list[dict] = []

    def log_failure(
        self,
        source_sensor: str,
        target_sensor: str,
        matcher: str,
        failure_code: str,
        features: dict | None = None,
    ) -> None:
        """Record a failure event."""
        entry = {
            "source_sensor": source_sensor,
            "target_sensor": target_sensor,
            "matcher": matcher,
            "failure_code": failure_code,
            "features": features or {},
        }
        self.failure_history.append(entry)
        logger.info("failure_logged", **entry)

    def recommend_matcher(
        self, source_sensor: str, target_sensor: str
    ) -> str:
        """Recommend the best matcher for a given sensor pair."""
        key = (source_sensor.upper(), target_sensor.upper())
        if key in _ROUTING_RULES:
            return _ROUTING_RULES[key]
        # Reverse lookup
        rev_key = (target_sensor.upper(), source_sensor.upper())
        if rev_key in _ROUTING_RULES:
            return _ROUTING_RULES[rev_key]

        # Check history: avoid matchers with high failure rate for this pair
        pair_failures: dict[str, int] = {}
        for entry in self.failure_history:
            if (
                entry["source_sensor"] == source_sensor
                and entry["target_sensor"] == target_sensor
            ):
                pair_failures[entry["matcher"]] = (
                    pair_failures.get(entry["matcher"], 0) + 1
                )
        if pair_failures:
            # Pick matcher with fewest failures
            all_matchers = [
                "roma_v2",
                "xfeat_lightglue",
                "mast3r",
                "gim_dkmv3",
                "rift2",
            ]
            best = min(
                all_matchers, key=lambda m: pair_failures.get(m, 0)
            )
            return best

        return _DEFAULT_MATCHER

    def save_history(self, path: Path) -> None:
        """Persist failure history to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.failure_history, indent=2))
        logger.info("history_saved", path=str(path))

    def load_history(self, path: Path) -> None:
        """Load failure history from JSON."""
        if path.exists():
            self.failure_history = json.loads(path.read_text())
            logger.info(
                "history_loaded",
                path=str(path),
                entries=len(self.failure_history),
            )
