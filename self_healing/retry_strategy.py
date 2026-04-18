"""Retry strategy for self-healing pipeline."""
from __future__ import annotations

import structlog

from self_healing.failure_detector import (
    FEW_MATCHES,
    HIGH_RMSE,
    LOW_INLIER_RATIO,
    GPU_OOM,
)

logger = structlog.get_logger(__name__)

MATCHER_FALLBACK_CHAIN: dict[str, list[str]] = {
    FEW_MATCHES: ["rift2", "gim_dkmv3", "mast3r", "xfeat_lightglue"],
    HIGH_RMSE: ["mast3r", "roma_v2", "gim_dkmv3"],
    LOW_INLIER_RATIO: ["rift2", "xoftr", "gim_dkmv3"],
    GPU_OOM: ["xfeat_lightglue", "rift2"],
    "default": ["xfeat_lightglue", "rift2", "gim_dkmv3", "mast3r"],
}


class RetryStrategy:
    """Determines which matcher to try next on failure."""

    def __init__(self) -> None:
        self.fallback_chain = MATCHER_FALLBACK_CHAIN

    def next_matcher(self, current: str, failure_code: str) -> str | None:
        """Return the next matcher to try, or None if exhausted."""
        chain = self.fallback_chain.get(
            failure_code, self.fallback_chain["default"]
        )
        try:
            idx = chain.index(current)
            return chain[idx + 1] if idx + 1 < len(chain) else None
        except ValueError:
            # current not in chain → start from beginning
            candidates = [m for m in chain if m != current]
            return candidates[0] if candidates else None

    def should_retry(self, attempt: int, max_attempts: int = 3) -> bool:
        """Return True if another retry is warranted."""
        return attempt < max_attempts
