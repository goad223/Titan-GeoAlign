"""Failure detection for the self-healing pipeline."""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Failure code constants
FEW_MATCHES = "FEW_MATCHES"
HIGH_RMSE = "HIGH_RMSE"
LOW_INLIER_RATIO = "LOW_INLIER_RATIO"
TIMEOUT = "TIMEOUT"
GPU_OOM = "GPU_OOM"
NO_OVERLAP = "NO_OVERLAP"

DEFAULT_THRESHOLDS = {
    "min_matches": 10,
    "max_rmse": 2.0,
    "min_inlier_ratio": 0.2,
}


class FailureDetector:
    """Detects registration failures and categorizes them."""

    FAILURE_CODES: dict[str, str] = {
        FEW_MATCHES: "Insufficient number of matches found",
        HIGH_RMSE: "RMSE exceeds acceptable threshold",
        LOW_INLIER_RATIO: "Too few inliers relative to total matches",
        TIMEOUT: "Processing timed out",
        GPU_OOM: "GPU out-of-memory error",
        NO_OVERLAP: "Images have insufficient overlap",
    }

    CRITICAL_CODES = {GPU_OOM, NO_OVERLAP}

    def __init__(self, thresholds: dict | None = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def detect(self, match_result, qa_report=None) -> list[str]:
        """Detect failure codes from match result and QA report.

        Args:
            match_result: MatchResult object.
            qa_report: Optional QAReport object.

        Returns:
            List of failure codes detected.
        """
        codes: list[str] = []

        if match_result is not None:
            if match_result.num_matches < self.thresholds["min_matches"]:
                codes.append(FEW_MATCHES)
                logger.warning(
                    "failure_detected",
                    code=FEW_MATCHES,
                    num_matches=match_result.num_matches,
                    threshold=self.thresholds["min_matches"],
                )

            if match_result.num_matches > 0:
                inlier_ratio = (
                    match_result.num_inliers / match_result.num_matches
                )
                if inlier_ratio < self.thresholds["min_inlier_ratio"]:
                    codes.append(LOW_INLIER_RATIO)
                    logger.warning(
                        "failure_detected",
                        code=LOW_INLIER_RATIO,
                        inlier_ratio=round(inlier_ratio, 3),
                    )

        if qa_report is not None:
            if qa_report.rmse > self.thresholds["max_rmse"]:
                codes.append(HIGH_RMSE)
                logger.warning(
                    "failure_detected",
                    code=HIGH_RMSE,
                    rmse=round(qa_report.rmse, 4),
                    threshold=self.thresholds["max_rmse"],
                )

        return codes

    def is_critical(self, failure_codes: list[str]) -> bool:
        """Return True if any failure code is critical (cannot retry)."""
        return bool(set(failure_codes) & self.CRITICAL_CODES)
