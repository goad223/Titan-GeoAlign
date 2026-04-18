"""Self-healing pipeline orchestrator."""
from __future__ import annotations

import time

import numpy as np
import structlog

from self_healing.failure_detector import FailureDetector
from self_healing.retry_strategy import RetryStrategy
from self_healing.meta_learner import MetaLearner

logger = structlog.get_logger(__name__)


class SelfHealingPipeline:
    """Orchestrates matching with automatic failure recovery."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.max_attempts: int = cfg.get("max_attempts", 3)
        self.rmse_threshold: float = cfg.get("rmse_threshold", 2.0)
        self.min_matches: int = cfg.get("min_matches", 10)
        self._failure_detector = FailureDetector(
            thresholds={
                "min_matches": self.min_matches,
                "max_rmse": self.rmse_threshold,
                "min_inlier_ratio": 0.2,
            }
        )
        self._retry_strategy = RetryStrategy()
        self._meta_learner = MetaLearner()

    def run(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        source_sensor: str = "UNKNOWN",
        target_sensor: str = "UNKNOWN",
        initial_matcher: str = "auto",
    ):
        """Run matching with self-healing retries.

        Args:
            img0: Source image array [H, W, C] uint8 or float32.
            img1: Target image array [H, W, C] uint8 or float32.
            source_sensor: Sensor type name for routing.
            target_sensor: Sensor type name for routing.
            initial_matcher: Matcher name or "auto".

        Returns:
            Tuple of (MatchResult, best_matcher_name).
        """
        from core.layer4_matchers.matcher_factory import get_matcher
        from core.layer6_ransac.estimators import RobustEstimator

        if initial_matcher == "auto":
            initial_matcher = self._meta_learner.recommend_matcher(
                source_sensor, target_sensor
            )

        current_matcher_name = initial_matcher
        best_result = None
        best_score = -1.0

        for attempt in range(self.max_attempts):
            logger.info(
                "matching_attempt",
                attempt=attempt + 1,
                matcher=current_matcher_name,
            )
            t0 = time.time()
            try:
                matcher = get_matcher(current_matcher_name)
                matcher.load()
                result = matcher.match(img0, img1)
            except Exception as exc:
                logger.warning(
                    "matcher_error",
                    matcher=current_matcher_name,
                    error=str(exc),
                )
                failure_codes = ["FEW_MATCHES"]
                self._meta_learner.log_failure(
                    source_sensor,
                    target_sensor,
                    current_matcher_name,
                    "EXCEPTION",
                )
                next_m = self._retry_strategy.next_matcher(
                    current_matcher_name, "FEW_MATCHES"
                )
                if next_m:
                    current_matcher_name = next_m
                continue

            # Evaluate result
            estimator = RobustEstimator()
            if result.num_matches >= 4:
                try:
                    est_result = estimator.estimate(
                        result.kpts0, result.kpts1
                    )
                    result.num_inliers = est_result.num_inliers
                    rmse = est_result.rmse
                except Exception:
                    rmse = 999.0
            else:
                rmse = 999.0

            score = result.num_inliers / (rmse + 1e-8)
            if score > best_score:
                best_score = score
                best_result = result

            failure_codes = self._failure_detector.detect(result)
            if not failure_codes or rmse <= self.rmse_threshold:
                logger.info(
                    "matching_succeeded",
                    matcher=current_matcher_name,
                    num_inliers=result.num_inliers,
                    rmse=round(rmse, 4),
                    elapsed=round(time.time() - t0, 2),
                )
                return best_result, current_matcher_name

            if self._failure_detector.is_critical(failure_codes):
                logger.error("critical_failure", codes=failure_codes)
                break

            next_m = self._retry_strategy.next_matcher(
                current_matcher_name, failure_codes[0]
            )
            if next_m is None or not self._retry_strategy.should_retry(
                attempt + 1, self.max_attempts
            ):
                break

            self._meta_learner.log_failure(
                source_sensor,
                target_sensor,
                current_matcher_name,
                failure_codes[0],
            )
            current_matcher_name = next_m

        if best_result is None:
            # Return empty result
            from core.layer4_matchers.base_matcher import MatchResult
            best_result = MatchResult(
                kpts0=np.zeros((0, 2), dtype=np.float32),
                kpts1=np.zeros((0, 2), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name=current_matcher_name,
            )

        return best_result, current_matcher_name
