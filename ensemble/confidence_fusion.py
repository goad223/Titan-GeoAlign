"""Confidence-weighted fusion of multiple MatchResults."""
from __future__ import annotations

from typing import Dict

import numpy as np
import structlog

from core.layer4_matchers.base_matcher import MatchResult

logger = structlog.get_logger(__name__)


class ConfidenceWeightedFusion:
    """Fuse results from named matchers using per-matcher weights."""

    def __init__(self) -> None:
        self.fusion_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def fuse(self, results: Dict[str, MatchResult]) -> MatchResult:
        """Weighted fusion combining match quality metrics.

        Quality metrics considered:
        - num_inliers
        - mean_confidence
        - processing_time (lower is better, used as inverse weight)

        Returns a MatchResult combining all inputs proportionally.
        """
        if not results:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(
                kpts0=empty,
                kpts1=empty,
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name="fused",
            )

        quality_scores: Dict[str, float] = {}
        for name, res in results.items():
            base_w = self.fusion_weights.get(name, 1.0)
            mean_conf = float(res.confidence.mean()) if len(res.confidence) > 0 else 0.0
            inlier_score = min(1.0, res.num_inliers / max(1, res.num_matches))
            time_penalty = 1.0 / (1.0 + res.processing_time_s)
            quality_scores[name] = base_w * (0.5 * mean_conf + 0.3 * inlier_score + 0.2 * time_penalty)

        total_quality = sum(quality_scores.values())
        if total_quality == 0:
            total_quality = 1.0

        all_kpts0 = []
        all_kpts1 = []
        all_conf = []
        total_time = 0.0
        total_inliers = 0

        for name, res in results.items():
            if res.num_matches == 0:
                continue
            w = quality_scores[name] / total_quality
            scaled_conf = res.confidence * w
            all_kpts0.append(res.kpts0)
            all_kpts1.append(res.kpts1)
            all_conf.append(scaled_conf)
            total_time += res.processing_time_s
            total_inliers += res.num_inliers

        if not all_kpts0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(
                kpts0=empty,
                kpts1=empty,
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name="fused",
            )

        fused_k0 = np.concatenate(all_kpts0, axis=0)
        fused_k1 = np.concatenate(all_kpts1, axis=0)
        fused_conf = np.concatenate(all_conf, axis=0)

        logger.debug(
            "confidence_weighted_fusion",
            matchers=list(results.keys()),
            quality_scores=quality_scores,
            total_matches=len(fused_k0),
        )

        return MatchResult(
            kpts0=fused_k0,
            kpts1=fused_k1,
            confidence=fused_conf,
            matcher_name="fused",
            num_inliers=total_inliers,
            processing_time_s=total_time,
            metadata={"quality_scores": quality_scores},
        )

    # ------------------------------------------------------------------
    def update_weights(self, matcher_name: str, performance_score: float) -> None:
        """Online weight update based on observed matcher performance.

        Uses exponential moving average: w_new = 0.8 * w_old + 0.2 * score.
        """
        current = self.fusion_weights.get(matcher_name, 1.0)
        self.fusion_weights[matcher_name] = 0.8 * current + 0.2 * performance_score
        logger.debug(
            "weight_updated",
            matcher=matcher_name,
            old_weight=current,
            new_weight=self.fusion_weights[matcher_name],
        )
