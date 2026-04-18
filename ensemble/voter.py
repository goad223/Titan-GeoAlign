"""Ensemble voter: fuses MatchResults from multiple matchers."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import structlog

from core.layer4_matchers.base_matcher import MatchResult

logger = structlog.get_logger(__name__)


class EnsembleVoter:
    """Fuses match results from multiple matchers using confidence-weighted voting."""

    def __init__(self, distance_threshold: float = 3.0) -> None:
        self.distance_threshold = distance_threshold

    # ------------------------------------------------------------------
    def vote(self, results: List[MatchResult]) -> MatchResult:
        """Combine results from multiple matchers via confidence-weighted voting.

        Steps:
        1. Pool all matches.
        2. Cluster spatially near-duplicate matches.
        3. Return fused MatchResult with averaged positions and combined confidences.
        """
        if not results:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(
                kpts0=empty,
                kpts1=empty,
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name="ensemble",
            )

        all_kpts0 = np.concatenate([r.kpts0 for r in results], axis=0)
        all_kpts1 = np.concatenate([r.kpts1 for r in results], axis=0)
        all_conf = np.concatenate([r.confidence for r in results], axis=0)

        if len(all_kpts0) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(
                kpts0=empty,
                kpts1=empty,
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name="ensemble",
            )

        fused_k0, fused_k1, fused_conf = self._cluster_matches(
            all_kpts0, all_kpts1, all_conf, self.distance_threshold
        )

        num_inliers = sum(r.num_inliers for r in results)
        total_time = sum(r.processing_time_s for r in results)

        logger.debug(
            "ensemble_vote",
            input_matches=len(all_kpts0),
            fused_matches=len(fused_k0),
            matchers=[r.matcher_name for r in results],
        )

        return MatchResult(
            kpts0=fused_k0,
            kpts1=fused_k1,
            confidence=fused_conf,
            matcher_name="ensemble",
            num_inliers=num_inliers,
            processing_time_s=total_time,
            metadata={"source_matchers": [r.matcher_name for r in results]},
        )

    # ------------------------------------------------------------------
    def _cluster_matches(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        confidences: np.ndarray,
        distance_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group nearby correspondences and return one representative per group.

        Uses a greedy radius-based clustering on the concatenated position space
        [x0, y0, x1, y1] for joint matching consistency.
        """
        n = len(kpts0)
        if n == 0:
            return kpts0, kpts1, confidences

        # joint 4-D positions for clustering
        positions = np.concatenate([kpts0, kpts1], axis=1)  # [N, 4]

        assigned = np.full(n, -1, dtype=np.int32)
        group_id = 0

        for i in range(n):
            if assigned[i] != -1:
                continue
            assigned[i] = group_id
            pi = positions[i]
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                dist = np.linalg.norm(positions[j] - pi)
                if dist < distance_threshold:
                    assigned[j] = group_id
            group_id += 1

        num_groups = group_id
        fused_k0 = np.empty((num_groups, 2), dtype=np.float32)
        fused_k1 = np.empty((num_groups, 2), dtype=np.float32)
        fused_conf = np.empty(num_groups, dtype=np.float32)

        for g in range(num_groups):
            mask = assigned == g
            group_conf = confidences[mask]
            fused_k0[g] = np.average(kpts0[mask], axis=0, weights=group_conf)
            fused_k1[g] = np.average(kpts1[mask], axis=0, weights=group_conf)
            fused_conf[g] = self.compute_ensemble_confidence(group_conf.tolist())

        return fused_k0, fused_k1, fused_conf

    # ------------------------------------------------------------------
    @staticmethod
    def compute_ensemble_confidence(group_confidences: List[float]) -> float:
        """Combine confidences: geometric mean weighted by count."""
        if not group_confidences:
            return 0.0
        n = len(group_confidences)
        log_sum = sum(math.log(max(c, 1e-9)) for c in group_confidences)
        geo_mean = math.exp(log_sum / n)
        # Boost slightly for larger groups (more matchers agree)
        count_boost = min(1.0, 0.5 + 0.5 * math.log1p(n) / math.log1p(5))
        return float(np.clip(geo_mean * count_boost, 0.0, 1.0))
