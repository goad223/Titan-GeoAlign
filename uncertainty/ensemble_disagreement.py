"""Ensemble disagreement: uncertainty from position spread across matchers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import structlog

from core.layer4_matchers.base_matcher import MatchResult

logger = structlog.get_logger(__name__)


class EnsembleDisagreement:
    """Compute positional uncertainty from disagreement between matchers."""

    def compute(self, results: Dict[str, MatchResult]) -> np.ndarray:
        """Compute per-match uncertainty as position std dev across matchers.

        For each common match (identified by proximity in image 0), computes
        the std dev of predicted positions in image 1.

        Returns
        -------
        np.ndarray [N]
            Scalar uncertainty per match of the reference (first) matcher.
        """
        if not results:
            return np.zeros(0, dtype=np.float32)

        names = list(results.keys())
        reference_name = names[0]
        reference = results[reference_name]

        if reference.num_matches == 0:
            return np.zeros(0, dtype=np.float32)

        n_ref = reference.num_matches
        n_matchers = len(results)

        # Collect kpts1 positions across matchers aligned to reference kpts0
        kpts1_collection = np.zeros((n_matchers, n_ref, 2), dtype=np.float32)
        kpts1_collection[0] = reference.kpts1

        for m_idx, name in enumerate(names[1:], start=1):
            other = results[name]
            if other.num_matches == 0:
                kpts1_collection[m_idx] = reference.kpts1
                continue
            for i in range(n_ref):
                dists = np.linalg.norm(other.kpts0 - reference.kpts0[i], axis=1)
                best = int(np.argmin(dists))
                if dists[best] < 10.0:  # accept if within 10px
                    kpts1_collection[m_idx, i] = other.kpts1[best]
                else:
                    kpts1_collection[m_idx, i] = reference.kpts1[i]

        # Std dev across matchers, collapsed to scalar per match
        std_dev = kpts1_collection.std(axis=0)  # [N, 2]
        uncertainty = np.linalg.norm(std_dev, axis=1).astype(np.float32)  # [N]

        logger.debug(
            "ensemble_disagreement",
            matchers=names,
            mean_uncertainty=float(uncertainty.mean()),
        )
        return uncertainty
