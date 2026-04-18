"""Unified uncertainty estimator combining MC dropout, ensemble disagreement, and residuals."""
from __future__ import annotations

import numpy as np
import structlog

from uncertainty.mc_dropout import MCDropoutSampler
from uncertainty.ensemble_disagreement import EnsembleDisagreement
from uncertainty.uncertainty_map import UncertaintyMap

logger = structlog.get_logger(__name__)


class UncertaintyEstimator:
    """Combines multiple uncertainty sources into per-match and spatial maps."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.mc_samples: int = cfg.get("mc_dropout_samples", 10)
        self.enabled: bool = cfg.get("enabled", True)
        self._mc_sampler = MCDropoutSampler(n_samples=self.mc_samples)
        self._disagreement = EnsembleDisagreement()

    def estimate(
        self,
        match_result,  # MatchResult
        ensemble_results: dict | None = None,
        residuals: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> dict:
        """Estimate uncertainty from available signals.

        Args:
            match_result: The primary MatchResult.
            ensemble_results: Optional dict of matcher_name -> MatchResult.
            residuals: Optional [N, 2] geometric residuals.
            image_shape: (H, W) for building uncertainty map.

        Returns:
            Dictionary with per_match_uncertainty, global_uncertainty,
            uncertainty_map, and confidence_calibrated.
        """
        if not self.enabled or match_result.num_matches == 0:
            n = match_result.num_matches
            return {
                "per_match_uncertainty": np.zeros(n, dtype=np.float32),
                "global_uncertainty": 0.0,
                "uncertainty_map": None,
                "confidence_calibrated": match_result.confidence.copy(),
            }

        components: list[np.ndarray] = []

        # --- Component 1: Ensemble disagreement ---
        if ensemble_results and len(ensemble_results) >= 2:
            try:
                disagreement = self._disagreement.compute(ensemble_results)
                if len(disagreement) == match_result.num_matches:
                    components.append(disagreement)
            except Exception as exc:
                logger.warning("ensemble_disagreement_failed", error=str(exc))

        # --- Component 2: Geometric residuals ---
        if residuals is not None and len(residuals) == match_result.num_matches:
            residual_magnitude = np.linalg.norm(residuals, axis=1).astype(np.float32)
            # Normalize to [0, 1]
            r_max = residual_magnitude.max() + 1e-8
            components.append(residual_magnitude / r_max)

        # --- Component 3: Inverse confidence ---
        inv_conf = 1.0 - match_result.confidence.astype(np.float32)
        components.append(inv_conf)

        # Combine components (mean)
        per_match_uncertainty = np.mean(
            np.stack(components, axis=0), axis=0
        ).astype(np.float32)

        global_uncertainty = float(np.mean(per_match_uncertainty))

        # Calibrated confidence: confidence * (1 - uncertainty)
        confidence_calibrated = (
            match_result.confidence.astype(np.float32)
            * (1.0 - per_match_uncertainty)
        )
        confidence_calibrated = np.clip(confidence_calibrated, 0.0, 1.0)

        # Build uncertainty map if shape provided
        u_map: UncertaintyMap | None = None
        if image_shape is not None and match_result.num_matches >= 4:
            try:
                u_map = UncertaintyMap(image_shape)
                if residuals is not None:
                    u_map = UncertaintyMap.from_residuals(
                        residuals, match_result.kpts0, image_shape
                    )
                else:
                    u_map = UncertaintyMap.from_ensemble(
                        per_match_uncertainty, match_result.kpts0, image_shape
                    )
            except Exception as exc:
                logger.warning("uncertainty_map_failed", error=str(exc))

        logger.info(
            "uncertainty_estimated",
            global_uncertainty=round(global_uncertainty, 4),
            num_matches=match_result.num_matches,
        )

        return {
            "per_match_uncertainty": per_match_uncertainty,
            "global_uncertainty": global_uncertainty,
            "uncertainty_map": u_map,
            "confidence_calibrated": confidence_calibrated,
        }
