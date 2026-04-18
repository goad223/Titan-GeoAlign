"""Sub-pixel refinement pipeline."""
from __future__ import annotations

import numpy as np
import structlog

from ..layer4_matchers.base_matcher import MatchResult

logger = structlog.get_logger(__name__)

_METHODS = ("phase_correlation", "lsm", "learned")


class SubpixelPipeline:
    """Orchestrates sub-pixel refinement for a MatchResult.

    Parameters
    ----------
    method:
        ``"phase_correlation"`` | ``"lsm"`` | ``"learned"``
    patch_size:
        Patch size for all refiners.
    device:
        Device string for ``LearnedSubpixelRefiner``.
    """

    def __init__(
        self,
        method: str = "phase_correlation",
        patch_size: int = 32,
        device: str = "auto",
        max_lsm_iterations: int = 100,
        lsm_convergence: float = 0.001,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
        self.method = method
        self.patch_size = patch_size
        self.device = device
        self.max_lsm_iterations = max_lsm_iterations
        self.lsm_convergence = lsm_convergence
        self.logger = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def refine(
        self,
        match_result: MatchResult,
        img0: np.ndarray,
        img1: np.ndarray,
        method: str | None = None,
    ) -> MatchResult:
        """Apply sub-pixel refinement to *match_result*.

        Parameters
        ----------
        match_result:
            Input coarse matches.
        img0, img1:
            Source images corresponding to ``match_result.kpts0/kpts1``.
        method:
            Override the instance default method.

        Returns
        -------
        MatchResult
            New MatchResult with refined keypoint positions.
        """
        m = method if method is not None else self.method
        if m not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{m}'")

        if match_result.num_matches == 0:
            self.logger.debug("No matches to refine, returning as-is")
            return match_result

        kpts0 = match_result.kpts0.copy()
        kpts1 = match_result.kpts1.copy()

        try:
            refined0, refined1 = self._apply(m, img0, img1, kpts0, kpts1)
        except Exception as exc:
            self.logger.warning(
                "Sub-pixel refinement failed; returning original matches",
                method=m,
                error=str(exc),
            )
            return match_result

        self.logger.info(
            "Sub-pixel refinement complete",
            method=m,
            n_matches=len(refined0),
            mean_shift=float(np.mean(np.linalg.norm(refined1 - kpts1, axis=1))),
        )

        return MatchResult(
            kpts0=refined0,
            kpts1=refined1,
            confidence=match_result.confidence.copy(),
            matcher_name=match_result.matcher_name,
            num_inliers=match_result.num_inliers,
            processing_time_s=match_result.processing_time_s,
            metadata={**match_result.metadata, "subpixel_method": m},
        )

    # ------------------------------------------------------------------
    def _apply(
        self,
        method: str,
        img0: np.ndarray,
        img1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if method == "phase_correlation":
            from .phase_correlation import PhaseCorrelationRefiner
            return PhaseCorrelationRefiner(patch_size=self.patch_size).refine(img0, img1, kpts0, kpts1)

        if method == "lsm":
            from .lsm_refiner import LSMRefiner
            return LSMRefiner(
                patch_size=self.patch_size,
                max_iterations=self.max_lsm_iterations,
                convergence_threshold=self.lsm_convergence,
            ).refine(img0, img1, kpts0, kpts1)

        if method == "learned":
            from .learned_refiner import LearnedSubpixelRefiner
            refiner = LearnedSubpixelRefiner(device=self.device, patch_size=self.patch_size)
            refiner.load()
            return refiner.refine(img0, img1, kpts0, kpts1)

        raise ValueError(f"Unknown method: {method}")
