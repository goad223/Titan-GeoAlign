"""Local Correction Engine — orchestrates AROSICS/DTPG correction with QA."""
from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..layer4_matchers.base_matcher import MatchResult
from ..layer6_ransac.estimators import RobustEstimator
from ..layer6_ransac.model_types import ModelType
from .residual_analysis import ResidualAnalyzer
from .qa_report import QAReport

logger = structlog.get_logger(__name__)

_DEFAULT_CONFIG = {
    "threshold_rmse": 2.0,
    "min_inliers": 10,
    "grid_res": 200,
}


class LocalCorrectionEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._estimator = RobustEstimator(method="magsac++", reproj_threshold=3.0)
        self._analyzer = ResidualAnalyzer()

    def correct(
        self,
        source: Any,
        target: Any,
        match_result: MatchResult,
    ) -> tuple[Any, QAReport]:
        """Apply local correction and return corrected ImageData + QAReport."""
        t0 = time.perf_counter()
        threshold = float(self.config.get("threshold_rmse", 2.0))

        # Robust estimation
        est = self._estimator.estimate(match_result.kpts0, match_result.kpts1,
                                       model_type=ModelType.HOMOGRAPHY)
        residuals = self._analyzer.compute_residuals(
            match_result.kpts0, match_result.kpts1, est.transform_matrix
        )
        stats = self._analyzer.compute_statistics(residuals)
        rmse = stats["rmse"]
        max_res = stats["max"]
        passes = rmse <= threshold and est.num_inliers >= int(self.config.get("min_inliers", 10))

        # Apply warp to source image if transform found
        corrected = source
        if est.transform_matrix is not None:
            try:
                from ..layer6_ransac.transforms import warp_image
                arr = source.data if hasattr(source, "data") else source
                if isinstance(arr, np.ndarray):
                    warped = warp_image(arr, est)
                    if hasattr(source, "data"):
                        import copy
                        corrected = copy.copy(source)
                        corrected.data = warped
                    else:
                        corrected = warped
            except Exception as exc:
                self.logger.warning("Warp application failed", error=str(exc))

        # Auto-retry with different model if RMSE too high
        if not passes and est.num_inliers >= 3:
            self.logger.info("RMSE above threshold; retrying with AFFINE model", rmse=rmse)
            est2 = self._estimator.estimate(match_result.kpts0, match_result.kpts1,
                                            model_type=ModelType.AFFINE)
            res2 = self._analyzer.compute_residuals(match_result.kpts0, match_result.kpts1, est2.transform_matrix)
            stats2 = self._analyzer.compute_statistics(res2)
            if stats2["rmse"] < rmse:
                est, stats, rmse, max_res = est2, stats2, stats2["rmse"], stats2["max"]
                passes = rmse <= threshold

        src_path = Path(source.path) if hasattr(source, "path") and source.path else Path("unknown")
        tgt_path = Path(target.path) if hasattr(target, "path") and target.path else Path("unknown")

        report = QAReport(
            source_path=src_path,
            target_path=tgt_path,
            num_tie_points=match_result.num_matches,
            num_inliers=est.num_inliers,
            rmse=rmse,
            max_residual=max_res,
            passes_qa=passes,
            threshold_rmse=threshold,
            matcher_used=match_result.matcher_name,
            processing_time_s=time.perf_counter() - t0,
            timestamp=datetime.utcnow(),
        )
        self.logger.info("Local correction complete", rmse=round(rmse, 4), passes_qa=passes,
                         n_inliers=est.num_inliers)
        return corrected, report
