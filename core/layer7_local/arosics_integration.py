"""AROSICS integration for global and local co-registration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class AROSICSIntegration:
    """Wraps the AROSICS library for global and local image co-registration.

    Falls back to our own phase-correlation-based global shift when
    AROSICS is not installed.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._arosics_available: bool | None = None

    # ------------------------------------------------------------------
    def _check_arosics(self) -> bool:
        if self._arosics_available is None:
            try:
                import arosics  # type: ignore[import]
                self._arosics_available = True
            except ImportError:
                self._arosics_available = False
                self.logger.warning("arosics not installed; using phase-correlation fallback")
        return self._arosics_available  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def global_correct(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        config: dict | None = None,
    ) -> dict:
        """Apply global co-registration.

        Parameters
        ----------
        source_path:
            Path to the source (moving) raster.
        target_path:
            Path to the target (reference) raster.
        output_path:
            Path where the corrected raster will be saved.
        config:
            Optional dict forwarded to ``arosics.COREG`` as keyword args.

        Returns
        -------
        dict with keys: ``shift_x``, ``shift_y``, ``rmse``, ``tie_points``,
        ``success``, ``backend``.
        """
        config = config or {}
        if self._check_arosics():
            return self._global_arosics(source_path, target_path, output_path, config)
        return self._global_phase_corr(source_path, target_path, output_path)

    def _global_arosics(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        config: dict,
    ) -> dict:
        try:
            import arosics  # type: ignore[import]

            cr = arosics.COREG(
                str(target_path),
                str(source_path),
                path_out=str(output_path),
                **config,
            )
            cr.calculate_spatial_shifts()
            cr.correct_shifts()
            result = {
                "shift_x": float(cr.coreg_info.get("corrected_shifts_px", {}).get("x", 0.0)),
                "shift_y": float(cr.coreg_info.get("corrected_shifts_px", {}).get("y", 0.0)),
                "rmse": float(cr.coreg_info.get("RMSE_SHIFT", 0.0)),
                "tie_points": 1,
                "success": True,
                "backend": "arosics_global",
            }
            self.logger.info("AROSICS global correction applied", **{k: v for k, v in result.items() if k != "tie_points"})
            return result
        except Exception as exc:
            self.logger.warning("AROSICS global correction failed; using fallback", error=str(exc))
            return self._global_phase_corr(source_path, target_path, output_path)

    def _global_phase_corr(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
    ) -> dict:
        """Phase-correlation global shift fallback (no AROSICS)."""
        try:
            import cv2  # type: ignore[import]

            src_img = self._read_gray(source_path)
            tgt_img = self._read_gray(target_path)

            h = min(src_img.shape[0], tgt_img.shape[0])
            w = min(src_img.shape[1], tgt_img.shape[1])
            src_crop = src_img[:h, :w]
            tgt_crop = tgt_img[:h, :w]

            shift, _ = cv2.phaseCorrelate(src_crop.astype(np.float64), tgt_crop.astype(np.float64))
            dx, dy = float(shift[0]), float(shift[1])

            # Apply shift and save
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            src_full = self._read_image(source_path)
            corrected = cv2.warpAffine(src_full, M, (src_full.shape[1], src_full.shape[0]))
            self._save_image(corrected, output_path, source_path)

            return {
                "shift_x": dx,
                "shift_y": dy,
                "rmse": 0.0,
                "tie_points": 1,
                "success": True,
                "backend": "phase_correlation_fallback",
            }
        except Exception as exc:
            self.logger.error("Phase-correlation global fallback failed", error=str(exc))
            return {"shift_x": 0.0, "shift_y": 0.0, "rmse": float("inf"), "tie_points": 0, "success": False, "backend": "none"}

    # ------------------------------------------------------------------
    def local_correct(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        grid_res: int = 200,
    ) -> dict:
        """Apply local co-registration using a tie-point grid.

        Parameters
        ----------
        grid_res:
            Grid resolution in pixels.

        Returns
        -------
        dict with keys: ``shift_x``, ``shift_y``, ``rmse``, ``tie_points``,
        ``success``, ``backend``.
        """
        if self._check_arosics():
            return self._local_arosics(source_path, target_path, output_path, grid_res)
        return self._local_dtpg(source_path, target_path, output_path, grid_res)

    def _local_arosics(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        grid_res: int,
    ) -> dict:
        try:
            import arosics  # type: ignore[import]

            cl = arosics.COREG_LOCAL(
                str(target_path),
                str(source_path),
                path_out=str(output_path),
                grid_res=grid_res,
            )
            cl.correct_shifts()
            tps = cl.CoRegPoints_table
            n_tps = len(tps) if tps is not None else 0
            shifts_x = tps["SHIFT_X"].dropna().tolist() if tps is not None and "SHIFT_X" in tps.columns else [0.0]
            shifts_y = tps["SHIFT_Y"].dropna().tolist() if tps is not None and "SHIFT_Y" in tps.columns else [0.0]
            rmse_vals = tps["RMSE"].dropna().tolist() if tps is not None and "RMSE" in tps.columns else [0.0]
            result = {
                "shift_x": float(np.mean(shifts_x)),
                "shift_y": float(np.mean(shifts_y)),
                "rmse": float(np.mean(rmse_vals)),
                "tie_points": n_tps,
                "success": True,
                "backend": "arosics_local",
            }
            self.logger.info("AROSICS local correction applied", **{k: v for k, v in result.items() if k not in ("tie_points",)})
            return result
        except Exception as exc:
            self.logger.warning("AROSICS local correction failed; using fallback", error=str(exc))
            return self._local_dtpg(source_path, target_path, output_path, grid_res)

    def _local_dtpg(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        grid_res: int,
    ) -> dict:
        """Dense tie-point grid fallback."""
        from .dtpg import DenseTiePointGrid

        try:
            src_img = self._read_gray(source_path)
            tgt_img = self._read_gray(target_path)
            dtpg = DenseTiePointGrid()
            tps = dtpg.generate(src_img, tgt_img, grid_size=grid_res)
            tps_filtered = dtpg.filter_outliers(sigma=3.0)

            shifts_x = [t["dx"] for t in tps]
            shifts_y = [t["dy"] for t in tps]
            result = {
                "shift_x": float(np.mean(shifts_x)) if shifts_x else 0.0,
                "shift_y": float(np.mean(shifts_y)) if shifts_y else 0.0,
                "rmse": float(np.std(shifts_x + shifts_y)) if shifts_x else 0.0,
                "tie_points": len(tps),
                "success": len(tps) > 0,
                "backend": "dtpg_fallback",
            }
            # Save corrected image using mean shift
            import cv2  # type: ignore[import]
            src_full = self._read_image(source_path)
            M = np.float32([[1, 0, -result["shift_x"]], [0, 1, -result["shift_y"]]])
            corrected = cv2.warpAffine(src_full, M, (src_full.shape[1], src_full.shape[0]))
            self._save_image(corrected, output_path, source_path)
            return result
        except Exception as exc:
            self.logger.error("DTPG local fallback failed", error=str(exc))
            return {"shift_x": 0.0, "shift_y": 0.0, "rmse": float("inf"), "tie_points": 0, "success": False, "backend": "none"}

    # ------------------------------------------------------------------
    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        import cv2  # type: ignore[import]

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # Try with rasterio
            try:
                import rasterio  # type: ignore[import]
                with rasterio.open(str(path)) as ds:
                    img = ds.read(1)
            except Exception:
                raise FileNotFoundError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        import cv2  # type: ignore[import]

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            try:
                import rasterio  # type: ignore[import]
                with rasterio.open(str(path)) as ds:
                    data = ds.read()
                    img = data.transpose(1, 2, 0) if data.ndim == 3 else data
            except Exception:
                raise FileNotFoundError(f"Cannot read {path}")
        return img

    @staticmethod
    def _save_image(img: np.ndarray, output_path: Path, reference_path: Path) -> None:
        """Save image, preserving geotransform if rasterio is available."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import rasterio  # type: ignore[import]
            from rasterio.transform import from_origin  # type: ignore[import]

            with rasterio.open(str(reference_path)) as ref:
                meta = ref.meta.copy()
            if img.ndim == 2:
                data = img[np.newaxis, :, :]
                meta.update(count=1, height=img.shape[0], width=img.shape[1])
            else:
                data = img.transpose(2, 0, 1)
                meta.update(count=data.shape[0], height=img.shape[0], width=img.shape[1])
            meta["driver"] = "GTiff"
            with rasterio.open(str(output_path), "w", **meta) as dst:
                dst.write(data)
        except Exception:
            import cv2  # type: ignore[import]
            cv2.imwrite(str(output_path), img)
