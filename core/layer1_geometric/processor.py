"""GeometricProcessor — chains DEM, RPC, and orthorectification steps."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .dem_provider import DEMProvider
from .rpc_corrector import RPCCorrector
from .orthorectification import OrthorectificationEngine

try:
    from core.layer0_ingestion.image_data import ImageData
except ImportError:
    from ..layer0_ingestion.image_data import ImageData  # type: ignore[no-redef]

logger = structlog.get_logger(__name__)


class GeometricProcessor:
    """High-level geometry processing pipeline.

    Chains :class:`DEMProvider`, :class:`RPCCorrector`, and
    :class:`OrthorectificationEngine` for end-to-end sensor correction.

    Parameters
    ----------
    dem_source:
        DEM dataset for terrain correction (default: ``"copernicus_glo30"``).
    dem_cache_dir:
        Cache directory for downloaded DEM tiles.
    dem_resolution_m:
        Ground resolution of the DEM used for terrain correction.
    resampling:
        Resampling algorithm for orthorectification.
    """

    def __init__(
        self,
        dem_source: str = "copernicus_glo30",
        dem_cache_dir: Path = Path("data/dem_cache"),
        dem_resolution_m: float = 30.0,
        resampling: str = "bilinear",
    ) -> None:
        self.dem_provider = DEMProvider(
            source=dem_source, cache_dir=dem_cache_dir
        )
        self.rpc_corrector = RPCCorrector()
        self.ortho_engine = OrthorectificationEngine(resampling=resampling)
        self.dem_resolution_m = dem_resolution_m

    # ------------------------------------------------------------------

    def process(self, image: ImageData, config: Optional[dict] = None) -> ImageData:
        """Apply geometric corrections to *image*.

        Parameters
        ----------
        image:
            Raw or map-projected sensor image.
        config:
            Optional configuration overrides.  Recognised keys:

            ``"skip_dem"`` (bool):
                Skip DEM download and terrain correction.
            ``"skip_rpc"`` (bool):
                Skip RPC correction step.
            ``"skip_ortho"`` (bool):
                Skip orthorectification step.
            ``"target_crs"`` (str):
                Desired output CRS (default: ``"EPSG:4326"``).
            ``"dem_resolution_m"`` (float):
                Override DEM resolution.
            ``"resampling"`` (str):
                Override resampling algorithm.
        """
        cfg = config or {}

        skip_dem = cfg.get("skip_dem", False)
        skip_rpc = cfg.get("skip_rpc", False)
        skip_ortho = cfg.get("skip_ortho", False)
        target_crs = cfg.get("target_crs", "EPSG:4326")
        dem_res = float(cfg.get("dem_resolution_m", self.dem_resolution_m))

        if "resampling" in cfg:
            self.ortho_engine.resampling = cfg["resampling"]

        logger.info(
            "geometric processing start",
            sensor=image.sensor_type.value,
            skip_dem=skip_dem,
            skip_rpc=skip_rpc,
            skip_ortho=skip_ortho,
        )

        # 1. Fetch DEM
        dem: Optional[np.ndarray] = None
        if not skip_dem:
            try:
                dem = self.dem_provider.get_dem(image.bounds, resolution_m=dem_res)
                logger.debug("DEM loaded", shape=dem.shape)
            except Exception as exc:
                logger.warning("DEM fetch failed", error=str(exc))

        # 2. Parse and apply RPC correction
        rpc: Optional[dict] = None
        if not skip_rpc:
            if image.filepath is not None:
                rpc = self.rpc_corrector.parse_rpc(image.filepath)
            if rpc:
                image = self.rpc_corrector.correct(image, dem or np.zeros((1, 1)), rpc)
                logger.debug("RPC correction applied")

        # 3. Orthorectification
        if not skip_ortho:
            image = self.ortho_engine.orthorectify(image, dem=dem, rpc=rpc)
            logger.debug("orthorectification applied")

        # 4. Final reproject to target CRS if needed
        if image.crs != target_crs:
            try:
                image = image.reproject(target_crs)
                logger.debug("reprojected to target CRS", crs=target_crs)
            except Exception as exc:
                logger.warning("final reproject failed", error=str(exc))

        logger.info("geometric processing complete", output_crs=image.crs)
        return image
