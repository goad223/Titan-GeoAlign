"""MatcherEnsemble: runs multiple matchers in parallel and fuses results."""
from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import structlog

from core.layer4_matchers.base_matcher import BaseMatcher, MatchResult
from .voter import EnsembleVoter

logger = structlog.get_logger(__name__)

_MATCHER_REGISTRY: Dict[str, str] = {
    "xoftr": "core.layer4_matchers.xoftr_matcher.XoFTRMatcher",
    "gim": "core.layer4_matchers.gim_matcher.GIMMatcher",
    "orb_bf": "core.layer4_matchers.base_matcher.BaseMatcher",
}


def _load_matcher(name: str, config: dict) -> BaseMatcher:
    """Dynamically load a matcher by name."""
    if name not in _MATCHER_REGISTRY:
        raise ValueError(f"Unknown matcher: {name!r}. Available: {list(_MATCHER_REGISTRY)}")
    module_path, class_name = _MATCHER_REGISTRY[name].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        matcher = cls(**config.get(name, {}))
        matcher.load()
        return matcher
    except Exception as exc:  # noqa: BLE001
        logger.warning("matcher_load_failed", name=name, error=str(exc))
        raise


class MatcherEnsemble:
    """Runs multiple matchers in parallel and fuses their results."""

    def __init__(self, matchers: List[str], config: dict) -> None:
        self.matcher_names = matchers
        self.config = config
        self._voter = EnsembleVoter(
            distance_threshold=config.get("distance_threshold", 3.0)
        )
        self._last_results: Dict[str, MatchResult] = {}
        self._loaded_matchers: Dict[str, BaseMatcher] = {}

        for name in matchers:
            try:
                self._loaded_matchers[name] = _load_matcher(name, config)
            except Exception:  # noqa: BLE001
                logger.warning("skipping_matcher", name=name)

    # ------------------------------------------------------------------
    def match(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Run all configured matchers in parallel and fuse results.

        Parameters
        ----------
        img0, img1:
            Input images (HxWxC or HxW, uint8 or float32).
        semantic_mask:
            Optional binary mask (H, W) marking unstable regions (1=unstable).
            Keypoints in unstable regions are filtered out post-matching.
        """
        results: Dict[str, MatchResult] = {}

        def _run_one(name: str, matcher: BaseMatcher) -> tuple[str, MatchResult]:
            t0 = time.perf_counter()
            res = matcher.match(img0, img1)
            res.processing_time_s = time.perf_counter() - t0
            return name, res

        with ThreadPoolExecutor(max_workers=len(self._loaded_matchers) or 1) as ex:
            futures = {
                ex.submit(_run_one, name, m): name
                for name, m in self._loaded_matchers.items()
            }
            for future in as_completed(futures):
                try:
                    name, res = future.result()
                    results[name] = res
                except Exception as exc:  # noqa: BLE001
                    logger.warning("matcher_error", name=futures[future], error=str(exc))

        if semantic_mask is not None:
            results = {
                name: self._apply_semantic_mask(res, semantic_mask)
                for name, res in results.items()
            }

        self._last_results = results

        if not results:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(
                kpts0=empty,
                kpts1=empty,
                confidence=np.zeros(0, dtype=np.float32),
                matcher_name="ensemble",
            )

        fused = self._voter.vote(list(results.values()))
        logger.info(
            "ensemble_match",
            matchers=list(results.keys()),
            individual_counts={k: v.num_matches for k, v in results.items()},
            fused_count=fused.num_matches,
        )
        return fused

    # ------------------------------------------------------------------
    def get_individual_results(self) -> Dict[str, MatchResult]:
        """Return results from the last call to match()."""
        return dict(self._last_results)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_semantic_mask(res: MatchResult, mask: np.ndarray) -> MatchResult:
        """Filter out keypoints that fall inside unstable semantic regions."""
        if res.num_matches == 0:
            return res
        h, w = mask.shape[:2]

        def _in_bounds(pts: np.ndarray) -> np.ndarray:
            x = np.clip(pts[:, 0].astype(int), 0, w - 1)
            y = np.clip(pts[:, 1].astype(int), 0, h - 1)
            return mask[y, x] == 0  # keep stable regions

        keep = _in_bounds(res.kpts0) & _in_bounds(res.kpts1)
        return MatchResult(
            kpts0=res.kpts0[keep],
            kpts1=res.kpts1[keep],
            confidence=res.confidence[keep],
            matcher_name=res.matcher_name,
            num_inliers=res.num_inliers,
            processing_time_s=res.processing_time_s,
            metadata=res.metadata,
        )
