"""Base matcher interface for all feature matchers."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import structlog

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class MatchResult:
    """Result of a matching operation."""

    kpts0: np.ndarray  # [N, 2] keypoints in image 0
    kpts1: np.ndarray  # [N, 2] keypoints in image 1
    confidence: np.ndarray  # [N] confidence scores [0, 1]
    matcher_name: str
    num_inliers: int = 0
    processing_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def num_matches(self) -> int:
        return len(self.kpts0)

    def filter_by_confidence(self, threshold: float = 0.5) -> MatchResult:
        mask = self.confidence >= threshold
        return MatchResult(
            kpts0=self.kpts0[mask],
            kpts1=self.kpts1[mask],
            confidence=self.confidence[mask],
            matcher_name=self.matcher_name,
            num_inliers=self.num_inliers,
            processing_time_s=self.processing_time_s,
        )


class BaseMatcher(abc.ABC):
    """Abstract base class for all matchers."""

    name: str = "base"

    def __init__(self, device: str = "auto", models_dir: str = "models/matchers"):
        if _TORCH_AVAILABLE:
            if device == "auto":
                self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            else:
                self.device = _torch.device(device)
        else:
            self.device = device  # type: ignore[assignment]
        self.models_dir = models_dir
        self._loaded = False
        self.logger = structlog.get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abc.abstractmethod
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        """Match two images. Both images should be HxWxC uint8 or float32."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Normalize image to float32 [0,1]."""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.dtype in (np.uint16, np.int16):
            mn, mx = img.min(), img.max()
            return (img.astype(np.float32) - mn) / (mx - mn + 1e-8)
        return img.astype(np.float32)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert image to single-channel grayscale float32."""
        img_f = BaseMatcher._preprocess(img)
        if img_f.ndim == 2:
            return img_f
        if img_f.shape[2] == 1:
            return img_f[:, :, 0]
        # weighted RGB → gray
        return (0.2989 * img_f[:, :, 0] + 0.5870 * img_f[:, :, 1] + 0.1140 * img_f[:, :, 2])

    @staticmethod
    def _orb_bf_fallback(img0: np.ndarray, img1: np.ndarray, matcher_name: str) -> MatchResult:
        """ORB + BruteForce fallback matcher (OpenCV required)."""
        import cv2  # type: ignore[import]
        import time

        t0 = time.perf_counter()
        g0 = (BaseMatcher._to_gray(img0) * 255).astype(np.uint8)
        g1 = (BaseMatcher._to_gray(img1) * 255).astype(np.uint8)
        orb = cv2.ORB_create(nfeatures=2000)
        kp0, des0 = orb.detectAndCompute(g0, None)
        kp1, des1 = orb.detectAndCompute(g1, None)
        if des0 is None or des1 is None or len(kp0) == 0 or len(kp1) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=matcher_name)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)
        pts0 = np.array([kp0[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.float32)
        max_dist = max(m.distance for m in matches) + 1e-8
        conf = np.array([1.0 - m.distance / max_dist for m in matches], dtype=np.float32)
        elapsed = time.perf_counter() - t0
        return MatchResult(kpts0=pts0, kpts1=pts1, confidence=conf, matcher_name=matcher_name, processing_time_s=elapsed)
