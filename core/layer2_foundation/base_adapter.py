"""Abstract base adapter for geospatial foundation models."""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _resolve_device(device: str = "auto") -> "torch.device":
    """Resolve a device string to a torch.device."""
    if not _HAS_TORCH:
        raise ImportError("torch is required for foundation model adapters")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class BaseFoundationAdapter(abc.ABC):
    """Abstract interface for geospatial foundation model adapters.

    Sub-classes must implement :meth:`load_model`, :meth:`encode`, and
    :meth:`get_semantic_mask`.  All operations fall back to CPU-safe
    implementations when a GPU is not available.
    """

    model_name: str = "base"
    embedding_dim: int = 768

    def __init__(self, device: str = "auto") -> None:
        if _HAS_TORCH:
            self.device: "torch.device" = _resolve_device(device)
        else:
            self.device = None  # type: ignore[assignment]
        self._model_loaded: bool = False
        logger.debug("adapter initialised", model=self.model_name, device=str(self.device))

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_model(self) -> None:
        """Download/load model weights into memory."""

    @abc.abstractmethod
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Compute dense patch embeddings for *image*.

        Parameters
        ----------
        image:
            Input image of shape ``(C, H, W)`` or ``(H, W, C)``, float32.

        Returns
        -------
        np.ndarray
            Embedding array of shape ``(H, W, D)`` where D = embedding_dim.
        """

    @abc.abstractmethod
    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a semantic category mask ``(H, W)`` integer array."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_stable_regions(self, image: np.ndarray) -> np.ndarray:
        """Return a binary stability mask (1 = stable, 0 = unstable).

        Stable regions are those with low temporal variance in embeddings
        (e.g. buildings, roads, bare rock).  Uses spectral variance as a
        proxy when temporal data is not available.

        Parameters
        ----------
        image:
            ``(C, H, W)`` or ``(H, W, C)`` float32 input image.

        Returns
        -------
        np.ndarray
            Binary ``(H, W)`` uint8 mask.
        """
        # Ensure (C, H, W)
        if image.ndim == 3 and image.shape[2] < image.shape[0]:
            image = image.transpose(2, 0, 1)

        # Spectral standard deviation — low STD → spectrally homogeneous → stable
        std_map = np.std(image.astype(np.float32), axis=0)  # (H, W)
        threshold = float(np.percentile(std_map, 40))
        stable = (std_map < threshold).astype(np.uint8)
        return stable

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True

    @staticmethod
    def _to_chw(image: np.ndarray) -> np.ndarray:
        """Ensure image is ``(C, H, W)`` float32."""
        if image.ndim == 2:
            image = image[np.newaxis]
        elif image.ndim == 3 and image.shape[-1] < image.shape[0]:
            image = image.transpose(2, 0, 1)
        return image.astype(np.float32)

    def _mock_embeddings(self, image: np.ndarray) -> np.ndarray:
        """Generate deterministic mock embeddings (CPU fallback)."""
        img = self._to_chw(image)
        _, H, W = img.shape
        patch_size = 16
        pH = (H + patch_size - 1) // patch_size
        pW = (W + patch_size - 1) // patch_size

        # Mean pooling per patch → upsampled to (H, W, D)
        rng = np.random.default_rng(seed=42)
        patches = rng.standard_normal((pH, pW, self.embedding_dim)).astype(np.float32)

        # Nearest-neighbour upsample
        repeat_H = (H + pH - 1) // pH
        repeat_W = (W + pW - 1) // pW
        upsampled = np.repeat(np.repeat(patches, repeat_H, axis=0), repeat_W, axis=1)
        return upsampled[:H, :W, :]
