"""SpectralGPT adapter for hyperspectral imagery."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .base_adapter import BaseFoundationAdapter

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from timm.models.vision_transformer import VisionTransformer
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

_SPECTRALGPT_HF_ID = "torchgeo/spectralgpt"


class SpectralGPTAdapter(BaseFoundationAdapter):
    """Adapter for SpectralGPT — a GPT-style encoder for hyperspectral imagery.

    SpectralGPT tokenizes imagery along the spectral dimension, treating each
    spectral band as a separate token.  This allows it to handle variable-band
    hyperspectral cubes without fixed channel assumptions.

    Parameters
    ----------
    device:
        Target device.
    hf_model_id:
        HuggingFace model identifier.
    patch_size:
        Spatial patch size (default 16).
    """

    model_name = "spectralgpt"
    embedding_dim = 768

    def __init__(
        self,
        device: str = "auto",
        hf_model_id: str = _SPECTRALGPT_HF_ID,
        patch_size: int = 16,
    ) -> None:
        super().__init__(device=device)
        self.hf_model_id = hf_model_id
        self.patch_size = patch_size
        self._model = None

    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if not (_HAS_TORCH and _HAS_TIMM):
            logger.warning("torch/timm unavailable; SpectralGPT uses mock embeddings")
            return

        logger.info("loading SpectralGPT", model_id=self.hf_model_id)
        try:
            # Try HuggingFace
            try:
                from transformers import AutoModel, AutoConfig
                config = AutoConfig.from_pretrained(
                    self.hf_model_id, trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    self.hf_model_id, config=config, trust_remote_code=True
                ).to(self.device).eval()
                self.embedding_dim = getattr(config, "hidden_size", 768)
                logger.info("SpectralGPT loaded from HuggingFace")
                return
            except Exception as hf_exc:
                logger.debug("HF load failed; building SpectralGPT from timm",
                             error=str(hf_exc))

            # Fallback: ViT with 1-channel input (process bands sequentially)
            self._model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, in_chans=1
            ).to(self.device).eval()
            self.embedding_dim = 768
            self._spectral_mode = "sequential"
            logger.info("SpectralGPT loaded (timm fallback, sequential spectral mode)")

        except Exception as exc:
            logger.warning("SpectralGPT load failed; using mock", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W, D)`` dense embeddings.

        For hyperspectral data, processes each band independently and
        averages the per-band embeddings.
        """
        self._ensure_loaded()

        img = self._to_chw(image)  # (C, H, W)
        C, H, W = img.shape

        if self._model is None or not _HAS_TORCH:
            return self._mock_embeddings(image)

        # SpectralGPT spectral tokenisation strategy:
        # Process each band independently through the ViT and aggregate
        try:
            band_embeddings = []
            for b in range(C):
                band_img = img[[b], :, :]  # (1, H, W) → single-band
                emb = self._encode_single_band(band_img, H, W)
                band_embeddings.append(emb)

            # Aggregate: mean over spectral bands
            stacked = np.stack(band_embeddings, axis=0)  # (C, H, W, D)
            return stacked.mean(axis=0).astype(np.float32)   # (H, W, D)

        except Exception as exc:
            logger.warning("SpectralGPT encode failed; using mock", error=str(exc))
            return self._mock_embeddings(image)

    def _encode_single_band(self, band: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
        """Encode a single (1, H, W) band and return (H, W, D) embeddings."""
        import torch

        tensor = torch.from_numpy(band).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        if orig_H != 224 or orig_W != 224:
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            features = self._model.forward_features(tensor)  # (1, n_patches+1, D)

        D = features.shape[-1]
        pH = pW = 224 // self.patch_size
        patch_tokens = features[0, 1:pH * pW + 1, :]

        grid = patch_tokens.reshape(pH, pW, D).permute(2, 0, 1).unsqueeze(0)
        upsampled = F.interpolate(grid, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
        return upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` semantic mask via k-means on embeddings."""
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]

        embeddings = self.encode(image)
        H, W, D = embeddings.shape
        flat = embeddings.reshape(-1, D).astype(np.float32)
        try:
            km = MiniBatchKMeans(n_clusters=16, random_state=0, n_init=3)
            labels = km.fit_predict(flat).reshape(H, W).astype(np.int32)
        except Exception as exc:
            logger.warning("k-means failed", error=str(exc))
            labels = np.zeros((H, W), dtype=np.int32)
        return labels
