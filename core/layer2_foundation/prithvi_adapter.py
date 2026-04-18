"""Prithvi EO 2.0 foundation model adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .base_adapter import BaseFoundationAdapter

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import AutoModel, AutoConfig, AutoProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


_PRITHVI_HF_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0"


class PrithviAdapter(BaseFoundationAdapter):
    """Adapter for IBM/NASA Prithvi EO 2.0 multi-temporal earth-observation ViT.

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    hf_model_id:
        HuggingFace model identifier.
    temporal_frames:
        Number of temporal frames expected by the model (default 1).
    """

    model_name = "prithvi_eo_2"
    embedding_dim = 768

    def __init__(
        self,
        device: str = "auto",
        hf_model_id: str = _PRITHVI_HF_ID,
        temporal_frames: int = 1,
    ) -> None:
        super().__init__(device=device)
        self.hf_model_id = hf_model_id
        self.temporal_frames = temporal_frames
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if not _HAS_TORCH:
            logger.warning("torch not available; Prithvi will use mock embeddings")
            return
        if not _HAS_TRANSFORMERS:
            logger.warning("transformers not available; Prithvi will use mock embeddings")
            return

        logger.info("loading Prithvi EO 2.0", model_id=self.hf_model_id)
        try:
            self._model = AutoModel.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            ).to(self.device).eval()

            # Try loading a processor / image processor
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.hf_model_id, trust_remote_code=True
                )
            except Exception:
                self._processor = None

            self.embedding_dim = self._model.config.hidden_size or 768
            logger.info("Prithvi loaded", embedding_dim=self.embedding_dim)
        except Exception as exc:
            logger.warning("Prithvi model load failed; using mock", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Return (H, W, D) dense embeddings."""
        self._ensure_loaded()

        if self._model is None or not _HAS_TORCH:
            return self._mock_embeddings(image)

        img = self._to_chw(image)   # (C, H, W)
        C, H, W = img.shape
        patch_size = 16

        try:
            tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
            # Prithvi expects temporal dimension: (B, T, C, H, W)
            if self.temporal_frames == 1:
                tensor = tensor.unsqueeze(1)  # (1, 1, C, H, W)

            with torch.no_grad():
                outputs = self._model(pixel_values=tensor, output_hidden_states=True)

            # Extract last hidden state patches → upsample to (H, W, D)
            hidden = outputs.last_hidden_state  # (B, num_patches, D)
            D = hidden.shape[-1]
            pH = H // patch_size
            pW = W // patch_size
            n_patches = pH * pW

            patch_embeds = hidden[0, :n_patches, :]   # (n_patches, D)
            grid = patch_embeds.reshape(pH, pW, D).permute(2, 0, 1).unsqueeze(0)  # (1, D, pH, pW)
            upsampled = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            result = upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, D)
            return result.astype(np.float32)

        except Exception as exc:
            logger.warning("Prithvi encode failed; using mock", error=str(exc))
            return self._mock_embeddings(image)

    # ------------------------------------------------------------------

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a (H, W) integer label mask via k-means on embeddings."""
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]

        embeddings = self.encode(image)  # (H, W, D)
        H, W, D = embeddings.shape
        n_clusters = 8
        flat = embeddings.reshape(-1, D).astype(np.float32)

        try:
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            labels = km.fit_predict(flat).reshape(H, W).astype(np.int32)
        except Exception as exc:
            logger.warning("k-means clustering failed; returning zeros", error=str(exc))
            labels = np.zeros((H, W), dtype=np.int32)

        return labels
