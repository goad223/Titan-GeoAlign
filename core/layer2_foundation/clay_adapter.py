"""Clay v1.5 foundation model adapter — multi-modal EO encoder."""
from __future__ import annotations

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
    from transformers import AutoModel, AutoConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

_CLAY_HF_ID = "made-with-clay/Clay-v1-5"


class ClayAdapter(BaseFoundationAdapter):
    """Adapter for Clay v1.5 multi-modal earth-observation encoder.

    Clay accepts optical, SAR, and thermal imagery with arbitrary band
    configurations.  Each input patch is processed with a per-modality
    projection head before entering the shared ViT backbone.

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    hf_model_id:
        HuggingFace model identifier.
    """

    model_name = "clay_v1_5"
    embedding_dim = 768

    def __init__(self, device: str = "auto", hf_model_id: str = _CLAY_HF_ID) -> None:
        super().__init__(device=device)
        self.hf_model_id = hf_model_id
        self._model = None

    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if not (_HAS_TORCH and _HAS_TRANSFORMERS):
            logger.warning("torch/transformers not available; Clay uses mock embeddings")
            return

        logger.info("loading Clay v1.5", model_id=self.hf_model_id)
        try:
            config = AutoConfig.from_pretrained(
                self.hf_model_id, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.hf_model_id, config=config, trust_remote_code=True
            ).to(self.device).eval()
            self.embedding_dim = getattr(config, "hidden_size", 768)
            logger.info("Clay loaded", embedding_dim=self.embedding_dim)
        except Exception as exc:
            logger.warning("Clay load failed; using mock", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W, D)`` dense embeddings."""
        self._ensure_loaded()

        if self._model is None or not _HAS_TORCH:
            return self._mock_embeddings(image)

        img = self._to_chw(image)  # (C, H, W)
        C, H, W = img.shape
        patch_size = 16

        try:
            tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
            with torch.no_grad():
                outputs = self._model(pixel_values=tensor, output_hidden_states=True)

            hidden = outputs.last_hidden_state  # (B, n_patches+cls, D)
            D = hidden.shape[-1]
            pH = H // patch_size
            pW = W // patch_size
            patch_seq = hidden[0, 1:pH * pW + 1, :]  # skip CLS token

            grid = patch_seq.reshape(pH, pW, D).permute(2, 0, 1).unsqueeze(0)
            upsampled = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            return upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        except Exception as exc:
            logger.warning("Clay encode failed; using mock", error=str(exc))
            return self._mock_embeddings(image)

    # ------------------------------------------------------------------

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` integer semantic label mask via k-means."""
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]

        embeddings = self.encode(image)
        H, W, D = embeddings.shape
        flat = embeddings.reshape(-1, D).astype(np.float32)
        try:
            km = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=3)
            labels = km.fit_predict(flat).reshape(H, W).astype(np.int32)
        except Exception as exc:
            logger.warning("k-means failed", error=str(exc))
            labels = np.zeros((H, W), dtype=np.int32)
        return labels
