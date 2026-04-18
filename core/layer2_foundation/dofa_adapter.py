"""DOFA (Dynamic One-For-All) unified encoder adapter."""
from __future__ import annotations

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
    from transformers import AutoModel, AutoConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

_DOFA_HF_ID = "xSkywalker/DOFA"


class DOFAAdapter(BaseFoundationAdapter):
    """Adapter for DOFA (Dynamic One-For-All) unified EO encoder.

    DOFA uses wavelength-conditioned dynamic weight generation to handle
    arbitrary numbers of input bands (optical, SAR, hyperspectral).

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    hf_model_id:
        HuggingFace model identifier.
    """

    model_name = "dofa"
    embedding_dim = 768

    def __init__(self, device: str = "auto", hf_model_id: str = _DOFA_HF_ID) -> None:
        super().__init__(device=device)
        self.hf_model_id = hf_model_id
        self._model = None

    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if not (_HAS_TORCH and _HAS_TRANSFORMERS):
            logger.warning("torch/transformers unavailable; DOFA uses mock embeddings")
            return

        logger.info("loading DOFA", model_id=self.hf_model_id)
        try:
            config = AutoConfig.from_pretrained(
                self.hf_model_id, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.hf_model_id, config=config, trust_remote_code=True
            ).to(self.device).eval()
            self.embedding_dim = getattr(config, "hidden_size", 768)
            logger.info("DOFA loaded", embedding_dim=self.embedding_dim)
        except Exception as exc:
            logger.warning("DOFA load failed; using mock", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray, wavelengths: list[float] | None = None) -> np.ndarray:
        """Return ``(H, W, D)`` dense embeddings.

        Parameters
        ----------
        image:
            ``(C, H, W)`` or ``(H, W, C)`` float32 input image.
        wavelengths:
            Central wavelength in nanometres for each band.  If *None*,
            evenly-spaced values between 400 and 2500 nm are used.
        """
        self._ensure_loaded()

        img = self._to_chw(image)  # (C, H, W)
        C, H, W = img.shape
        patch_size = 16

        if wavelengths is None:
            wavelengths = list(np.linspace(400.0, 2500.0, C).tolist())

        if self._model is None or not _HAS_TORCH:
            return self._mock_embeddings(image)

        try:
            tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
            wl_tensor = torch.tensor(wavelengths, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self._model(
                    pixel_values=tensor,
                    wavelengths=wl_tensor,
                    output_hidden_states=True,
                )

            hidden = outputs.last_hidden_state
            D = hidden.shape[-1]
            pH = H // patch_size
            pW = W // patch_size
            patch_seq = hidden[0, 1:pH * pW + 1, :]

            grid = patch_seq.reshape(pH, pW, D).permute(2, 0, 1).unsqueeze(0)
            upsampled = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            return upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        except Exception as exc:
            logger.warning("DOFA encode failed; using mock", error=str(exc))
            return self._mock_embeddings(image)

    # ------------------------------------------------------------------

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` semantic mask via k-means on embeddings."""
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]

        embeddings = self.encode(image)
        H, W, D = embeddings.shape
        flat = embeddings.reshape(-1, D).astype(np.float32)
        try:
            km = MiniBatchKMeans(n_clusters=12, random_state=0, n_init=3)
            labels = km.fit_predict(flat).reshape(H, W).astype(np.int32)
        except Exception as exc:
            logger.warning("k-means failed", error=str(exc))
            labels = np.zeros((H, W), dtype=np.int32)
        return labels
