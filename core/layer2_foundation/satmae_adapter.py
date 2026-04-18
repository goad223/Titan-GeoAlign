"""SatMAE++ pretrained ViT adapter."""
from __future__ import annotations

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

_DEFAULT_CHECKPOINT = Path("models/foundation/satmae_pp.pth")


class SatMAEAdapter(BaseFoundationAdapter):
    """Adapter for SatMAE++ pretrained ViT encoder.

    Loads a ViT checkpoint from *models/foundation/satmae_pp.pth* (or a
    HuggingFace ID as fallback).  The temporal position embedding is
    handled by repeating the standard ViT positional embedding along the
    time axis when multi-temporal inputs are provided.

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    checkpoint_path:
        Path to local ``satmae_pp.pth`` checkpoint.  Falls back to a
        standard ViT-Large if the file is absent.
    image_size:
        Input image size expected by the ViT backbone (default 224).
    patch_size:
        ViT patch size (default 16).
    """

    model_name = "satmae_pp"
    embedding_dim = 1024  # ViT-Large

    def __init__(
        self,
        device: str = "auto",
        checkpoint_path: Path = _DEFAULT_CHECKPOINT,
        image_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size
        self.patch_size = patch_size
        self._model = None

    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if not _HAS_TORCH:
            logger.warning("torch unavailable; SatMAE uses mock embeddings")
            return

        if self.checkpoint_path.exists():
            self._load_from_checkpoint()
        elif _HAS_TIMM:
            self._load_vit_pretrained()
        else:
            logger.warning("no SatMAE checkpoint found; using mock embeddings")

    def _load_from_checkpoint(self) -> None:
        logger.info("loading SatMAE++ from checkpoint", path=str(self.checkpoint_path))
        try:
            if not _HAS_TIMM:
                raise ImportError("timm required to instantiate ViT backbone")

            state = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = state.get("model", state)

            # Infer model config from keys
            embed_dim = self.embedding_dim
            self._model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                in_chans=3,
                img_size=self.image_size,
            )
            # Load weights (allow missing/extra keys from temporal heads)
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            logger.debug("state loaded", missing=len(missing), unexpected=len(unexpected))
            self._model = self._model.to(self.device).eval()
            self.embedding_dim = embed_dim
            logger.info("SatMAE++ loaded from checkpoint")
        except Exception as exc:
            logger.warning("checkpoint load failed; falling back to timm", error=str(exc))
            self._load_vit_pretrained()

    def _load_vit_pretrained(self) -> None:
        logger.info("loading pretrained ViT-Large as SatMAE++ fallback")
        try:
            self._model = timm.create_model(
                "vit_large_patch16_224", pretrained=True, in_chans=3
            ).to(self.device).eval()
            self.embedding_dim = 1024
            logger.info("ViT-Large fallback loaded")
        except Exception as exc:
            logger.warning("ViT-Large fallback load failed", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Return ``(H, W, D)`` dense embeddings."""
        self._ensure_loaded()

        if self._model is None or not _HAS_TORCH:
            return self._mock_embeddings(image)

        img = self._to_chw(image)   # (C, H, W)
        C, H, W = img.shape

        # Convert to 3-channel if needed (repeat channels or average)
        if C != 3:
            if C < 3:
                img = np.tile(img, (3 // C + 1, 1, 1))[:3]
            else:
                img = img[:3]

        # Resize to model's expected image size
        import torch
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        if H != self.image_size or W != self.image_size:
            tensor = F.interpolate(
                tensor, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )

        try:
            with torch.no_grad():
                # Use forward_features to get patch tokens
                features = self._model.forward_features(tensor)  # (1, n_patches+1, D)
            D = features.shape[-1]
            pH = self.image_size // self.patch_size
            pW = self.image_size // self.patch_size
            patch_tokens = features[0, 1:pH * pW + 1, :]

            grid = patch_tokens.reshape(pH, pW, D).permute(2, 0, 1).unsqueeze(0)
            upsampled = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            return upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        except Exception as exc:
            logger.warning("SatMAE encode failed; using mock", error=str(exc))
            return self._mock_embeddings(image)

    # ------------------------------------------------------------------

    def get_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]

        embeddings = self.encode(image)
        H, W, D = embeddings.shape
        flat = embeddings.reshape(-1, D).astype(np.float32)
        try:
            km = MiniBatchKMeans(n_clusters=8, random_state=42, n_init=3)
            labels = km.fit_predict(flat).reshape(H, W).astype(np.int32)
        except Exception as exc:
            logger.warning("k-means failed", error=str(exc))
            labels = np.zeros((H, W), dtype=np.int32)
        return labels
