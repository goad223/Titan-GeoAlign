"""SAR-to-Optical diffusion model translator."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False

try:
    from scipy.ndimage import median_filter
    from scipy.stats import rankdata
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class SAR2OpticalDiffusion:
    """Translates SAR imagery to pseudo-optical using a diffusion model.

    When the model is unavailable (no GPU, no checkpoint) a classical
    SAR→pseudo-optical fallback is used:

    1. Convert SAR amplitude/dB to normalised power.
    2. Apply adaptive histogram equalisation (CLAHE via OpenCV or
       skimage).
    3. Apply a median filter for speckle suppression.
    4. Output as a 3-band pseudo-RGB array.

    Parameters
    ----------
    model_path:
        Path to a local HuggingFace diffusers directory or checkpoint.
        If *None*, uses the fallback classical method.
    device:
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._unet: Optional["UNet2DConditionModel"] = None
        self._scheduler = None
        self._device = self._resolve_device(device)
        logger.debug("SAR2OpticalDiffusion init", device=str(self._device))

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load diffusion model weights from *path* or ``self.model_path``."""
        target = path or self.model_path
        if target is None:
            logger.warning("No model path specified; SAR2Optical will use fallback")
            return
        if not _HAS_DIFFUSERS or not _HAS_TORCH:
            logger.warning("diffusers/torch not available; using fallback")
            return

        target = Path(target)
        logger.info("loading SAR→Optical diffusion model", path=str(target))
        try:
            self._unet = UNet2DConditionModel.from_pretrained(
                str(target), subfolder="unet"
            ).to(self._device)
            self._scheduler = DDIMScheduler.from_pretrained(
                str(target), subfolder="scheduler"
            )
            logger.info("SAR2Optical diffusion model loaded")
        except Exception as exc:
            logger.warning("diffusion model load failed; using fallback", error=str(exc))
            self._unet = None

    # ------------------------------------------------------------------

    def translate(self, sar_image: np.ndarray) -> np.ndarray:
        """Translate a SAR image to pseudo-optical.

        Parameters
        ----------
        sar_image:
            Input SAR array, shape ``(C, H, W)`` or ``(H, W, C)`` or
            ``(H, W)`` (single polarisation).  Values may be in dB scale
            or linear amplitude.

        Returns
        -------
        np.ndarray
            Pseudo-optical array ``(H, W, 3)`` uint8.
        """
        sar = self._normalise_sar(sar_image)   # (C, H, W), float32 [0,1]

        if self._unet is not None and _HAS_TORCH:
            try:
                return self._translate_diffusion(sar)
            except Exception as exc:
                logger.warning("diffusion translate failed; using fallback", error=str(exc))

        return self._translate_classical(sar)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_sar(arr: np.ndarray) -> np.ndarray:
        """Normalise SAR to ``(C, H, W)`` float32 in ``[0, 1]``."""
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = arr.transpose(2, 0, 1)
        arr = arr.astype(np.float32)

        # dB → linear if values mostly negative (typical dB range: -30 to 5)
        if arr.mean() < 0.0:
            arr = 10.0 ** (arr / 10.0)  # dB to power

        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            band = arr[i]
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            denom = (hi - lo) if hi != lo else 1.0
            out[i] = np.clip((band - lo) / denom, 0.0, 1.0)
        return out

    # ------------------------------------------------------------------
    # Diffusion translate
    # ------------------------------------------------------------------

    def _translate_diffusion(self, sar: np.ndarray) -> np.ndarray:
        """Run SAR through UNet2DConditionModel for optical generation."""
        import torch
        import torch.nn.functional as F

        C, H, W = sar.shape
        # Pad to multiple of 8 for UNet
        pH = ((H + 7) // 8) * 8
        pW = ((W + 7) // 8) * 8

        tensor = torch.from_numpy(sar).unsqueeze(0).to(self._device)
        if pH != H or pW != W:
            tensor = F.pad(tensor, (0, pW - W, 0, pH - H))

        # UNet expects 3 channels for unconditional generation
        if C != 3:
            tensor = tensor.expand(-1, 3, -1, -1) if C == 1 else tensor[:, :3]

        self._scheduler.set_timesteps(20)
        latents = torch.randn_like(tensor).to(self._device)

        # Simplified denoising loop (DDIM)
        for t in self._scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self._unet(
                    latents,
                    t,
                    encoder_hidden_states=tensor.reshape(1, C if C == 3 else 3, -1).permute(0, 2, 1),
                ).sample
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        result = latents.squeeze(0).permute(1, 2, 0).cpu().numpy()[:H, :W, :]
        result = np.clip(result, 0.0, 1.0)
        return (result * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Classical fallback
    # ------------------------------------------------------------------

    def _translate_classical(self, sar: np.ndarray) -> np.ndarray:
        """Classical SAR→pseudo-optical: CLAHE + median filter."""
        C, H, W = sar.shape

        # Use first (or only) two polarisations
        if C >= 2:
            vv = sar[0]
            vh = sar[1]
        else:
            vv = sar[0]
            vh = sar[0]

        # Compute ratio channel (VV/VH) for pseudo-NIR
        ratio = np.where(vh > 1e-6, vv / (vh + 1e-6), vv)
        ratio = np.clip(ratio, 0.0, 1.0)

        channels = [vv, vh, ratio]
        result_channels = []
        for ch in channels:
            ch = ch.astype(np.float32)
            # CLAHE via skimage or opencv
            enhanced = self._apply_clahe(ch)
            if _HAS_SCIPY:
                enhanced = median_filter(enhanced, size=3)
            result_channels.append(np.clip(enhanced, 0.0, 1.0))

        rgb = np.stack(result_channels, axis=-1)
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def _apply_clahe(band: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        try:
            import cv2
            img_u8 = (band * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img_u8).astype(np.float32) / 255.0
        except ImportError:
            pass

        try:
            from skimage.exposure import equalize_adapthist
            return equalize_adapthist(band.astype(np.float64), clip_limit=0.02).astype(np.float32)
        except ImportError:
            pass

        # Last resort: simple histogram equalisation
        flat = band.ravel()
        n = len(flat)
        sorted_vals = np.sort(flat)
        cdf = np.arange(1, n + 1) / n
        lookup = np.interp(flat, sorted_vals, cdf)
        return lookup.reshape(band.shape).astype(np.float32)
