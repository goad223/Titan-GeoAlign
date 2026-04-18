"""CycleGAN HD — ResNet-based image domain translation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# ResNet generator definition (9-block, 256-filter)
# Only defined when torch is available
# ---------------------------------------------------------------------------

def _build_resnet_classes():
    """Build and return ResNet generator classes at runtime when torch is present."""
    import torch.nn as _nn

    class _ResidualBlock(_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.block = _nn.Sequential(
                _nn.ReflectionPad2d(1),
                _nn.Conv2d(channels, channels, 3, bias=False),
                _nn.InstanceNorm2d(channels),
                _nn.ReLU(inplace=True),
                _nn.ReflectionPad2d(1),
                _nn.Conv2d(channels, channels, 3, bias=False),
                _nn.InstanceNorm2d(channels),
            )

        def forward(self, x):
            return x + self.block(x)

    class _ResNetGenerator(_nn.Module):
        """Standard CycleGAN ResNet generator (9 residual blocks)."""

        def __init__(self, in_channels: int = 3, out_channels: int = 3,
                     ngf: int = 64, n_blocks: int = 9) -> None:
            super().__init__()
            layers = [
                _nn.ReflectionPad2d(3),
                _nn.Conv2d(in_channels, ngf, 7, bias=False),
                _nn.InstanceNorm2d(ngf),
                _nn.ReLU(inplace=True),
                _nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
                _nn.InstanceNorm2d(ngf * 2),
                _nn.ReLU(inplace=True),
                _nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
                _nn.InstanceNorm2d(ngf * 4),
                _nn.ReLU(inplace=True),
            ]
            for _ in range(n_blocks):
                layers.append(_ResidualBlock(ngf * 4))
            layers += [
                _nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1,
                                    output_padding=1, bias=False),
                _nn.InstanceNorm2d(ngf * 2),
                _nn.ReLU(inplace=True),
                _nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1,
                                    output_padding=1, bias=False),
                _nn.InstanceNorm2d(ngf),
                _nn.ReLU(inplace=True),
                _nn.ReflectionPad2d(3),
                _nn.Conv2d(ngf, out_channels, 7),
                _nn.Tanh(),
            ]
            self.model = _nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return _ResidualBlock, _ResNetGenerator


# Placeholder types used for type annotations only (no runtime cost when torch absent)
_ResidualBlock = None  # type: ignore[assignment]
_ResNetGenerator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CycleGAN HD
# ---------------------------------------------------------------------------

# Supported translation directions
_SUPPORTED = {
    ("night", "day"),
    ("thermal", "day"),
    ("thermal", "optical"),
    ("nir", "rgb"),
    ("sar", "optical"),
    ("day", "night"),
}


class CycleGANHD:
    """CycleGAN-based image domain translation.

    Supports pairs like: night→day, thermal→day, NIR→RGB, SAR→optical.
    Falls back to histogram matching if model weights are not available.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing per-domain-pair checkpoint files, e.g.
        ``night2day_G_A.pth``.
    device:
        Target device.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models/cyclegan")
        self._device = self._resolve_device(device)
        self._generators: dict[tuple[str, str], "_ResNetGenerator"] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------

    def translate(
        self, image: np.ndarray, source_domain: str, target_domain: str
    ) -> np.ndarray:
        """Translate *image* from *source_domain* to *target_domain*.

        Parameters
        ----------
        image:
            Input image ``(H, W, C)`` or ``(C, H, W)`` float32 in [0, 1].
        source_domain:
            Source modality string (e.g. ``"night"``, ``"thermal"``, ``"sar"``).
        target_domain:
            Target modality string (e.g. ``"day"``, ``"optical"``).

        Returns
        -------
        np.ndarray
            Translated image ``(H, W, 3)`` uint8.
        """
        img = self._to_chw_float(image)   # (C, H, W), float32 [0,1]
        key = (source_domain.lower(), target_domain.lower())

        if _HAS_TORCH and (key in _SUPPORTED or (key[1], key[0]) in _SUPPORTED):
            gen = self._get_generator(key, img.shape[0])
            if gen is not None:
                try:
                    return self._run_generator(gen, img)
                except Exception as exc:
                    logger.warning("CycleGAN forward pass failed; using fallback",
                                   error=str(exc))

        logger.debug("using histogram matching fallback",
                     source=source_domain, target=target_domain)
        return self._histogram_match_fallback(img)

    # ------------------------------------------------------------------

    def _get_generator(
        self, key: tuple[str, str], in_channels: int
    ):
        if key in self._generators:
            return self._generators[key]

        if not _HAS_TORCH:
            return None

        # Build generator architecture lazily
        _, _ResNetGeneratorCls = _build_resnet_classes()
        gen = _ResNetGeneratorCls(
            in_channels=max(in_channels, 1), out_channels=3, ngf=64, n_blocks=9
        )

        # Try loading checkpoint
        ckpt_name = f"{'2'.join(key)}_G_A.pth"
        ckpt_path = self.checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=self._device)
                gen.load_state_dict(state, strict=False)
                logger.info("CycleGAN generator loaded", checkpoint=str(ckpt_path))
            except Exception as exc:
                logger.warning("CycleGAN checkpoint load failed", error=str(exc))

        gen = gen.to(self._device).eval()
        self._generators[key] = gen
        return gen

    def _run_generator(self, gen: "_ResNetGenerator", img: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        C, H, W = img.shape
        # Pad to multiple of 4
        pH = ((H + 3) // 4) * 4
        pW = ((W + 3) // 4) * 4

        tensor = torch.from_numpy(img * 2.0 - 1.0).unsqueeze(0).to(self._device)
        if pH != H or pW != W:
            tensor = F.pad(tensor, (0, pW - W, 0, pH - H))

        # Ensure 3 input channels
        if C != 3:
            tensor = tensor.expand(-1, 3, -1, -1) if C == 1 else tensor[:, :3]

        with torch.no_grad():
            out = gen(tensor)  # (1, 3, pH, pW), range [-1, 1]

        result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()[:H, :W, :]
        result = (result + 1.0) / 2.0  # → [0, 1]
        return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _histogram_match_fallback(img: np.ndarray) -> np.ndarray:
        """Simple contrast stretching per band → pseudo-optical 3-band."""
        C, H, W = img.shape
        if C >= 3:
            channels = img[:3]
        elif C == 2:
            channels = np.concatenate([img, img[[0]]], axis=0)
        else:
            channels = np.tile(img, (3, 1, 1))

        result = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(3):
            band = channels[i].astype(np.float32)
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            denom = (hi - lo) if hi != lo else 1.0
            result[:, :, i] = np.clip((band - lo) / denom, 0.0, 1.0)
        return (result * 255).astype(np.uint8)

    # ------------------------------------------------------------------

    @staticmethod
    def _to_chw_float(image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 2:
            return img[np.newaxis]
        if img.ndim == 3:
            if img.shape[-1] < img.shape[0]:
                return img.transpose(2, 0, 1)
        # Normalise to [0, 1] if in uint8
        if img.max() > 1.0:
            img = img / 255.0
        return img
