"""Foundation model factory — returns the appropriate adapter by name."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from .base_adapter import BaseFoundationAdapter

logger = structlog.get_logger(__name__)

_DEFAULT_MODELS_DIR = Path("models/foundation")

# Supported model names
SUPPORTED_MODELS = {
    "prithvi_eo_2",
    "clay_v1_5",
    "dofa",
    "satmae_pp",
    "spectralgpt",
}


def get_foundation_model(
    name: str,
    device: str = "auto",
    models_dir: Path = _DEFAULT_MODELS_DIR,
) -> BaseFoundationAdapter:
    """Instantiate and return a foundation model adapter.

    Parameters
    ----------
    name:
        Model name.  One of: ``"prithvi_eo_2"``, ``"clay_v1_5"``,
        ``"dofa"``, ``"satmae_pp"``, ``"spectralgpt"``.
    device:
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
    models_dir:
        Base directory for locally stored model checkpoints.

    Returns
    -------
    BaseFoundationAdapter
        Adapter instance (model is *not* loaded yet; call
        :meth:`~BaseFoundationAdapter.load_model` explicitly or let the
        adapter lazy-load on first use).

    Raises
    ------
    ValueError
        If *name* is not a supported model identifier.
    """
    name = name.lower().strip()
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown foundation model {name!r}. "
            f"Supported: {sorted(SUPPORTED_MODELS)}"
        )

    models_dir = Path(models_dir)
    logger.info("creating foundation model adapter", model=name, device=device)

    if name == "prithvi_eo_2":
        from .prithvi_adapter import PrithviAdapter
        return PrithviAdapter(device=device)

    if name == "clay_v1_5":
        from .clay_adapter import ClayAdapter
        return ClayAdapter(device=device)

    if name == "dofa":
        from .dofa_adapter import DOFAAdapter
        return DOFAAdapter(device=device)

    if name == "satmae_pp":
        from .satmae_adapter import SatMAEAdapter
        checkpoint = models_dir / "satmae_pp.pth"
        return SatMAEAdapter(device=device, checkpoint_path=checkpoint)

    if name == "spectralgpt":
        from .spectralgpt_adapter import SpectralGPTAdapter
        return SpectralGPTAdapter(device=device)

    # This line is unreachable due to the earlier check, but kept for safety
    raise ValueError(f"Unhandled model name: {name!r}")  # pragma: no cover
