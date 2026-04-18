"""Model type enum for geometric estimation."""
from __future__ import annotations

from enum import Enum, auto


class ModelType(Enum):
    """Geometric model types for robust estimation."""

    HOMOGRAPHY = auto()
    AFFINE = auto()
    AFFINE_PARTIAL = auto()
    SIMILARITY = auto()
    TPS = auto()
    POLYNOMIAL = auto()
    PIECEWISE = auto()
