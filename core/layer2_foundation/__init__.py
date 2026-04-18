"""Layer 2 — AI Foundation Models."""
from .base_adapter import BaseFoundationAdapter
from .model_factory import get_foundation_model
from .semantic_mask import SemanticMask

__all__ = [
    "BaseFoundationAdapter",
    "get_foundation_model",
    "SemanticMask",
]
