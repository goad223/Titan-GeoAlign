"""Ensemble matching module."""
from .voter import EnsembleVoter
from .confidence_fusion import ConfidenceWeightedFusion
from .matcher_ensemble import MatcherEnsemble

__all__ = ["EnsembleVoter", "ConfidenceWeightedFusion", "MatcherEnsemble"]
