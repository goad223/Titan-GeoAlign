"""Layer 4 — Multi-Matcher Ensemble."""
from .base_matcher import BaseMatcher, MatchResult
from .roma_matcher import RoMaV2Matcher
from .mast3r_matcher import MASt3RMatcher
from .gim_matcher import GIMMatcher
from .xfeat_lightglue_matcher import XFeatLightGlueMatcher
from .xoftr_matcher import XoFTRMatcher
from .rift2_matcher import RIFT2Matcher
from .omniglue_matcher import OmniGlueMatcher
from .matcher_factory import get_matcher, list_matchers

__all__ = [
    "BaseMatcher",
    "MatchResult",
    "RoMaV2Matcher",
    "MASt3RMatcher",
    "GIMMatcher",
    "XFeatLightGlueMatcher",
    "XoFTRMatcher",
    "RIFT2Matcher",
    "OmniGlueMatcher",
    "get_matcher",
    "list_matchers",
]
