"""Matcher factory — instantiate any registered matcher by name."""
from __future__ import annotations

from typing import Any

import structlog

from .base_matcher import BaseMatcher

logger = structlog.get_logger(__name__)

_REGISTRY: dict[str, type] = {}


def _register() -> None:
    global _REGISTRY
    from .roma_matcher import RoMaV2Matcher
    from .mast3r_matcher import MASt3RMatcher
    from .gim_matcher import GIMMatcher
    from .xfeat_lightglue_matcher import XFeatLightGlueMatcher
    from .xoftr_matcher import XoFTRMatcher
    from .rift2_matcher import RIFT2Matcher
    from .omniglue_matcher import OmniGlueMatcher

    _REGISTRY = {
        "roma_v2": RoMaV2Matcher,
        "mast3r": MASt3RMatcher,
        "gim_dkmv3": GIMMatcher,
        "xfeat_lightglue": XFeatLightGlueMatcher,
        "xoftr": XoFTRMatcher,
        "rift2": RIFT2Matcher,
        "omniglue": OmniGlueMatcher,
    }


def get_matcher(name: str, **kwargs: Any) -> BaseMatcher:
    """Instantiate a matcher by name.

    Parameters
    ----------
    name:
        One of ``"roma_v2"``, ``"mast3r"``, ``"gim_dkmv3"``,
        ``"xfeat_lightglue"``, ``"xoftr"``, ``"rift2"``, ``"omniglue"``.
    **kwargs:
        Forwarded to the matcher constructor (e.g. ``device``, ``models_dir``).

    Returns
    -------
    BaseMatcher
        An **unloaded** matcher instance; call ``.load()`` before ``.match()``.
    """
    if not _REGISTRY:
        _register()

    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown matcher '{name}'. Available: {available}")

    matcher = _REGISTRY[name](**kwargs)
    logger.info("Matcher instantiated", name=name, backend_class=matcher.__class__.__name__)
    return matcher


def list_matchers() -> list[str]:
    """Return list of all registered matcher names."""
    if not _REGISTRY:
        _register()
    return list(_REGISTRY.keys())
