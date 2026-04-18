"""Monte-Carlo Dropout sampler for uncertainty estimation."""
from __future__ import annotations

from typing import List

import numpy as np
import structlog

from core.layer4_matchers.base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class MCDropoutSampler:
    """Estimate positional uncertainty via repeated stochastic forward passes."""

    def __init__(self, n_samples: int = 10) -> None:
        self.n_samples = n_samples

    # ------------------------------------------------------------------
    def sample(
        self,
        matcher: BaseMatcher,
        img0: np.ndarray,
        img1: np.ndarray,
    ) -> List[MatchResult]:
        """Run the matcher n_samples times with stochastic perturbations.

        For neural matchers: calls match() repeatedly (assumes dropout is enabled
        at inference via model.train() mode or explicit dropout layers).
        For classical matchers: adds Gaussian noise to images before matching.
        """
        samples: List[MatchResult] = []
        is_neural = _is_neural_matcher(matcher)

        for i in range(self.n_samples):
            if is_neural:
                _enable_dropout(matcher)
                result = matcher.match(img0, img1)
            else:
                noise_scale = 2.0  # std dev in pixel intensity
                noisy0 = _add_gaussian_noise(img0, noise_scale)
                noisy1 = _add_gaussian_noise(img1, noise_scale)
                result = matcher.match(noisy0, noisy1)
            samples.append(result)
            logger.debug("mc_sample", iteration=i + 1, matches=result.num_matches)

        return samples

    # ------------------------------------------------------------------
    def compute_positional_uncertainty(
        self, samples: List[MatchResult]
    ) -> np.ndarray:
        """Compute per-keypoint positional std dev across MC samples.

        Returns [N, 2] array of std devs for the first sample's keypoints
        (aligned by nearest neighbour across samples).
        """
        if not samples:
            return np.zeros((0, 2), dtype=np.float32)

        reference = samples[0]
        n_ref = reference.num_matches
        if n_ref == 0:
            return np.zeros((0, 2), dtype=np.float32)

        kpts0_stack = np.zeros((self.n_samples, n_ref, 2), dtype=np.float32)
        kpts0_stack[0] = reference.kpts0

        for s_idx, sample in enumerate(samples[1:], start=1):
            if sample.num_matches == 0:
                kpts0_stack[s_idx] = reference.kpts0
                continue
            # Nearest-neighbour alignment
            for i in range(n_ref):
                dists = np.linalg.norm(sample.kpts0 - reference.kpts0[i], axis=1)
                best = int(np.argmin(dists))
                kpts0_stack[s_idx, i] = sample.kpts0[best]

        std_dev = kpts0_stack.std(axis=0)  # [N, 2]
        return std_dev.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_neural_matcher(matcher: BaseMatcher) -> bool:
    try:
        import torch
        for module in vars(matcher).values():
            if isinstance(module, torch.nn.Module):
                return True
    except ImportError:
        pass
    return False


def _enable_dropout(matcher: BaseMatcher) -> None:
    """Set all Dropout layers to training mode for stochastic inference."""
    try:
        import torch
        for module in vars(matcher).values():
            if isinstance(module, torch.nn.Module):
                for layer in module.modules():
                    if isinstance(layer, torch.nn.Dropout):
                        layer.train()
    except ImportError:
        pass


def _add_gaussian_noise(img: np.ndarray, std: float = 2.0) -> np.ndarray:
    """Add zero-mean Gaussian noise and clip to valid range."""
    noisy = img.astype(np.float32) + np.random.normal(0, std, img.shape).astype(np.float32)
    if img.dtype == np.uint8:
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)
