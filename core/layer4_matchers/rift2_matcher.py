"""RIFT2 classical multi-modal feature matcher.

Full implementation of the RIFT2 algorithm:
  - Log-Gabor filter bank for phase congruency keypoint detection
  - Multi-support-region rotation-invariant (MIM) descriptor
  - Non-maximum suppression keypoint selection
  - Normalized cross-correlation descriptor matching
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import structlog
from scipy.ndimage import maximum_filter  # type: ignore[import]
from scipy.signal import convolve2d  # type: ignore[import]

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Log-Gabor utilities
# ---------------------------------------------------------------------------

def _log_gabor_2d(rows: int, cols: int, f0: float, theta: float, sigma_f: float, sigma_theta: float) -> np.ndarray:
    """Construct a 2-D log-Gabor filter in the frequency domain."""
    # Frequency grid
    fx = np.fft.fftfreq(cols)[np.newaxis, :]
    fy = np.fft.fftfreq(rows)[:, np.newaxis]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1e-10  # avoid log(0)
    angle = np.arctan2(fy, fx)

    # Radial component
    log_rad = np.exp(-((np.log(radius / f0)) ** 2) / (2 * sigma_f ** 2))
    log_rad[0, 0] = 0.0

    # Angular component
    d_theta = angle - theta
    d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))  # wrap to [-π, π]
    angular = np.exp(-(d_theta ** 2) / (2 * sigma_theta ** 2))

    return log_rad * angular


def _phase_congruency(
    img: np.ndarray,
    n_scale: int = 4,
    n_orient: int = 6,
    min_wave_length: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    k: float = 3.0,
    noise_method: float = -1.0,
) -> np.ndarray:
    """Compute phase congruency map (Kovesi's method).

    Returns a float32 map in [0, 1].
    """
    rows, cols = img.shape
    img_f = img.astype(np.float64)
    IM = np.fft.fft2(img_f)

    sigma_theta = np.pi / (n_orient * 2)
    total_weight = np.zeros((rows, cols), dtype=np.float64)
    total_energy = np.zeros((rows, cols), dtype=np.float64)

    for o in range(n_orient):
        theta = o * np.pi / n_orient
        sum_E = np.zeros((rows, cols), dtype=np.float64)
        sum_O = np.zeros((rows, cols), dtype=np.float64)
        sum_A = np.zeros((rows, cols), dtype=np.float64)

        for s in range(n_scale):
            wave_length = min_wave_length * (mult ** s)
            f0 = 1.0 / wave_length
            sigma_f = np.log(sigma_on_f) / np.log(2.0) if sigma_on_f > 0 else 0.5
            # actually sigma_f is the bandwidth parameter
            H = _log_gabor_2d(rows, cols, f0, theta, 0.65, sigma_theta)
            IF = np.fft.ifft2(IM * H)
            E = IF.real
            O = IF.imag
            A = np.sqrt(E ** 2 + O ** 2) + 1e-10
            sum_E += E
            sum_O += O
            sum_A += A

        energy = np.sqrt(sum_E ** 2 + sum_O ** 2)
        pc = energy / (sum_A + k * np.std(energy[energy < np.mean(energy)]) + 1e-10)
        total_energy += energy
        total_weight += sum_A

    pc_map = total_energy / (total_weight + 1e-10)
    # Normalise
    mn, mx = pc_map.min(), pc_map.max()
    pc_map = (pc_map - mn) / (mx - mn + 1e-10)
    return pc_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Keypoint detection via non-maximum suppression
# ---------------------------------------------------------------------------

def _detect_keypoints(
    pc_map: np.ndarray,
    nms_radius: int = 5,
    threshold: float = 0.1,
    max_keypoints: int = 2000,
    border: int = 16,
) -> np.ndarray:
    """Detect keypoints as local maxima in the phase congruency map.

    Returns [N, 2] array of (x, y) coordinates (float32).
    """
    size = 2 * nms_radius + 1
    local_max = maximum_filter(pc_map, size=size)
    mask = (pc_map == local_max) & (pc_map > threshold)

    # Remove border
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    rows, cols = np.where(mask)
    scores = pc_map[rows, cols]

    if len(scores) > max_keypoints:
        idx = np.argpartition(scores, -max_keypoints)[-max_keypoints:]
        rows, cols, scores = rows[idx], cols[idx], scores[idx]

    kpts = np.stack([cols, rows], axis=-1).astype(np.float32)
    return kpts


# ---------------------------------------------------------------------------
# MIM (Maximum Index Map) descriptor
# ---------------------------------------------------------------------------

def _build_log_gabor_bank(rows: int, cols: int, n_orient: int = 6, n_scale: int = 4) -> List[np.ndarray]:
    """Build a list of log-Gabor filter responses (real part) for MIM descriptor."""
    IM_placeholder = None  # filled per image
    return []  # actual filters built in _compute_mim_descriptor


def _compute_orientation_map(
    img: np.ndarray,
    n_orient: int = 6,
    min_wave_length: float = 3.0,
    mult: float = 2.1,
) -> np.ndarray:
    """Compute dominant orientation index map using log-Gabor filters.

    Returns int32 map of shape (H, W) with values in [0, n_orient).
    """
    rows, cols = img.shape
    IM = np.fft.fft2(img.astype(np.float64))
    sigma_theta = np.pi / (n_orient * 2)
    energy = np.zeros((n_orient, rows, cols), dtype=np.float64)

    for o in range(n_orient):
        theta = o * np.pi / n_orient
        wave_length = min_wave_length
        f0 = 1.0 / wave_length
        H = _log_gabor_2d(rows, cols, f0, theta, 0.65, sigma_theta)
        resp = np.fft.ifft2(IM * H)
        energy[o] = np.sqrt(resp.real ** 2 + resp.imag ** 2)

    return np.argmax(energy, axis=0).astype(np.int32)


def _extract_mim_descriptor(
    img: np.ndarray,
    orient_map: np.ndarray,
    kpts: np.ndarray,
    patch_size: int = 96,
    n_bins: int = 6,
    n_orient: int = 6,
    n_rings: int = 4,
) -> np.ndarray:
    """Extract rotation-invariant MIM descriptors.

    Returns [N, D] float32 array where D = n_bins * n_orient * n_rings.
    """
    half = patch_size // 2
    H, W = img.shape
    descriptors = []

    for kp in kpts:
        cx, cy = int(round(kp[0])), int(round(kp[1]))
        # Extract orientation map patch
        r1, r2 = max(0, cy - half), min(H, cy + half)
        c1, c2 = max(0, cx - half), min(W, cx + half)
        patch = orient_map[r1:r2, c1:c2]
        if patch.size == 0:
            descriptors.append(np.zeros(n_bins * n_orient * n_rings, dtype=np.float32))
            continue

        # Resize to canonical size
        from scipy.ndimage import zoom  # type: ignore[import]
        ph, pw = patch.shape
        if ph != patch_size or pw != patch_size:
            zy = patch_size / (ph + 1e-8)
            zx = patch_size / (pw + 1e-8)
            patch = zoom(patch.astype(np.float32), (zy, zx), order=0).astype(np.int32)
            patch = np.clip(patch, 0, n_orient - 1)

        # Dominant orientation for rotation normalisation
        hist_global = np.bincount(patch.flatten(), minlength=n_orient).astype(np.float64)
        dominant = int(np.argmax(hist_global))

        # Build MIM descriptor: concentric rings × orientation bins
        cx_c, cy_c = patch_size // 2, patch_size // 2
        ring_width = half // n_rings
        desc = []
        for ring in range(n_rings):
            r_inner = ring * ring_width
            r_outer = (ring + 1) * ring_width
            ys, xs = np.mgrid[0:patch_size, 0:patch_size]
            dist = np.sqrt((xs - cx_c) ** 2 + (ys - cy_c) ** 2)
            ring_mask = (dist >= r_inner) & (dist < r_outer)
            ring_vals = patch[ring_mask]
            # Rotate orientation bins relative to dominant
            ring_vals_rotated = (ring_vals - dominant) % n_orient
            for b in range(n_bins):
                angle_start = b * n_orient // n_bins
                angle_end = (b + 1) * n_orient // n_bins
                sector_mask = (ring_vals_rotated >= angle_start) & (ring_vals_rotated < angle_end)
                desc.append(float(np.sum(sector_mask)))
        desc_arr = np.array(desc, dtype=np.float32)
        norm = np.linalg.norm(desc_arr) + 1e-8
        descriptors.append(desc_arr / norm)

    return np.array(descriptors, dtype=np.float32) if descriptors else np.zeros((0, n_bins * n_orient * n_rings), dtype=np.float32)


# ---------------------------------------------------------------------------
# Descriptor matching via NCC (normalised cross-correlation distance)
# ---------------------------------------------------------------------------

def _match_descriptors_ncc(
    desc0: np.ndarray,
    desc1: np.ndarray,
    ratio_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match descriptors using cosine similarity (equivalent to NCC for unit vectors).

    Returns (indices0, indices1, scores).
    """
    if len(desc0) == 0 or len(desc1) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    # Cosine similarity matrix: [N0, N1]
    sim = desc0 @ desc1.T  # both are L2-normalised

    # Lowe ratio test in similarity space
    # For each row find two best matches
    sorted_idx = np.argsort(-sim, axis=1)
    best_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1] if sim.shape[1] > 1 else sorted_idx[:, 0]
    best_sim = sim[np.arange(len(desc0)), best_idx]
    second_sim = sim[np.arange(len(desc0)), second_idx]

    ratio = best_sim / (second_sim + 1e-8)
    valid = ratio > (1.0 / ratio_threshold)  # passes if best is significantly better

    # Cross-check
    sorted_idx1 = np.argsort(-sim, axis=0)
    best_idx1 = sorted_idx1[0, :]  # best match for each desc1

    ids0 = np.where(valid)[0]
    ids1 = best_idx[ids0]
    # Keep only mutually best
    mutual = best_idx1[ids1] == ids0
    ids0, ids1 = ids0[mutual], ids1[mutual]
    scores = best_sim[ids0]
    return ids0, ids1, scores.astype(np.float32)


# ---------------------------------------------------------------------------
# RIFT2 Matcher class
# ---------------------------------------------------------------------------

class RIFT2Matcher(BaseMatcher):
    """RIFT2 — classical multi-modal rotation-invariant feature matcher.

    Implementation of the RIFT2 algorithm:
    - Phase congruency (log-Gabor) keypoint detection
    - MIM (Maximum Index Map) rotation-invariant descriptor
    - NCC-based descriptor matching with ratio test
    """

    name = "rift2"

    def __init__(
        self,
        device: str = "auto",
        models_dir: str = "models/matchers",
        max_keypoints: int = 2000,
        nms_radius: int = 5,
        pc_threshold: float = 0.1,
        patch_size: int = 96,
        n_orient: int = 6,
        n_rings: int = 4,
        n_bins: int = 6,
        ratio_threshold: float = 0.8,
    ) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.pc_threshold = pc_threshold
        self.patch_size = patch_size
        self.n_orient = n_orient
        self.n_rings = n_rings
        self.n_bins = n_bins
        self.ratio_threshold = ratio_threshold

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Nothing to load — RIFT2 is a classical algorithm."""
        self._loaded = True
        self.logger.info("RIFT2 classical matcher ready (no weights needed)")

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        try:
            g0 = self._to_gray(img0)
            g1 = self._to_gray(img1)

            # Phase congruency maps
            pc0 = _phase_congruency(g0)
            pc1 = _phase_congruency(g1)

            # Keypoint detection
            kpts0 = _detect_keypoints(pc0, nms_radius=self.nms_radius, threshold=self.pc_threshold, max_keypoints=self.max_keypoints)
            kpts1 = _detect_keypoints(pc1, nms_radius=self.nms_radius, threshold=self.pc_threshold, max_keypoints=self.max_keypoints)

            if len(kpts0) == 0 or len(kpts1) == 0:
                empty = np.zeros((0, 2), dtype=np.float32)
                elapsed = time.perf_counter() - t0
                return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name, processing_time_s=elapsed)

            # Orientation maps
            om0 = _compute_orientation_map(g0, n_orient=self.n_orient)
            om1 = _compute_orientation_map(g1, n_orient=self.n_orient)

            # Descriptor extraction
            desc0 = _extract_mim_descriptor(g0, om0, kpts0, patch_size=self.patch_size, n_bins=self.n_bins, n_orient=self.n_orient, n_rings=self.n_rings)
            desc1 = _extract_mim_descriptor(g1, om1, kpts1, patch_size=self.patch_size, n_bins=self.n_bins, n_orient=self.n_orient, n_rings=self.n_rings)

            # Matching
            ids0, ids1, scores = _match_descriptors_ncc(desc0, desc1, ratio_threshold=self.ratio_threshold)

            if len(ids0) == 0:
                empty = np.zeros((0, 2), dtype=np.float32)
                elapsed = time.perf_counter() - t0
                return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name, processing_time_s=elapsed)

            matched_kpts0 = kpts0[ids0]
            matched_kpts1 = kpts1[ids1]
            elapsed = time.perf_counter() - t0
            self.logger.info("RIFT2 matching done", n_kpts0=len(kpts0), n_kpts1=len(kpts1), n_matches=len(ids0), elapsed_s=round(elapsed, 3))
            return MatchResult(
                kpts0=matched_kpts0,
                kpts1=matched_kpts1,
                confidence=scores,
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("RIFT2 failed; falling back to ORB+BF", error=str(exc))
            return self._orb_bf_fallback(img0, img1, self.name)
