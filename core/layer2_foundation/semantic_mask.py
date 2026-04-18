"""SemanticMask — wraps a boolean stability mask and provides keypoint filtering."""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class SemanticMask:
    """Binary semantic stability mask derived from foundation model embeddings.

    Parameters
    ----------
    mask:
        Boolean ``(H, W)`` array; ``True`` means the pixel is *stable* and
        suitable for use as a control point.
    """

    def __init__(self, mask: np.ndarray) -> None:
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
        self.mask: np.ndarray = mask.astype(bool)

    # ------------------------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        n_stable_clusters: int = 4,
        n_clusters: int = 8,
        stability_threshold: Optional[float] = None,
    ) -> "SemanticMask":
        """Build a :class:`SemanticMask` from dense ``(H, W, D)`` embeddings.

        Stable regions are identified by applying k-means on the embedding
        space and labelling the cluster(s) with lowest intra-cluster variance
        as stable (e.g., buildings, roads, bare soil).

        Parameters
        ----------
        embeddings:
            Dense feature array of shape ``(H, W, D)``.
        n_stable_clusters:
            Number of k-means clusters to mark as stable.
        n_clusters:
            Total number of k-means clusters.
        stability_threshold:
            If provided, overrides the cluster-based approach: pixels whose
            L2 norm of embeddings is below this threshold are marked stable.
        """
        H, W, D = embeddings.shape
        flat = embeddings.reshape(-1, D).astype(np.float32)

        # Norm-based shortcut when sklearn is unavailable or threshold given
        if stability_threshold is not None or not _HAS_SKLEARN:
            norms = np.linalg.norm(flat, axis=1)
            thr = stability_threshold if stability_threshold is not None else float(
                np.percentile(norms, 40)
            )
            stable = (norms <= thr).reshape(H, W)
            return cls(stable)

        # K-means clustering
        try:
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            labels = km.fit_predict(flat)
        except Exception as exc:
            logger.warning("SemanticMask k-means failed; using norm fallback",
                           error=str(exc))
            norms = np.linalg.norm(flat, axis=1)
            thr = float(np.percentile(norms, 40))
            stable = (norms <= thr).reshape(H, W)
            return cls(stable)

        # Rank clusters by intra-cluster variance (low variance = stable)
        variances = np.array([
            float(flat[labels == k].var()) if (labels == k).any() else np.inf
            for k in range(n_clusters)
        ])
        stable_cluster_ids = set(np.argsort(variances)[:n_stable_clusters].tolist())
        stable = np.array([l in stable_cluster_ids for l in labels], dtype=bool).reshape(H, W)

        logger.debug(
            "SemanticMask built",
            stable_fraction=float(stable.mean()),
            n_stable_clusters=n_stable_clusters,
        )
        return cls(stable)

    # ------------------------------------------------------------------

    def apply_to_keypoints(self, kpts: np.ndarray) -> np.ndarray:
        """Filter keypoints to those falling within stable regions.

        Parameters
        ----------
        kpts:
            Array of shape ``(N, 2)`` with columns ``[col, row]``
            (OpenCV / pixel convention).

        Returns
        -------
        np.ndarray
            Filtered keypoints of shape ``(M, 2)`` where M ≤ N.
        """
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            raise ValueError("kpts must be (N, 2) with columns [col, row]")

        H, W = self.mask.shape
        cols = kpts[:, 0].astype(int)
        rows = kpts[:, 1].astype(int)

        # Clamp to valid range
        cols_c = np.clip(cols, 0, W - 1)
        rows_c = np.clip(rows, 0, H - 1)

        valid = self.mask[rows_c, cols_c]
        filtered = kpts[valid]
        logger.debug(
            "keypoints filtered",
            total=len(kpts),
            kept=int(valid.sum()),
        )
        return filtered

    # ------------------------------------------------------------------

    def dilate(self, radius: int = 3) -> "SemanticMask":
        """Return a morphologically dilated copy of this mask."""
        from scipy.ndimage import binary_dilation  # type: ignore[import]

        structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
        dilated = binary_dilation(self.mask, structure=structure)
        return SemanticMask(dilated)

    def erode(self, radius: int = 2) -> "SemanticMask":
        """Return a morphologically eroded copy of this mask."""
        from scipy.ndimage import binary_erosion  # type: ignore[import]

        structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
        eroded = binary_erosion(self.mask, structure=structure)
        return SemanticMask(eroded)

    def __repr__(self) -> str:
        h, w = self.mask.shape
        frac = float(self.mask.mean())
        return f"SemanticMask(shape=({h},{w}), stable_fraction={frac:.3f})"
