"""HITL annotator: exports and imports human annotations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class HITLAnnotator:
    """Exports match pairs for human review and imports corrections."""

    def export_for_annotation(
        self, items: list[dict], output_dir: Path
    ) -> None:
        """Export PNG crops with keypoint overlays for human review.

        Args:
            items: List of queue items from ActiveLearningQueue.
            output_dir: Directory to write exported files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            item_id = item["id"]
            item_dir = output_dir / item_id
            item_dir.mkdir(exist_ok=True)

            # Write metadata JSON
            meta_path = item_dir / "metadata.json"
            meta_path.write_text(json.dumps(item, indent=2, default=str))

            # Attempt to create visual crop overlay
            try:
                import cv2  # type: ignore

                for key, path_key in [("source", "source_path"), ("target", "target_path")]:
                    img_path = Path(item.get(path_key, ""))
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            thumb = cv2.resize(img, (512, 512))
                            cv2.imwrite(str(item_dir / f"{key}.png"), thumb)
            except Exception as exc:
                logger.warning(
                    "export_image_failed",
                    item_id=item_id,
                    error=str(exc),
                )

            logger.info("item_exported", item_id=item_id, path=str(item_dir))

    def import_annotations(self, annotation_dir: Path) -> list[dict]:
        """Read reviewed annotations from exported directories.

        Args:
            annotation_dir: Root directory containing item subdirectories.

        Returns:
            List of annotation dicts.
        """
        results: list[dict] = []
        annotation_dir = Path(annotation_dir)
        for item_dir in annotation_dir.iterdir():
            if not item_dir.is_dir():
                continue
            meta_path = item_dir / "metadata.json"
            correction_path = item_dir / "correction.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            correction: dict = {}
            if correction_path.exists():
                correction = json.loads(correction_path.read_text())
            results.append({**meta, "correction": correction})
        return results

    def compute_correction(
        self, original: dict, annotated: dict
    ) -> dict:
        """Compute the delta between original and annotated matches."""
        orig_kpts = np.array(
            original.get("kpts", []), dtype=np.float32
        )
        ann_kpts = np.array(
            annotated.get("kpts", []), dtype=np.float32
        )
        if orig_kpts.shape == ann_kpts.shape and len(orig_kpts) > 0:
            delta = ann_kpts - orig_kpts
            return {
                "mean_correction_px": float(
                    np.linalg.norm(delta, axis=1).mean()
                ),
                "delta": delta.tolist(),
            }
        return {"mean_correction_px": 0.0, "delta": []}
