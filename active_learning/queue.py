"""Active learning queue for uncertain or failed registrations."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class ActiveLearningQueue:
    """Queue of samples that require human review."""

    def __init__(
        self,
        queue_dir: Path = Path("data/active_learning"),
        uncertainty_threshold: float = 0.5,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.uncertainty_threshold = uncertainty_threshold
        self._index_path = self.queue_dir / "index.json"
        self._items: dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._items, indent=2, default=str)
        )

    # ------------------------------------------------------------------
    def add(
        self,
        source_path: Path,
        target_path: Path,
        uncertainty: float,
        qa_report=None,
    ) -> str:
        """Add a sample to the queue if it meets the threshold."""
        qa_passes = True
        if qa_report is not None:
            qa_passes = qa_report.passes_qa

        if uncertainty < self.uncertainty_threshold and qa_passes:
            return ""  # Not worth reviewing

        item_id = str(uuid.uuid4())
        self._items[item_id] = {
            "id": item_id,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "uncertainty": uncertainty,
            "qa_passes": qa_passes,
            "reviewed": False,
            "annotations": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_index()
        logger.info(
            "item_added_to_queue",
            item_id=item_id,
            uncertainty=uncertainty,
        )
        return item_id

    def get_next(self, n: int = 10) -> list[dict]:
        """Return top-n unreviewed items sorted by uncertainty (desc)."""
        unreviewed = [
            v for v in self._items.values() if not v["reviewed"]
        ]
        unreviewed.sort(key=lambda x: x["uncertainty"], reverse=True)
        return unreviewed[:n]

    def mark_reviewed(self, item_id: str, annotations: dict) -> None:
        """Mark an item as reviewed with human annotations."""
        if item_id in self._items:
            self._items[item_id]["reviewed"] = True
            self._items[item_id]["annotations"] = annotations
            self._save_index()
            logger.info("item_reviewed", item_id=item_id)

    def get_statistics(self) -> dict:
        total = len(self._items)
        reviewed = sum(1 for v in self._items.values() if v["reviewed"])
        return {
            "total": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "review_rate": reviewed / total if total else 0.0,
        }
