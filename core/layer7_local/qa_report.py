"""QA Report dataclass."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class QAReport:
    source_path: Path
    target_path: Path
    num_tie_points: int
    num_inliers: int
    rmse: float
    max_residual: float
    passes_qa: bool
    threshold_rmse: float
    matcher_used: str
    processing_time_s: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_path"] = str(self.source_path)
        d["target_path"] = str(self.target_path)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
