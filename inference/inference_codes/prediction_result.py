"""
Responsibility (SRP):
    Hold structured prediction results. Pure data, no logic, no
    dependency on ultralytics - the rest of the project (and any
    caller) works with THESE classes, never with raw ultralytics
    Results objects directly.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InstancePrediction:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]  # pixel coordinates
    # Normalized (x, y) polygon points in [0, 1], same convention as
    # our VOC->YOLO label polygons. None if the model produced a box
    # but no mask for this instance (shouldn't happen for a -seg model,
    # but we don't assume it can't).
    polygon: Optional[List[Tuple[float, float]]] = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box_xyxy": list(self.box_xyxy),
            "polygon": [list(p) for p in self.polygon] if self.polygon is not None else None,
        }


@dataclass
class ImagePrediction:
    source_path: str
    instances: List[InstancePrediction] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "inference_time_ms": self.inference_time_ms,
            "instances": [instance.to_dict() for instance in self.instances],
        }
