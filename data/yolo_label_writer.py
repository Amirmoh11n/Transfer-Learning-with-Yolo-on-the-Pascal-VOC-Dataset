"""
Responsibility (SRP):
    Serialize a list of InstanceAnnotation objects to a YOLO-seg
    label .txt file. Knows the YOLO-seg text format and nothing else -
    no mask logic, no path resolution beyond writing to a given path.

YOLO-seg label line format (one line per object instance):
    <class_id> x1 y1 x2 y2 x3 y3 ... xn yn
    - class_id: integer, 0-indexed
    - coordinates: normalized floats in [0, 1], polygon points in order
"""

from pathlib import Path
from typing import List

from data.voc_instance_extractor import InstanceAnnotation


class YOLOLabelWriter:
    def write(self, label_path: Path, annotations: List[InstanceAnnotation]) -> None:
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [self._format_line(ann) for ann in annotations]

        # Write even if empty: an image with zero valid instances still
        # needs an (empty) label file so the loader doesn't treat it as
        # "missing label" during training.
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    @staticmethod
    def _format_line(annotation: InstanceAnnotation) -> str:
        coords = " ".join(
            f"{x:.6f} {y:.6f}" for x, y in annotation.polygon
        )
        return f"{annotation.yolo_class_id} {coords}"
