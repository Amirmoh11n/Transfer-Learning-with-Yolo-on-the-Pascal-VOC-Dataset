"""
Responsibility (SRP):
    Translate ONE raw ultralytics Results object into our own
    ImagePrediction/InstancePrediction dataclasses (prediction_result.py).

    This is the ONLY place in the project allowed to know ultralytics'
    result attribute names (boxes.cls, boxes.conf, boxes.xyxy,
    masks.xyn, .names, .speed). If ultralytics changes that shape in a
    future version, only this file needs updating.
"""

from typing import Optional

from inference.inference_codes.prediction_result import (
    ImagePrediction,
    InstancePrediction,
)


class UltralyticsResultParser:
    def parse(self, source_path: str, raw_result) -> ImagePrediction:
        names = raw_result.names
        boxes = raw_result.boxes
        masks = raw_result.masks

        num_detections = len(boxes) if boxes is not None else 0
        instances = []

        for i in range(num_detections):
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            box_xyxy = tuple(float(v) for v in boxes.xyxy[i])
            polygon = self._extract_polygon(masks, i)

            instances.append(
                InstancePrediction(
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    confidence=confidence,
                    box_xyxy=box_xyxy,
                    polygon=polygon,
                )
            )

        inference_time_ms = self._total_speed_ms(raw_result)
        resolved_path = str(getattr(raw_result, "path", source_path) or source_path)

        return ImagePrediction(
            source_path=resolved_path,
            instances=instances,
            inference_time_ms=inference_time_ms,
        )

    @staticmethod
    def _extract_polygon(masks, instance_index: int) -> Optional[list]:
        """
        masks is None when the image had zero detections (even for a
        -seg model). Guard against that, and against an instance index
        that (in principle) has a box but no corresponding mask entry.
        """
        if masks is None:
            return None
        if instance_index >= len(masks.xyn):
            return None
        return [(float(x), float(y)) for x, y in masks.xyn[instance_index]]

    @staticmethod
    def _total_speed_ms(raw_result) -> float:
        speed = getattr(raw_result, "speed", None)
        if not speed:
            return 0.0
        return float(sum(speed.values()))
