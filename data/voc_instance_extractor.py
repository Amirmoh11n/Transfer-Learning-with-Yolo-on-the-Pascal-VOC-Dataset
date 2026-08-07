"""
Responsibility (SRP):
    Given a pair of aligned VOC masks (class mask + object/instance mask)
    for ONE image, produce a list of (yolo_class_id, normalized_polygon)
    tuples - one per object instance.

    This is the only module that knows HOW to go from "pixels" to
    "polygon". Everything upstream (mask loading) and downstream
    (writing labels) is unaware of this logic.

Key VOC detail this module encodes:
    SegmentationObject encodes WHICH instance a pixel belongs to
    (id 1, 2, 3, ...), but not what CLASS that instance is.
    SegmentationClass encodes WHICH class a pixel belongs to, but not
    which instance. We combine both: for each instance region (from the
    object mask), we look up the class via majority vote over the
    class mask pixels in that same region.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from data.class_mapping import (
    VOC_BACKGROUND_VALUE,
    VOC_IGNORE_BOUNDARY_VALUE,
    voc_label_to_yolo_class_id,
)

Polygon = List[Tuple[float, float]]


@dataclass
class InstanceAnnotation:
    yolo_class_id: int
    polygon: Polygon  # normalized (x, y) in [0, 1], in image coordinate order


@dataclass
class ExtractionWarning:
    image_id: str
    instance_id: int
    reason: str


class VOCInstanceExtractor:
    def __init__(
        self,
        min_contour_area_px: float = 4.0,
        polygon_epsilon_ratio: float = 0.005,
        keep_largest_contour_only: bool = True,
    ):
        """
        Args:
            min_contour_area_px: contours smaller than this (in raw pixel
                area, before normalization) are dropped as noise.
            polygon_epsilon_ratio: cv2.approxPolyDP epsilon as a fraction
                of the contour's perimeter. Reduces point count while
                preserving shape. 0 disables simplification.
            keep_largest_contour_only: if an instance mask produces
                multiple disconnected regions (e.g. object split by
                occlusion), keep only the largest one. If False, each
                region is emitted as a separate polygon line with the
                same class id.
        """
        self.min_contour_area_px = min_contour_area_px
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.keep_largest_contour_only = keep_largest_contour_only

    def extract(
        self,
        image_id: str,
        class_mask: np.ndarray,
        object_mask: np.ndarray,
    ) -> Tuple[List[InstanceAnnotation], List[ExtractionWarning]]:
        if class_mask.shape != object_mask.shape:
            raise ValueError(
                f"[{image_id}] class_mask shape {class_mask.shape} != "
                f"object_mask shape {object_mask.shape}"
            )

        height, width = object_mask.shape
        annotations: List[InstanceAnnotation] = []
        warnings: List[ExtractionWarning] = []

        instance_ids = self._instance_ids(object_mask)

        for instance_id in instance_ids:
            region = object_mask == instance_id

            yolo_class_id = self._resolve_class_for_region(class_mask, region)
            if yolo_class_id is None:
                warnings.append(
                    ExtractionWarning(
                        image_id, int(instance_id), "no valid class pixels in region"
                    )
                )
                continue

            contours = self._find_contours(region)
            contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area_px]

            if not contours:
                warnings.append(
                    ExtractionWarning(
                        image_id, int(instance_id), "no contour above min area threshold"
                    )
                )
                continue

            if self.keep_largest_contour_only:
                contours = [max(contours, key=cv2.contourArea)]

            for contour in contours:
                polygon = self._contour_to_normalized_polygon(contour, width, height)
                if len(polygon) < 3:
                    warnings.append(
                        ExtractionWarning(
                            image_id,
                            int(instance_id),
                            "degenerate polygon (<3 points) after simplification",
                        )
                    )
                    continue
                annotations.append(InstanceAnnotation(yolo_class_id, polygon))

        return annotations, warnings

    @staticmethod
    def _instance_ids(object_mask: np.ndarray) -> List[int]:
        ids = np.unique(object_mask)
        return [
            int(i)
            for i in ids
            if i not in (VOC_BACKGROUND_VALUE, VOC_IGNORE_BOUNDARY_VALUE)
        ]

    @staticmethod
    def _resolve_class_for_region(class_mask: np.ndarray, region: np.ndarray):
        class_pixels = class_mask[region]
        valid = class_pixels[
            (class_pixels != VOC_BACKGROUND_VALUE)
            & (class_pixels != VOC_IGNORE_BOUNDARY_VALUE)
        ]
        if valid.size == 0:
            return None
        most_common_voc_label, _ = Counter(valid.tolist()).most_common(1)[0]
        return voc_label_to_yolo_class_id(most_common_voc_label)

    @staticmethod
    def _find_contours(region: np.ndarray):
        mask_uint8 = (region.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _contour_to_normalized_polygon(
        self, contour: np.ndarray, width: int, height: int
    ) -> Polygon:
        if self.polygon_epsilon_ratio > 0:
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.polygon_epsilon_ratio * perimeter
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        points = contour.reshape(-1, 2)
        return [(float(x) / width, float(y) / height) for x, y in points]
