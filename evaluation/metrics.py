"""
Responsibility (SRP), split into two clear halves in this one file:

1. MaskOverlapMetrics
   Pure, framework-independent geometric metrics on binary masks
   (IoU, Dice, pixel accuracy). These don't know about ultralytics,
   YOLO, or our project at all - they just take numpy boolean arrays.
   This makes them trivially unit-testable and reusable even if we
   ever swap the underlying model framework.

   Why these, in addition to precision/recall/mAP:
   - IoU (Intersection over Union): the foundational overlap measure;
     COCO-style mAP itself is built by thresholding IoU, so reporting
     mean IoU directly gives a more intuitive "how good are the masks"
     number than mAP alone.
   - Dice coefficient (a.k.a. F1 over pixels): standard in segmentation
     literature (especially medical imaging), more sensitive to small
     objects than IoU since it weights overlap differently. Reporting
     both is a common, recommended practice for segmentation tasks.
   - Pixel accuracy: simplest possible sanity-check metric - fraction
     of pixels correctly classified. Useful for catching gross bugs
     (e.g. an inverted mask) that a low-level IoU might obscure.

2. UltralyticsMetricsParser
   The ONLY place in the project that knows ultralytics' result-object
   attribute names (box.mp, box.mr, seg.map50, etc.). Everything else
   works with our own DetectionMetrics/SegmentationEvaluationResult
   dataclasses instead.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 1. Pure mask-overlap metrics
# ----------------------------------------------------------------------


class MaskOverlapMetrics:
    """Stateless geometric metrics on binary (boolean) masks."""

    @staticmethod
    def iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
        """Intersection over Union. Returns 1.0 for two empty masks
        (both-empty is treated as a perfect, trivial match)."""
        predicted_mask = predicted_mask.astype(bool)
        ground_truth_mask = ground_truth_mask.astype(bool)

        intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
        union = np.logical_or(predicted_mask, ground_truth_mask).sum()

        if union == 0:
            return 1.0
        return float(intersection) / float(union)

    @staticmethod
    def dice(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
        """Dice coefficient (= 2*|A∩B| / (|A|+|B|)). Returns 1.0 for two
        empty masks."""
        predicted_mask = predicted_mask.astype(bool)
        ground_truth_mask = ground_truth_mask.astype(bool)

        intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
        total = predicted_mask.sum() + ground_truth_mask.sum()

        if total == 0:
            return 1.0
        return float(2 * intersection) / float(total)

    @staticmethod
    def pixel_accuracy(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
        """Fraction of pixels where predicted and ground truth agree."""
        predicted_mask = predicted_mask.astype(bool)
        ground_truth_mask = ground_truth_mask.astype(bool)

        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_mask.shape} vs "
                f"ground_truth {ground_truth_mask.shape}"
            )

        correct = (predicted_mask == ground_truth_mask).sum()
        return float(correct) / float(predicted_mask.size)

    @classmethod
    def mean_iou(cls, mask_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Mean IoU across a list of (predicted_mask, ground_truth_mask) pairs."""
        if not mask_pairs:
            raise ValueError("mean_iou requires at least one mask pair.")
        scores = [cls.iou(pred, gt) for pred, gt in mask_pairs]
        return float(np.mean(scores))

    @classmethod
    def mean_dice(cls, mask_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Mean Dice coefficient across a list of (predicted_mask, ground_truth_mask) pairs."""
        if not mask_pairs:
            raise ValueError("mean_dice requires at least one mask pair.")
        scores = [cls.dice(pred, gt) for pred, gt in mask_pairs]
        return float(np.mean(scores))


# ----------------------------------------------------------------------
# 2. Structured results + ultralytics parsing
# ----------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    """Generic precision/recall/mAP bundle - used for BOTH the box
    branch and the mask branch of a YOLO-seg evaluation, since both
    share the same metric shape."""

    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class_ap50: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "per_class_ap50": self.per_class_ap50,
        }


@dataclass
class SegmentationEvaluationResult:
    box: DetectionMetrics
    mask: DetectionMetrics
    extra_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "box": self.box.to_dict(),
            "mask": self.mask.to_dict(),
            "extra_metrics": self.extra_metrics,
        }


class UltralyticsMetricsParser:
    """
    The only class in the project allowed to know ultralytics' result
    object attribute names. If ultralytics changes its API, only this
    class needs updating.
    """

    @staticmethod
    def parse(val_results) -> SegmentationEvaluationResult:
        box_metrics = UltralyticsMetricsParser._parse_branch(val_results.box, val_results.names)
        mask_metrics = UltralyticsMetricsParser._parse_branch(val_results.seg, val_results.names)
        return SegmentationEvaluationResult(box=box_metrics, mask=mask_metrics)

    @staticmethod
    def _parse_branch(metric_branch, class_names: Dict[int, str]) -> DetectionMetrics:
        per_class_ap50 = UltralyticsMetricsParser._per_class_ap50(metric_branch, class_names)
        return DetectionMetrics(
            precision=float(metric_branch.mp),
            recall=float(metric_branch.mr),
            map50=float(metric_branch.map50),
            map50_95=float(metric_branch.map),
            per_class_ap50=per_class_ap50,
        )

    @staticmethod
    def _per_class_ap50(metric_branch, class_names: Dict[int, str]) -> Optional[Dict[str, float]]:
        ap50_values = getattr(metric_branch, "ap50", None)
        class_indices = getattr(metric_branch, "ap_class_index", None)

        if ap50_values is None or class_indices is None or len(ap50_values) == 0:
            return None

        return {
            class_names.get(int(class_idx), str(int(class_idx))): float(ap50)
            for class_idx, ap50 in zip(class_indices, ap50_values)
        }
