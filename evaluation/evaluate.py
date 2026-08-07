"""
Responsibility (SRP):
    Orchestrate evaluation: ask the model to validate itself, parse
    the raw results into our own structured dataclasses (via
    metrics.py), and produce human-readable / machine-readable reports.

    This file contains NO knowledge of ultralytics' attribute names
    (that lives in UltralyticsMetricsParser) and NO raw mask-math
    (that lives in MaskOverlapMetrics). It only sequences calls.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from evaluation.metrics import (
    MaskOverlapMetrics,
    SegmentationEvaluationResult,
    UltralyticsMetricsParser,
)
from models.yolo import YOLOSegmentationModel


class SegmentationEvaluator:
    def __init__(self, model: YOLOSegmentationModel):
        self.model = model

    def evaluate(
        self, data_yaml_path: Optional[str] = None, split: str = "val"
    ) -> SegmentationEvaluationResult:
        """
        Runs the model's built-in validation (precision/recall/mAP for
        both box and mask branches) and returns our structured result.

        Args:
            split: "val" (default, safe to call repeatedly) or "test"
                (the held-out set - intended for a single final,
                unbiased report, never during tuning).
        """
        raw_results = self.model.validate(data_yaml_path, split=split)
        return UltralyticsMetricsParser.parse(raw_results)

    def evaluate_mask_overlap(
        self, mask_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> dict:
        """
        Supplementary custom metric, computed directly on
        (predicted_mask, ground_truth_mask) pairs the caller provides -
        useful for a quick standalone sanity check outside the full
        ultralytics validation loop (e.g. on a handful of sample images).

        Returns a dict with mean_iou, mean_dice, and mean_pixel_accuracy.
        """
        if not mask_pairs:
            raise ValueError("evaluate_mask_overlap requires at least one mask pair.")

        pixel_accuracies = [
            MaskOverlapMetrics.pixel_accuracy(pred, gt) for pred, gt in mask_pairs
        ]

        return {
            "mean_iou": MaskOverlapMetrics.mean_iou(mask_pairs),
            "mean_dice": MaskOverlapMetrics.mean_dice(mask_pairs),
            "mean_pixel_accuracy": float(np.mean(pixel_accuracies)),
            "num_samples": len(mask_pairs),
        }

    @staticmethod
    def summary(result: SegmentationEvaluationResult) -> str:
        lines = [
            "Segmentation Evaluation Summary",
            "-------------------------------",
            "Box detection:",
            f"  Precision:   {result.box.precision:.4f}",
            f"  Recall:      {result.box.recall:.4f}",
            f"  mAP@0.5:     {result.box.map50:.4f}",
            f"  mAP@0.5:0.95:{result.box.map50_95:.4f}",
            "Mask segmentation:",
            f"  Precision:   {result.mask.precision:.4f}",
            f"  Recall:      {result.mask.recall:.4f}",
            f"  mAP@0.5:     {result.mask.map50:.4f}",
            f"  mAP@0.5:0.95:{result.mask.map50_95:.4f}",
        ]
        if result.extra_metrics:
            lines.append("Extra metrics:")
            for key, value in result.extra_metrics.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def save_report(result: SegmentationEvaluationResult, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
