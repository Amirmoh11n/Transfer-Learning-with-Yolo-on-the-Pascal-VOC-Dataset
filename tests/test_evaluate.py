"""
Tests for SegmentationEvaluator.

model.validate() is mocked (via a fake YOLOSegmentationModel), and we
reuse the FakeValResults/FakeMetricBranch shapes from
test_ultralytics_metrics_parser.py so we don't need a real model,
real dataset, or real ultralytics validation run.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from evaluation.evaluate import SegmentationEvaluator
from tests.test_ultralytics_metrics_parser import FakeMetricBranch, FakeValResults


def make_fake_model(fake_val_results):
    fake_model = MagicMock()
    fake_model.validate.return_value = fake_val_results
    return fake_model


def make_fake_val_results():
    return FakeValResults(
        box=FakeMetricBranch(mp=0.8, mr=0.7, map50=0.75, map50_95=0.5),
        seg=FakeMetricBranch(mp=0.75, mr=0.65, map50=0.70, map50_95=0.45),
        names={0: "person", 1: "dog"},
    )


# ----------------------------------------------------------------------
# evaluate()
# ----------------------------------------------------------------------


def test_evaluate_calls_model_validate_and_returns_parsed_result():
    fake_val_results = make_fake_val_results()
    fake_model = make_fake_model(fake_val_results)

    evaluator = SegmentationEvaluator(fake_model)
    result = evaluator.evaluate("configs/data.yaml")

    fake_model.validate.assert_called_once_with("configs/data.yaml", split="val")
    assert result.box.precision == pytest.approx(0.8)
    assert result.mask.map50 == pytest.approx(0.70)


def test_evaluate_without_data_path_passes_none():
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)

    evaluator.evaluate()

    fake_model.validate.assert_called_once_with(None, split="val")


def test_evaluate_can_request_test_split_explicitly():
    """Confirms the held-out test split is reachable, but only when the
    caller deliberately asks for it - never the silent default."""
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)

    evaluator.evaluate("configs/data.yaml", split="test")

    fake_model.validate.assert_called_once_with("configs/data.yaml", split="test")


# ----------------------------------------------------------------------
# evaluate_mask_overlap()
# ----------------------------------------------------------------------


def test_evaluate_mask_overlap_returns_expected_keys():
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)

    mask = np.ones((5, 5), dtype=bool)
    result = evaluator.evaluate_mask_overlap([(mask, mask)])

    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["mean_dice"] == pytest.approx(1.0)
    assert result["mean_pixel_accuracy"] == pytest.approx(1.0)
    assert result["num_samples"] == 1


def test_evaluate_mask_overlap_empty_list_raises():
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)

    with pytest.raises(ValueError):
        evaluator.evaluate_mask_overlap([])


# ----------------------------------------------------------------------
# summary() / save_report()
# ----------------------------------------------------------------------


def test_summary_contains_key_metric_values():
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)
    result = evaluator.evaluate()

    text = SegmentationEvaluator.summary(result)

    assert "0.8000" in text  # box precision
    assert "0.7000" in text  # mask map50
    assert "Box detection" in text
    assert "Mask segmentation" in text


def test_save_report_writes_valid_json(tmp_path):
    fake_model = make_fake_model(make_fake_val_results())
    evaluator = SegmentationEvaluator(fake_model)
    result = evaluator.evaluate()

    output_path = tmp_path / "reports" / "eval.json"
    SegmentationEvaluator.save_report(result, str(output_path))

    assert output_path.is_file()
    with open(output_path) as f:
        loaded = json.load(f)

    assert loaded["box"]["precision"] == pytest.approx(0.8)
    assert loaded["mask"]["map50"] == pytest.approx(0.70)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
