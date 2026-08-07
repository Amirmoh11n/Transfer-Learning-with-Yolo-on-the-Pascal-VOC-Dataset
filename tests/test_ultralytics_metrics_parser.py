"""
Tests for UltralyticsMetricsParser.

We build a lightweight FAKE object that mimics the shape of
ultralytics' real SegmentMetrics result (val_results.box.*,
val_results.seg.*, val_results.names) without depending on
ultralytics or running any real validation. This verifies OUR parsing
logic in isolation.
"""

import numpy as np
import pytest

from evaluation.metrics import DetectionMetrics, UltralyticsMetricsParser


class FakeMetricBranch:
    """Mimics ultralytics' Metric class (the .box / .seg sub-objects)."""

    def __init__(self, mp, mr, map50, map50_95, ap50=None, ap_class_index=None):
        self.mp = mp
        self.mr = mr
        self.map50 = map50
        self.map = map50_95  # note: ultralytics calls this attribute "map", not "map50_95"
        self.ap50 = ap50 if ap50 is not None else np.array([])
        self.ap_class_index = ap_class_index if ap_class_index is not None else np.array([])


class FakeValResults:
    """Mimics ultralytics' SegmentMetrics top-level result object."""

    def __init__(self, box: FakeMetricBranch, seg: FakeMetricBranch, names: dict):
        self.box = box
        self.seg = seg
        self.names = names


def test_parse_extracts_box_and_mask_metrics():
    fake_results = FakeValResults(
        box=FakeMetricBranch(mp=0.80, mr=0.70, map50=0.75, map50_95=0.50),
        seg=FakeMetricBranch(mp=0.78, mr=0.68, map50=0.72, map50_95=0.48),
        names={0: "aeroplane", 1: "bicycle"},
    )

    result = UltralyticsMetricsParser.parse(fake_results)

    assert isinstance(result.box, DetectionMetrics)
    assert result.box.precision == pytest.approx(0.80)
    assert result.box.recall == pytest.approx(0.70)
    assert result.box.map50 == pytest.approx(0.75)
    assert result.box.map50_95 == pytest.approx(0.50)

    assert result.mask.precision == pytest.approx(0.78)
    assert result.mask.recall == pytest.approx(0.68)
    assert result.mask.map50 == pytest.approx(0.72)
    assert result.mask.map50_95 == pytest.approx(0.48)


def test_parse_builds_per_class_ap50_dict():
    fake_results = FakeValResults(
        box=FakeMetricBranch(
            mp=0.8, mr=0.7, map50=0.75, map50_95=0.5,
            ap50=np.array([0.9, 0.6]),
            ap_class_index=np.array([0, 1]),
        ),
        seg=FakeMetricBranch(mp=0.7, mr=0.6, map50=0.65, map50_95=0.4),
        names={0: "aeroplane", 1: "bicycle"},
    )

    result = UltralyticsMetricsParser.parse(fake_results)

    assert result.box.per_class_ap50 == {
        "aeroplane": pytest.approx(0.9),
        "bicycle": pytest.approx(0.6),
    }


def test_parse_handles_empty_ap50_gracefully():
    fake_results = FakeValResults(
        box=FakeMetricBranch(mp=0.0, mr=0.0, map50=0.0, map50_95=0.0),
        seg=FakeMetricBranch(mp=0.0, mr=0.0, map50=0.0, map50_95=0.0),
        names={},
    )

    result = UltralyticsMetricsParser.parse(fake_results)

    assert result.box.per_class_ap50 is None
    assert result.mask.per_class_ap50 is None


def test_detection_metrics_to_dict_round_trips_values():
    metrics = DetectionMetrics(
        precision=0.5, recall=0.6, map50=0.7, map50_95=0.4,
        per_class_ap50={"dog": 0.8},
    )
    as_dict = metrics.to_dict()
    assert as_dict["precision"] == 0.5
    assert as_dict["per_class_ap50"] == {"dog": 0.8}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
