"""
Tests for UltralyticsResultParser. We build small fake stand-ins for
ultralytics' Boxes/Masks/Results objects (matching the minimal
attributes the parser actually reads: cls, conf, xyxy, xyn, names,
speed, path) rather than depending on a real model - fast, no
GPU/network needed.
"""

import pytest

from inference.inference_codes.result_parser import UltralyticsResultParser


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy

    def __len__(self):
        return len(self.cls)


class FakeMasks:
    def __init__(self, xyn):
        self.xyn = xyn


class FakeRawResult:
    def __init__(self, path, names, boxes=None, masks=None, speed=None):
        self.path = path
        self.names = names
        self.boxes = boxes
        self.masks = masks
        self.speed = speed or {}

    def plot(self):
        import numpy as np

        return np.zeros((10, 10, 3), dtype="uint8")


NAMES = {0: "person", 1: "dog"}


def test_parse_zero_detections_returns_empty_instances():
    raw = FakeRawResult(
        path="img1.jpg", names=NAMES, boxes=FakeBoxes([], [], []), masks=None
    )
    parser = UltralyticsResultParser()

    prediction = parser.parse("img1.jpg", raw)

    assert prediction.source_path == "img1.jpg"
    assert prediction.instances == []


def test_parse_single_detection_with_mask():
    boxes = FakeBoxes(cls=[1], conf=[0.87], xyxy=[[10.0, 20.0, 50.0, 60.0]])
    masks = FakeMasks(xyn=[[(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)]])
    raw = FakeRawResult(path="img2.jpg", names=NAMES, boxes=boxes, masks=masks)
    parser = UltralyticsResultParser()

    prediction = parser.parse("img2.jpg", raw)

    assert len(prediction.instances) == 1
    instance = prediction.instances[0]
    assert instance.class_id == 1
    assert instance.class_name == "dog"
    assert instance.confidence == pytest.approx(0.87)
    assert instance.box_xyxy == (10.0, 20.0, 50.0, 60.0)
    assert instance.polygon == [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)]


def test_parse_detection_without_masks_field_gives_none_polygon():
    boxes = FakeBoxes(cls=[0], conf=[0.5], xyxy=[[0.0, 0.0, 5.0, 5.0]])
    raw = FakeRawResult(path="img3.jpg", names=NAMES, boxes=boxes, masks=None)
    parser = UltralyticsResultParser()

    prediction = parser.parse("img3.jpg", raw)

    assert len(prediction.instances) == 1
    assert prediction.instances[0].polygon is None


def test_parse_unknown_class_id_falls_back_to_string():
    boxes = FakeBoxes(cls=[99], conf=[0.5], xyxy=[[0.0, 0.0, 5.0, 5.0]])
    raw = FakeRawResult(path="img4.jpg", names=NAMES, boxes=boxes, masks=None)
    parser = UltralyticsResultParser()

    prediction = parser.parse("img4.jpg", raw)

    assert prediction.instances[0].class_name == "99"


def test_parse_sums_speed_dict_into_inference_time_ms():
    boxes = FakeBoxes(cls=[], conf=[], xyxy=[])
    speed = {"preprocess": 2.0, "inference": 10.0, "postprocess": 1.5}
    raw = FakeRawResult(path="img5.jpg", names=NAMES, boxes=boxes, speed=speed)
    parser = UltralyticsResultParser()

    prediction = parser.parse("img5.jpg", raw)

    assert prediction.inference_time_ms == pytest.approx(13.5)


def test_parse_missing_speed_defaults_to_zero():
    boxes = FakeBoxes(cls=[], conf=[], xyxy=[])
    raw = FakeRawResult(path="img6.jpg", names=NAMES, boxes=boxes, speed=None)
    parser = UltralyticsResultParser()

    prediction = parser.parse("img6.jpg", raw)

    assert prediction.inference_time_ms == 0.0


def test_parse_falls_back_to_source_path_when_raw_result_has_no_path():
    class NoPathResult(FakeRawResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            del self.path

    boxes = FakeBoxes(cls=[], conf=[], xyxy=[])
    raw = NoPathResult(path="ignored.jpg", names=NAMES, boxes=boxes)
    parser = UltralyticsResultParser()

    prediction = parser.parse("fallback_source.jpg", raw)

    assert prediction.source_path == "fallback_source.jpg"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
