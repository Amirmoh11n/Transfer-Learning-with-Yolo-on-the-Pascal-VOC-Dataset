"""
Tests for InferenceRunner. Uses a fake YOLOSegmentationModel-like object
so no real model loading, GPU, or network access is needed.
"""

import numpy as np
import pytest

from inference.inference_codes.inference_runner import InferenceRunner


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy

    def __len__(self):
        return len(self.cls)


class FakeRawResult:
    def __init__(self, path, names, boxes, masks=None, speed=None):
        self.path = path
        self.names = names
        self.boxes = boxes
        self.masks = masks
        self.speed = speed or {}

    def plot(self):
        return np.zeros((10, 10, 3), dtype="uint8")


class FakeModel:
    """Stands in for YOLOSegmentationModel: predict() returns
    caller-supplied fake results and records how it was called."""

    def __init__(self, results_to_return):
        self.results_to_return = results_to_return
        self.predict_called_with = None

    def predict(self, source, **kwargs):
        self.predict_called_with = {"source": source, **kwargs}
        return self.results_to_return


NAMES = {0: "person"}


def make_fake_result(path="img.jpg", n_detections=1):
    boxes = FakeBoxes(
        cls=[0] * n_detections,
        conf=[0.9] * n_detections,
        xyxy=[[0.0, 0.0, 5.0, 5.0]] * n_detections,
    )
    return FakeRawResult(path=path, names=NAMES, boxes=boxes)


def test_run_returns_one_prediction_per_raw_result():
    fake_results = [make_fake_result("a.jpg"), make_fake_result("b.jpg")]
    model = FakeModel(fake_results)
    runner = InferenceRunner(model)

    predictions = runner.run("some_source")

    assert len(predictions) == 2
    assert predictions[0].source_path == "a.jpg"
    assert predictions[1].source_path == "b.jpg"


def test_run_passes_confidence_threshold_to_model_predict():
    model = FakeModel([make_fake_result()])
    runner = InferenceRunner(model)

    runner.run("some_source", confidence_threshold=0.6)

    assert model.predict_called_with["source"] == "some_source"
    assert model.predict_called_with["conf"] == 0.6


def test_run_without_save_annotated_writes_no_files(tmp_path):
    model = FakeModel([make_fake_result("img.jpg")])
    runner = InferenceRunner(model)

    runner.run("some_source", save_annotated=False, output_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_run_with_save_annotated_writes_annotated_image(tmp_path):
    model = FakeModel([make_fake_result("img.jpg")])
    runner = InferenceRunner(model)

    runner.run("some_source", save_annotated=True, output_dir=str(tmp_path))

    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name == "img_annotated.jpg"


def test_run_with_zero_detections_still_returns_prediction():
    empty_result = make_fake_result("empty.jpg", n_detections=0)
    model = FakeModel([empty_result])
    runner = InferenceRunner(model)

    predictions = runner.run("some_source")

    assert len(predictions) == 1
    assert predictions[0].instances == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
