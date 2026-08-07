"""
Unit tests for YOLOSegmentationModel and ModelConfig.

We mock ultralytics.YOLO everywhere here. This means:
    - Tests run in milliseconds, no network access, no GPU needed.
    - We verify OUR wrapper logic (state checks, argument passing,
      error handling) - NOT ultralytics' own internals, which is
      ultralytics' responsibility to test, not ours.

If you want to verify the REAL integration (actual download, actual
predict on a real image), that's a separate, slower "smoke test" -
not something to run on every code change.
"""

from unittest.mock import MagicMock, patch

import pytest

from models.model_config import ModelConfig
from models.yolo import ModelNotLoadedError, YOLOSegmentationModel


def make_config(**overrides) -> ModelConfig:
    defaults = dict(
        model_variant="yolov8n-seg.pt",
        image_size=640,
        epochs=10,
        batch_size=4,
        device="cpu",
        project_dir="training/runs",
        run_name="test_run",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ----------------------------------------------------------------------
# ModelConfig
# ----------------------------------------------------------------------


def test_model_config_from_dict_ignores_unknown_keys():
    raw = {
        "model_variant": "yolov8s-seg.pt",
        "epochs": 20,
        "some_unrelated_dataset_field": "should_be_ignored",
    }
    config = ModelConfig.from_dict(raw)
    assert config.model_variant == "yolov8s-seg.pt"
    assert config.epochs == 20
    assert not hasattr(config, "some_unrelated_dataset_field")


# ----------------------------------------------------------------------
# Guard behavior: operations before loading
# ----------------------------------------------------------------------


def test_operations_raise_before_load():
    model = YOLOSegmentationModel(make_config())
    assert model.is_loaded() is False

    with pytest.raises(ModelNotLoadedError):
        model.train("configs/dummy_data.yaml")

    with pytest.raises(ModelNotLoadedError):
        model.validate()

    with pytest.raises(ModelNotLoadedError):
        model.predict("dummy.jpg")

    with pytest.raises(ModelNotLoadedError):
        model.save("out.pt")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


@patch("models.yolo.YOLO")
def test_load_pretrained_uses_configured_variant(mock_yolo_class):
    config = make_config(model_variant="yolov8m-seg.pt")
    model = YOLOSegmentationModel(config)

    model.load_pretrained()

    mock_yolo_class.assert_called_once_with("yolov8m-seg.pt")
    assert model.is_loaded() is True


@patch("models.yolo.YOLO")
def test_load_checkpoint_raises_if_file_missing(mock_yolo_class):
    model = YOLOSegmentationModel(make_config())
    with pytest.raises(FileNotFoundError):
        model.load_checkpoint("nonexistent/path/weights.pt")
    mock_yolo_class.assert_not_called()


@patch("models.yolo.YOLO")
def test_load_checkpoint_success(mock_yolo_class, tmp_path):
    fake_checkpoint = tmp_path / "my_weights.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint, just a placeholder")

    model = YOLOSegmentationModel(make_config())
    model.load_checkpoint(str(fake_checkpoint))

    mock_yolo_class.assert_called_once_with(str(fake_checkpoint))
    assert model.is_loaded() is True


# ----------------------------------------------------------------------
# train() / validate() / predict() argument passing
# ----------------------------------------------------------------------


@patch("models.yolo.YOLO")
def test_train_passes_config_values_to_underlying_model(mock_yolo_class):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    config = make_config(epochs=30, batch_size=16, image_size=512, device="0")
    model = YOLOSegmentationModel(config)
    model.load_pretrained()

    model.train("configs/data.yaml")

    mock_instance.train.assert_called_once_with(
        data="configs/data.yaml",
        epochs=30,
        imgsz=512,
        batch=16,
        device="0",
        project=config.project_dir,
        name=config.run_name,
    )


@patch("models.yolo.YOLO")
def test_validate_without_data_path_omits_data_kwarg(mock_yolo_class):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config())
    model.load_pretrained()
    model.validate()

    called_kwargs = mock_instance.val.call_args.kwargs
    assert "data" not in called_kwargs


@patch("models.yolo.YOLO")
def test_validate_with_data_path_includes_data_kwarg(mock_yolo_class):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config())
    model.load_pretrained()
    model.validate("configs/data.yaml")

    called_kwargs = mock_instance.val.call_args.kwargs
    assert called_kwargs["data"] == "configs/data.yaml"


@patch("models.yolo.YOLO")
def test_validate_defaults_to_val_split(mock_yolo_class):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config())
    model.load_pretrained()
    model.validate("configs/data.yaml")

    called_kwargs = mock_instance.val.call_args.kwargs
    assert called_kwargs["split"] == "val"


@patch("models.yolo.YOLO")
def test_validate_can_request_test_split(mock_yolo_class):
    """The held-out test split must be reachable, but only when the
    caller explicitly asks - "val" stays the safe default."""
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config())
    model.load_pretrained()
    model.validate("configs/data.yaml", split="test")

    called_kwargs = mock_instance.val.call_args.kwargs
    assert called_kwargs["split"] == "test"


@patch("models.yolo.YOLO")
def test_predict_forwards_extra_kwargs(mock_yolo_class):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config(image_size=640))
    model.load_pretrained()
    model.predict("image.jpg", conf=0.4, save=True)

    mock_instance.predict.assert_called_once_with(
        source="image.jpg", imgsz=640, conf=0.4, save=True
    )


@patch("models.yolo.YOLO")
def test_save_creates_parent_directory(mock_yolo_class, tmp_path):
    mock_instance = MagicMock()
    mock_yolo_class.return_value = mock_instance

    model = YOLOSegmentationModel(make_config())
    model.load_pretrained()

    output_path = tmp_path / "nested" / "dir" / "weights.pt"
    model.save(str(output_path))

    assert output_path.parent.is_dir()
    mock_instance.save.assert_called_once_with(str(output_path))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
