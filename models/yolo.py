"""
Responsibility (SRP):
    Wrap ultralytics' YOLO model behind our own stable interface.

    Every other part of the project (training/, evaluation/,
    inference/) should depend on THIS class's public methods only -
    never import `ultralytics` directly. That way, if we ever need to
    swap the underlying framework, only this one file changes.

Design notes:
    - Loading is explicit (load_pretrained / load_checkpoint), not done
      implicitly in __init__. This makes "which weights are we starting
      from" a visible, deliberate step rather than a hidden side effect.
    - Methods raise a clear RuntimeError if called before loading,
      instead of failing deep inside ultralytics with a confusing
      AttributeError.
"""

from pathlib import Path
from typing import Any, Optional

from ultralytics import YOLO

from models.model_config import ModelConfig


class ModelNotLoadedError(RuntimeError):
    """Raised when an operation needs a loaded model but none was loaded yet."""


class YOLOSegmentationModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._underlying: Optional[YOLO] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        """
        Load the stock pretrained checkpoint named in config.model_variant
        (e.g. "yolov8n-seg.pt"). Ultralytics downloads this automatically
        from its GitHub releases on first use, then caches it locally -
        subsequent loads are offline.
        """
        self._underlying = YOLO(self.config.model_variant)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load OUR OWN previously-saved/trained weights, instead of the
        stock pretrained checkpoint."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self._underlying = YOLO(str(path))

    def is_loaded(self) -> bool:
        return self._underlying is not None

    def _ensure_loaded(self) -> YOLO:
        if self._underlying is None:
            raise ModelNotLoadedError(
                "No model loaded yet. Call load_pretrained() or "
                "load_checkpoint() before using this method."
            )
        return self._underlying

    # ------------------------------------------------------------------
    # Training / evaluation / inference
    # ------------------------------------------------------------------

    def train(self, data_yaml_path: str) -> Any:
        """
        Fine-tune on the dataset described by data_yaml_path (the
        Ultralytics-style data.yaml produced by our data pipeline).
        Returns ultralytics' training results object.
        """
        model = self._ensure_loaded()
        return model.train(
            data=data_yaml_path,
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            device=self.config.device,
            project=self.config.project_dir,
            name=self.config.run_name,
        )

    def validate(
        self, data_yaml_path: Optional[str] = None, split: str = "val"
    ) -> Any:
        """
        Run validation. If data_yaml_path is None, ultralytics reuses
        the dataset config from the most recent train() call.

        Args:
            split: which section of data.yaml to evaluate on - "val"
                (default, safe to call repeatedly during development/
                tuning) or "test" (the held-out set - call this only
                once, for a final, unbiased report).
        Returns ultralytics' metrics object.
        """
        model = self._ensure_loaded()
        kwargs = {
            "imgsz": self.config.image_size,
            "device": self.config.device,
            "split": split,
        }
        if data_yaml_path is not None:
            kwargs["data"] = data_yaml_path
        return model.val(**kwargs)

    def predict(self, source: str, **kwargs) -> Any:
        """
        Run inference on an image, folder, or video path.
        Extra ultralytics-specific options can be passed through kwargs
        (e.g. conf=0.25, save=True) without this wrapper needing to
        know about every possible option.
        """
        model = self._ensure_loaded()
        return model.predict(source=source, imgsz=self.config.image_size, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """Save the current model weights to output_path."""
        model = self._ensure_loaded()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
