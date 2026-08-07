"""
Responsibility (SRP):
    Hold model/training configuration values. Pure data, no logic,
    no dependency on ultralytics itself - so this file stays stable
    even if we ever swap the underlying framework.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    # Which pretrained checkpoint to start from. Ultralytics auto-downloads
    # this from its GitHub releases the first time it's used, and caches
    # it locally afterwards (no re-download on subsequent runs).
    model_variant: str = "yolov8n-seg.pt"

    image_size: int = 640
    epochs: int = 50
    batch_size: int = 8
    device: str = "0"  # "0" = first GPU, "cpu" = CPU, "0,1" = multi-GPU

    # Where ultralytics writes training runs (weights, logs, plots).
    project_dir: str = "training/runs"
    run_name: str = "voc_seg_exp"

    # Optional: path to a previously-trained .pt checkpoint of OUR OWN,
    # to resume/fine-tune from instead of the stock pretrained weights.
    resume_from_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build a ModelConfig from a plain dict (e.g. loaded from YAML),
        ignoring unknown keys so config files can carry extra fields
        (like dataset paths) without breaking this constructor."""
        known_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
