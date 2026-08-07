"""
Entry point for running inference.

Usage (run from the project root):
    python -m inference.inference_codes.inference \\
        --source path/to/image_or_folder \\
        --checkpoint training/runs/voc_seg_exp/weights/best.pt \\
        --save-annotated --save-json inference/inference_output/predictions.json

If --checkpoint is omitted, the stock pretrained weights named in
configs/model_config.yaml are used instead of a fine-tuned checkpoint.
"""

import argparse
import json
from pathlib import Path

import yaml

from inference.inference_codes.inference_runner import InferenceRunner
from models.model_config import ModelConfig
from models.yolo import YOLOSegmentationModel


def main():
    parser = argparse.ArgumentParser(description="Run YOLO-seg inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained .pt checkpoint. If omitted, uses the "
        "stock pretrained weights from --model-config.",
    )
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--output-dir", type=str, default="inference/inference_output")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save all predictions as a JSON file.",
    )
    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        config = ModelConfig.from_dict(yaml.safe_load(f))

    model = YOLOSegmentationModel(config)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    else:
        model.load_pretrained()

    runner = InferenceRunner(model)
    predictions = runner.run(
        args.source,
        confidence_threshold=args.confidence,
        save_annotated=args.save_annotated,
        output_dir=args.output_dir,
    )

    for prediction in predictions:
        print(
            f"{prediction.source_path}: {len(prediction.instances)} instance(s), "
            f"{prediction.inference_time_ms:.1f} ms"
        )
        for instance in prediction.instances:
            print(
                f"  - {instance.class_name} ({instance.confidence:.2f}) "
                f"box={instance.box_xyxy}"
            )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in predictions], f, indent=2)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
