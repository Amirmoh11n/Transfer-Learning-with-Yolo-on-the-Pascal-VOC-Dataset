"""
main.py - MVP entry point for the YOLOv8-seg VOC2012 fine-tuning project.

This is the single top-level script that ties together every module
built so far:
    data/        -> VOC2012 -> YOLO-seg conversion (train/val/test)
    models/      -> YOLOSegmentationModel wrapper (ultralytics, encapsulated)
    training/    -> hyperparameter grid search
    evaluation/  -> precision/recall/mAP + IoU/Dice metrics
    inference/   -> running the trained model on new images

main.py itself contains NO business logic - it only wires together the
composition-root functions/classes that already exist in each module.
Each subcommand below mirrors one stage of the v0.0 MVP pipeline.

Split policy reminder (see data/voc_split_plan_builder.py):
    train -> model fitting
    val   -> monitoring + hyperparameter tuning (safe to reuse repeatedly)
    test  -> ONE final, unbiased report only - pass --split test
             explicitly to `evaluate`; never the default.

Usage:
    python main.py prepare-data
    python main.py train    [--checkpoint path/to/resume.pt]
    python main.py tune
    python main.py evaluate [--split val|test] [--save-report path.json]
    python main.py infer --source path/to/image_or_folder [--checkpoint ...]
    python main.py all                     # prepare-data -> train -> evaluate(val)

Run `python main.py <command> --help` for each command's full options.
"""

import argparse
import json
from pathlib import Path

import yaml

from data.data_pipeline import build_converter, build_split_plan, write_data_yaml
from data.dataset_downloader import VOCDatasetDownloader
from evaluation.evaluate import SegmentationEvaluator
from inference.inference_codes.inference_runner import InferenceRunner
from models.model_config import ModelConfig
from models.yolo import YOLOSegmentationModel
from training.hyperparameter_tuner import HyperparameterTuner

DEFAULT_VOC_CONFIG = "configs/voc_seg.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_config.yaml"
DEFAULT_GRID_CONFIG = "configs/hyperparameter_grid.yaml"


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_data_yaml_path(voc_config_path: str) -> str:
    """Given voc_seg.yaml, compute where its data.yaml lives
    (dataset.output_root/data.yaml), so commands don't need the path
    typed out separately if --data-yaml wasn't given."""
    config = _load_yaml(voc_config_path)
    output_root = Path(config["dataset"]["output_root"])
    return str(output_root / "data.yaml")


def _build_model(model_config_path: str, checkpoint: str = None) -> YOLOSegmentationModel:
    config = ModelConfig.from_dict(_load_yaml(model_config_path))
    model = YOLOSegmentationModel(config)
    if checkpoint:
        model.load_checkpoint(checkpoint)
    else:
        model.load_pretrained()
    return model


# ----------------------------------------------------------------------
# Stage 1: data preparation
# ----------------------------------------------------------------------


def cmd_prepare_data(args) -> str:
    """VOC2012 -> YOLO-seg conversion for all 3 splits. Returns the
    resulting data.yaml path (used by later stages).

    Auto-downloads VOC2012 first IF it isn't already present at
    configs/voc_seg.yaml's voc_root - safe to run repeatedly, since
    an already-present dataset is detected and left untouched."""
    config = _load_yaml(args.config)

    downloader = VOCDatasetDownloader(
        voc_root=config["dataset"]["voc_root"],
        download_url=config["dataset"].get("download_url"),
    )
    downloader.ensure_available()

    plan = build_split_plan(config)
    converter = build_converter(config)

    for split_name, image_ids in plan.as_dict().items():
        report = converter.convert_split(split_name, image_ids)
        print(report.summary())
        if report.warnings:
            print(f"  (showing first 5 of {len(report.warnings)} warnings)")
            for w in report.warnings[:5]:
                print(f"    - image={w.image_id} instance={w.instance_id} reason={w.reason}")

    output_root = Path(config["dataset"]["output_root"])
    data_yaml_path = write_data_yaml(output_root)
    print(f"\nWrote {data_yaml_path}")
    return str(data_yaml_path)


# ----------------------------------------------------------------------
# Stage 2: training
# ----------------------------------------------------------------------


def cmd_train(args) -> YOLOSegmentationModel:
    """Fine-tune on the "train" split (ultralytics also monitors "val"
    during training automatically). Returns the trained model, ready
    for immediate evaluation/inference without reloading from disk."""
    data_yaml_path = args.data_yaml or _resolve_data_yaml_path(args.voc_config)
    model = _build_model(args.model_config, args.checkpoint)
    model.train(data_yaml_path)
    return model


def cmd_tune(args) -> None:
    """Grid-search hyperparameters, scored on "val" only. "test" is
    never referenced here by construction (see HyperparameterTuner)."""
    data_yaml_path = args.data_yaml or _resolve_data_yaml_path(args.voc_config)
    base_config = ModelConfig.from_dict(_load_yaml(args.model_config))
    grid_config = _load_yaml(args.grid_config)

    tuner = HyperparameterTuner(
        base_config=base_config,
        data_yaml_path=data_yaml_path,
        selection_metric=grid_config.get("selection_metric", "mask.map50_95"),
    )
    result = tuner.run(grid_config["grid"])
    print(result.summary())
    print(f"\nBest config overrides: {result.best_trial.config_overrides}")


# ----------------------------------------------------------------------
# Stage 3: evaluation
# ----------------------------------------------------------------------


def cmd_evaluate(args) -> None:
    """
    Evaluate a (trained or stock-pretrained) model.

    --split defaults to "val" (safe to run repeatedly during
    development). Pass --split test deliberately for the final,
    one-time unbiased report - do this only once you're done tuning.
    """
    data_yaml_path = args.data_yaml or _resolve_data_yaml_path(args.voc_config)
    model = _build_model(args.model_config, args.checkpoint)

    evaluator = SegmentationEvaluator(model)
    report = evaluator.evaluate(data_yaml_path, split=args.split)
    print(evaluator.summary(report))

    if args.split == "test":
        print(
            "\nNOTE: this was a TEST-split evaluation. If you plan to "
            "keep tuning/training afterwards, treat this number as a "
            "final report, not a signal to act on further."
        )

    if args.save_report:
        evaluator.save_report(report, args.save_report)
        print(f"\nWrote {args.save_report}")


# ----------------------------------------------------------------------
# Stage 4: inference
# ----------------------------------------------------------------------


def cmd_infer(args) -> None:
    """Run inference on new images/folder/video with a (trained or
    stock-pretrained) model."""
    model = _build_model(args.model_config, args.checkpoint)
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


# ----------------------------------------------------------------------
# Full MVP pipeline
# ----------------------------------------------------------------------


def cmd_all(args) -> None:
    """
    Runs the complete v0.0 MVP pipeline end-to-end:
        prepare-data -> train -> evaluate (on "val")

    Deliberately does NOT touch "test" - that final, one-time check is
    a separate, deliberate step: `python main.py evaluate --split test`.
    """
    print("=" * 60)
    print("STAGE 1/3: Preparing data (VOC2012 -> YOLO-seg)")
    print("=" * 60)
    data_yaml_path = cmd_prepare_data(args)

    print("\n" + "=" * 60)
    print("STAGE 2/3: Training")
    print("=" * 60)
    args.data_yaml = data_yaml_path
    args.checkpoint = None
    model = cmd_train(args)

    print("\n" + "=" * 60)
    print("STAGE 3/3: Evaluating on val")
    print("=" * 60)
    evaluator = SegmentationEvaluator(model)
    report = evaluator.evaluate(data_yaml_path, split="val")
    print(evaluator.summary(report))

    print(
        "\nMVP pipeline complete. When you're done tuning, run a final "
        "unbiased check with:\n"
        "  python main.py evaluate --split test "
        f"--checkpoint {model.config.project_dir}/{model.config.run_name}/weights/best.pt"
    )


# ----------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8-seg fine-tuning on VOC2012 - MVP entry point"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_data = subparsers.add_parser("prepare-data", help="Convert VOC2012 -> YOLO-seg format")
    p_data.add_argument("--config", type=str, default=DEFAULT_VOC_CONFIG)
    p_data.set_defaults(func=cmd_prepare_data)

    p_train = subparsers.add_parser("train", help="Fine-tune the model")
    p_train.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    p_train.add_argument("--voc-config", type=str, default=DEFAULT_VOC_CONFIG)
    p_train.add_argument("--data-yaml", type=str, default=None)
    p_train.add_argument("--checkpoint", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_tune = subparsers.add_parser("tune", help="Grid-search hyperparameters on val")
    p_tune.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    p_tune.add_argument("--grid-config", type=str, default=DEFAULT_GRID_CONFIG)
    p_tune.add_argument("--voc-config", type=str, default=DEFAULT_VOC_CONFIG)
    p_tune.add_argument("--data-yaml", type=str, default=None)
    p_tune.set_defaults(func=cmd_tune)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    p_eval.add_argument("--voc-config", type=str, default=DEFAULT_VOC_CONFIG)
    p_eval.add_argument("--data-yaml", type=str, default=None)
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate on. Use 'test' only for the "
        "final, one-time unbiased report.",
    )
    p_eval.add_argument("--save-report", type=str, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Run inference on new images")
    p_infer.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    p_infer.add_argument("--checkpoint", type=str, default=None)
    p_infer.add_argument("--source", type=str, required=True)
    p_infer.add_argument("--confidence", type=float, default=0.25)
    p_infer.add_argument("--save-annotated", action="store_true")
    p_infer.add_argument("--output-dir", type=str, default="inference/inference_output")
    p_infer.add_argument("--save-json", type=str, default=None)
    p_infer.set_defaults(func=cmd_infer)

    p_all = subparsers.add_parser(
        "all", help="Run the full MVP pipeline: prepare-data -> train -> evaluate(val)"
    )
    p_all.add_argument("--config", type=str, default=DEFAULT_VOC_CONFIG)
    p_all.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
