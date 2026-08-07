"""
Entry point for hyperparameter grid search.

Usage (run from the project root):
    python -m training.tune_hyperparameters \\
        --model-config configs/model_config.yaml \\
        --grid-config configs/hyperparameter_grid.yaml \\
        --data-yaml dataset/voc2012_yolo_seg/data.yaml

This trains and evaluates one model per grid combination, scoring each
on the "val" split only. It never touches "test" - after finding the
best combination here, evaluate that one final model on "test"
separately (e.g. via evaluation/evaluate.py) for your unbiased report.
"""

import argparse
import json
from pathlib import Path

import yaml

from models.model_config import ModelConfig
from training.hyperparameter_tuner import HyperparameterTuner


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="YOLO-seg hyperparameter grid search")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument(
        "--grid-config", type=str, default="configs/hyperparameter_grid.yaml"
    )
    parser.add_argument("--data-yaml", type=str, required=True)
    parser.add_argument(
        "--output-report",
        type=str,
        default="training/runs/hyperparameter_search_report.json",
    )
    args = parser.parse_args()

    base_config = ModelConfig.from_dict(load_yaml(args.model_config))
    grid_config = load_yaml(args.grid_config)

    tuner = HyperparameterTuner(
        base_config=base_config,
        data_yaml_path=args.data_yaml,
        selection_metric=grid_config.get("selection_metric", "mask.map50_95"),
    )
    result = tuner.run(grid_config["grid"])

    print(result.summary())

    best = result.best_trial
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {
                "selection_metric": result.selection_metric,
                "best_config_overrides": best.config_overrides,
                "best_report": best.report.to_dict(),
                "all_trials": [
                    {"config_overrides": t.config_overrides, "report": t.report.to_dict()}
                    for t in result.trials
                ],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {report_path}")
    print(f"Best config overrides: {best.config_overrides}")


if __name__ == "__main__":
    main()
