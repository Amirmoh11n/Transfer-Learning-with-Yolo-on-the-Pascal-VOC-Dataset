"""
Entry point for the data pipeline.

Responsibility (SRP at this level):
    Read configuration, WIRE UP the small single-purpose objects from
    data/, build the 3-way {train, val, test} id plan, run the
    conversion for each of those 3 splits, print a report, and write
    the Ultralytics data.yaml that YOLO training/tuning will consume.

    This file itself contains no mask/polygon/path/splitting-policy
    logic - that all lives in data/. This is the "composition root"
    of the pipeline.

Split policy (decided and approved for this project):
    - VOC's official train.txt is further split OURSELVES into our own
      train (85%) + val (15%). "val" is used for monitoring training
      and for hyperparameter tuning.
    - VOC's official val.txt is kept COMPLETELY UNTOUCHED and becomes
      our held-out "test" set - never used for training or tuning,
      only for a final, unbiased evaluation.
    See data/voc_split_plan_builder.py for the implementation of this policy.

Usage (run from the project root, using -m so package imports resolve):
    python -m data.data_pipeline --config configs/voc_seg.yaml
"""

import argparse
from pathlib import Path

import yaml

from data.class_mapping import VOC_CLASSES
from data.dataset_downloader import VOCDatasetDownloader
from data.dataset_splitter import DatasetSplitter
from data.image_exporter import ImageExporter
from data.voc_instance_extractor import VOCInstanceExtractor
from data.voc_mask_loader import VOCMaskLoader
from data.voc_paths import VOCPaths
from data.voc_split_plan_builder import SplitPlan, VOCSplitPlanBuilder
from data.voc_split_reader import VOCSplitReader
from data.voc_to_yolo_converter import VOCToYOLOConverter
from data.yolo_label_writer import YOLOLabelWriter


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_split_plan(config: dict) -> SplitPlan:
    dataset_cfg = config["dataset"]
    split_cfg = config.get("split", {})

    voc_paths = VOCPaths(dataset_cfg["voc_root"])
    split_reader = VOCSplitReader(voc_paths)
    dataset_splitter = DatasetSplitter(seed=split_cfg.get("seed", 42))

    plan_builder = VOCSplitPlanBuilder(
        split_reader=split_reader,
        dataset_splitter=dataset_splitter,
        train_ratio=split_cfg.get("train_ratio", 0.85),
    )
    return plan_builder.build()


def build_converter(config: dict) -> VOCToYOLOConverter:
    dataset_cfg = config["dataset"]
    extraction_cfg = config.get("extraction", {})
    export_cfg = config.get("export", {})

    voc_paths = VOCPaths(dataset_cfg["voc_root"])
    mask_loader = VOCMaskLoader(voc_paths)
    instance_extractor = VOCInstanceExtractor(
        min_contour_area_px=extraction_cfg.get("min_contour_area_px", 4.0),
        polygon_epsilon_ratio=extraction_cfg.get("polygon_epsilon_ratio", 0.005),
        keep_largest_contour_only=extraction_cfg.get(
            "keep_largest_contour_only", True
        ),
    )
    label_writer = YOLOLabelWriter()
    image_exporter = ImageExporter(use_symlink=export_cfg.get("use_symlink", True))

    return VOCToYOLOConverter(
        voc_paths=voc_paths,
        mask_loader=mask_loader,
        instance_extractor=instance_extractor,
        label_writer=label_writer,
        image_exporter=image_exporter,
        output_root=Path(dataset_cfg["output_root"]),
    )


def write_data_yaml(output_root: Path) -> Path:
    """
    Writes the Ultralytics-style data.yaml pointing at the converted
    dataset, so training/tuning/testing can run as:
        yolo train data=<output_root>/data.yaml model=yolov8n-seg.pt
        yolo val   data=<output_root>/data.yaml split=test
    """
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(VOC_CLASSES)},
    }

    data_yaml_path = output_root / "data.yaml"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return data_yaml_path


def main():
    parser = argparse.ArgumentParser(description="VOC2012-Seg -> YOLO-seg converter")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/voc_seg.yaml",
        help="Path to pipeline config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

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


if __name__ == "__main__":
    main()
