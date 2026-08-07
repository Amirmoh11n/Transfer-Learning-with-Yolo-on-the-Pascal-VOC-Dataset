"""
Responsibility (SRP):
    Coordinate the other single-purpose modules to convert ONE output
    split (identified by whatever name the caller gives it, e.g.
    "train", "val", or "test") into a YOLO-seg dataset folder, given an
    explicit list of VOC image ids to include in that split.

    This class deliberately does NOT know about VOC's official split
    file names or how the 3-way train/val/test plan was decided - that
    policy lives in VOCSplitPlanBuilder. This converter only knows
    "given this list of ids, produce this output folder" - which keeps
    it reusable for any splitting policy we choose later.

    This class contains NO mask logic, NO path-format knowledge, and
    NO label-format knowledge itself - it only sequences calls to the
    modules that own those responsibilities. This keeps it easy to
    swap any one piece (e.g. a future COCOToYOLOConverter reuses
    YOLOLabelWriter and ImageExporter as-is).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from data.image_exporter import ImageExporter
from data.voc_instance_extractor import ExtractionWarning, VOCInstanceExtractor
from data.voc_mask_loader import VOCMaskLoader
from data.voc_paths import VOCPaths
from data.yolo_label_writer import YOLOLabelWriter


@dataclass
class ConversionReport:
    split_name: str
    total_images: int = 0
    images_with_zero_instances: int = 0
    total_instances_written: int = 0
    warnings: List[ExtractionWarning] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Split: {self.split_name}",
            f"  Images processed:            {self.total_images}",
            f"  Images with zero instances:  {self.images_with_zero_instances}",
            f"  Total instances written:     {self.total_instances_written}",
            f"  Warnings:                    {len(self.warnings)}",
        ]
        return "\n".join(lines)


class VOCToYOLOConverter:
    def __init__(
        self,
        voc_paths: VOCPaths,
        mask_loader: VOCMaskLoader,
        instance_extractor: VOCInstanceExtractor,
        label_writer: YOLOLabelWriter,
        image_exporter: ImageExporter,
        output_root: Path,
    ):
        self.voc_paths = voc_paths
        self.mask_loader = mask_loader
        self.instance_extractor = instance_extractor
        self.label_writer = label_writer
        self.image_exporter = image_exporter
        self.output_root = Path(output_root)

    def convert_split(self, split_name: str, image_ids: List[str]) -> ConversionReport:
        """
        Args:
            split_name: output folder name (e.g. "train", "val", "test") -
                purely a label for where files get written; carries no
                meaning about VOC's own file naming.
            image_ids: explicit list of VOC image ids to include in this
                split, as decided by the caller (e.g. VOCSplitPlanBuilder).
        """
        report = ConversionReport(split_name=split_name, total_images=len(image_ids))

        images_out_dir = self.output_root / "images" / split_name
        labels_out_dir = self.output_root / "labels" / split_name

        for image_id in image_ids:
            class_mask = self.mask_loader.load_class_mask(image_id)
            object_mask = self.mask_loader.load_object_mask(image_id)

            annotations, warnings = self.instance_extractor.extract(
                image_id, class_mask, object_mask
            )
            report.warnings.extend(warnings)

            if not annotations:
                report.images_with_zero_instances += 1
            report.total_instances_written += len(annotations)

            label_path = labels_out_dir / f"{image_id}.txt"
            self.label_writer.write(label_path, annotations)

            source_image = self.voc_paths.image_path(image_id)
            destination_image = images_out_dir / f"{image_id}.jpg"
            self.image_exporter.export(source_image, destination_image)

        return report
