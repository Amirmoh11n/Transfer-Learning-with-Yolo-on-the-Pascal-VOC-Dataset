"""
Responsibility (SRP):
    Know the layout of a VOCdevkit/VOC2012 folder and hand back paths.
    This class does NOT read files or check file existence beyond
    basic sanity checks on directories. It is pure path resolution,
    so if VOC's folder layout ever changes, only this file needs edits.
"""

from pathlib import Path


class VOCPaths:
    def __init__(self, voc_root: str):
        """
        Args:
            voc_root: path to .../VOCdevkit/VOC2012
        """
        self.root = Path(voc_root)
        self.jpeg_images_dir = self.root / "JPEGImages"
        self.segmentation_class_dir = self.root / "SegmentationClass"
        self.segmentation_object_dir = self.root / "SegmentationObject"
        self.imagesets_segmentation_dir = self.root / "ImageSets" / "Segmentation"

        self._validate()

    def _validate(self) -> None:
        required_dirs = [
            self.jpeg_images_dir,
            self.segmentation_class_dir,
            self.segmentation_object_dir,
            self.imagesets_segmentation_dir,
        ]
        missing = [str(d) for d in required_dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                "VOCPaths: expected VOC2012 subdirectories are missing:\n"
                + "\n".join(missing)
            )

    def image_path(self, image_id: str) -> Path:
        return self.jpeg_images_dir / f"{image_id}.jpg"

    def class_mask_path(self, image_id: str) -> Path:
        return self.segmentation_class_dir / f"{image_id}.png"

    def object_mask_path(self, image_id: str) -> Path:
        return self.segmentation_object_dir / f"{image_id}.png"

    def split_file_path(self, split_name: str) -> Path:
        return self.imagesets_segmentation_dir / f"{split_name}.txt"
