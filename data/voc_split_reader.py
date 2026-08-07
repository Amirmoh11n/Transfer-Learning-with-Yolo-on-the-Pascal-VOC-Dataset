"""
Responsibility (SRP):
    Read VOC's official ImageSets/Segmentation/{split}.txt files and
    return the list of image ids for that split.

    We deliberately reuse VOC's own official split rather than building
    our own splitter for this dataset (see project discussion: our
    custom DatasetSplitter is reserved for datasets that don't ship
    an official split, e.g. a future COCO-based version).
"""

from typing import List

from data.voc_paths import VOCPaths


class VOCSplitReader:
    def __init__(self, paths: VOCPaths):
        self.paths = paths

    def read_split(self, split_name: str) -> List[str]:
        """
        Args:
            split_name: "train" or "val" (matches VOC's file names).

        Returns:
            List of image ids, e.g. ["2011_000455", "2011_000456", ...]
        """
        split_file = self.paths.split_file_path(split_name)
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]

        if not image_ids:
            raise ValueError(f"Split file is empty: {split_file}")

        return image_ids
