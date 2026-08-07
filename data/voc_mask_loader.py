"""
Responsibility (SRP):
    Load VOC's PNG masks into numpy arrays. Nothing else.
    No polygon logic, no class mapping - just "give me the raw pixel
    array for this image id".

    VOC's segmentation PNGs are palette-indexed images: each pixel's
    integer value IS the class/instance id (not an RGB color to decode).
    Pillow gives us this directly via convert("P") -> array of indices.
"""

import numpy as np
from PIL import Image

from data.voc_paths import VOCPaths


class VOCMaskLoader:
    def __init__(self, paths: VOCPaths):
        self.paths = paths

    def load_class_mask(self, image_id: str) -> np.ndarray:
        """Pixel values: 0=background, 1..20=class, 255=ignore/boundary."""
        path = self.paths.class_mask_path(image_id)
        return self._load_indexed_png(path)

    def load_object_mask(self, image_id: str) -> np.ndarray:
        """Pixel values: 0=background, 1..N=instance id, 255=ignore/boundary."""
        path = self.paths.object_mask_path(image_id)
        return self._load_indexed_png(path)

    @staticmethod
    def _load_indexed_png(path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")
        with Image.open(path) as img:
            # "P" mode = palette-indexed; pixel value is the class/instance id itself.
            return np.array(img.convert("P"), dtype=np.uint8)
