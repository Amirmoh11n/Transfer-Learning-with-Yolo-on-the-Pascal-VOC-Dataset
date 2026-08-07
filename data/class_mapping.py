"""
Responsibility (SRP):
    Own the VOC2012 class taxonomy and the mapping between VOC's
    pixel-value class indices (1..20 in SegmentationClass masks)
    and YOLO's 0-indexed class ids.

Nothing else in this module. No file I/O, no mask logic.
"""

from typing import List

# Official VOC2012 class order (matches the devkit's colormap indices 1..20).
# Index 0 in this list -> VOC pixel value 1 -> YOLO class id 0.
VOC_CLASSES: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Reserved VOC mask pixel values that are NOT object classes.
VOC_BACKGROUND_VALUE = 0
VOC_IGNORE_BOUNDARY_VALUE = 255


def voc_label_to_yolo_class_id(voc_pixel_value: int) -> int:
    """
    Convert a VOC SegmentationClass pixel value (1..20) to a
    0-indexed YOLO class id (0..19).

    Raises ValueError for background(0)/ignore(255)/out-of-range values,
    since callers must filter those out before calling this.
    """
    if voc_pixel_value in (VOC_BACKGROUND_VALUE, VOC_IGNORE_BOUNDARY_VALUE):
        raise ValueError(
            f"Pixel value {voc_pixel_value} is background/ignore, not a class."
        )
    if not (1 <= voc_pixel_value <= len(VOC_CLASSES)):
        raise ValueError(f"Pixel value {voc_pixel_value} is out of VOC class range.")
    return voc_pixel_value - 1


def num_classes() -> int:
    return len(VOC_CLASSES)
