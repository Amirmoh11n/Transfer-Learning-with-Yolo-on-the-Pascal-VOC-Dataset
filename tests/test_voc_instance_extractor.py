"""
Validates VOCInstanceExtractor logic using synthetic, hand-built masks.
No real VOC files needed - this catches regressions in the mask->polygon
logic independent of any dataset being present on disk.
"""

import numpy as np

from data.voc_instance_extractor import VOCInstanceExtractor


def make_masks(height=20, width=20):
    class_mask = np.zeros((height, width), dtype=np.uint8)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    return class_mask, object_mask


def test_single_rectangular_instance_extracts_one_polygon():
    class_mask, object_mask = make_masks()

    # Draw a 6x8 rectangle: rows 5..10, cols 4..11 -> VOC class "car" = 7
    class_mask[5:11, 4:12] = 7
    object_mask[5:11, 4:12] = 1  # instance id 1

    extractor = VOCInstanceExtractor(min_contour_area_px=1.0)
    annotations, warnings = extractor.extract("synthetic_1", class_mask, object_mask)

    assert warnings == []
    assert len(annotations) == 1

    ann = annotations[0]
    assert ann.yolo_class_id == 6  # VOC label 7 ("car") -> yolo id 6
    assert len(ann.polygon) >= 3
    for x, y in ann.polygon:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_two_disjoint_instances_of_different_classes():
    class_mask, object_mask = make_masks()

    # Instance 1: "person" (VOC label 15) -> yolo id 14
    class_mask[2:6, 2:6] = 15
    object_mask[2:6, 2:6] = 1

    # Instance 2: "dog" (VOC label 12) -> yolo id 11
    class_mask[12:16, 12:16] = 12
    object_mask[12:16, 12:16] = 2

    extractor = VOCInstanceExtractor(min_contour_area_px=1.0)
    annotations, warnings = extractor.extract("synthetic_2", class_mask, object_mask)

    assert warnings == []
    assert len(annotations) == 2

    class_ids = sorted(ann.yolo_class_id for ann in annotations)
    assert class_ids == [11, 14]


def test_ignore_boundary_and_background_are_excluded():
    class_mask, object_mask = make_masks()

    class_mask[5:10, 5:10] = 255  # boundary/ignore only, no real class
    object_mask[5:10, 5:10] = 1

    extractor = VOCInstanceExtractor(min_contour_area_px=1.0)
    annotations, warnings = extractor.extract("synthetic_3", class_mask, object_mask)

    assert annotations == []
    assert len(warnings) == 1
    assert warnings[0].reason == "no valid class pixels in region"


def test_tiny_noise_below_min_area_is_dropped():
    class_mask, object_mask = make_masks()

    class_mask[0, 0] = 3  # single pixel
    object_mask[0, 0] = 1

    extractor = VOCInstanceExtractor(min_contour_area_px=4.0)
    annotations, warnings = extractor.extract("synthetic_4", class_mask, object_mask)

    assert annotations == []
    assert len(warnings) == 1
    assert warnings[0].reason == "no contour above min area threshold"


if __name__ == "__main__":
    test_single_rectangular_instance_extracts_one_polygon()
    test_two_disjoint_instances_of_different_classes()
    test_ignore_boundary_and_background_are_excluded()
    test_tiny_noise_below_min_area_is_dropped()
    print("All tests passed.")
