"""
Tests for MaskOverlapMetrics: pure numpy logic, hand-computable
expected values, no ultralytics/model/dataset needed at all.
"""

import numpy as np
import pytest

from evaluation.metrics import MaskOverlapMetrics


def square_mask(size=10, box=None) -> np.ndarray:
    """Build a boolean mask of shape (size, size). box=(r0,r1,c0,c1) marks True region."""
    mask = np.zeros((size, size), dtype=bool)
    if box is not None:
        r0, r1, c0, c1 = box
        mask[r0:r1, c0:c1] = True
    return mask


# ----------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------


def test_iou_identical_masks_is_one():
    mask = square_mask(box=(2, 6, 2, 6))
    assert MaskOverlapMetrics.iou(mask, mask) == pytest.approx(1.0)


def test_iou_disjoint_masks_is_zero():
    mask_a = square_mask(box=(0, 3, 0, 3))
    mask_b = square_mask(box=(6, 9, 6, 9))
    assert MaskOverlapMetrics.iou(mask_a, mask_b) == pytest.approx(0.0)


def test_iou_partial_overlap_known_value():
    # A: rows/cols 0..4 (4x4=16 px). B: rows/cols 2..6 (4x4=16 px).
    # Intersection: rows/cols 2..4 (2x2=4 px). Union: 16+16-4=28.
    mask_a = square_mask(size=10, box=(0, 4, 0, 4))
    mask_b = square_mask(size=10, box=(2, 6, 2, 6))
    expected = 4 / 28
    assert MaskOverlapMetrics.iou(mask_a, mask_b) == pytest.approx(expected)


def test_iou_both_empty_is_one():
    empty_a = square_mask()
    empty_b = square_mask()
    assert MaskOverlapMetrics.iou(empty_a, empty_b) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Dice
# ----------------------------------------------------------------------


def test_dice_identical_masks_is_one():
    mask = square_mask(box=(1, 5, 1, 5))
    assert MaskOverlapMetrics.dice(mask, mask) == pytest.approx(1.0)


def test_dice_partial_overlap_known_value():
    # Same setup as IoU test: intersection=4, |A|=16, |B|=16 -> dice = 2*4/32 = 0.25
    mask_a = square_mask(size=10, box=(0, 4, 0, 4))
    mask_b = square_mask(size=10, box=(2, 6, 2, 6))
    assert MaskOverlapMetrics.dice(mask_a, mask_b) == pytest.approx(0.25)


def test_dice_matches_iou_relationship():
    # Mathematical identity: dice = 2*iou / (1 + iou)
    mask_a = square_mask(size=10, box=(0, 5, 0, 5))
    mask_b = square_mask(size=10, box=(3, 8, 3, 8))

    iou = MaskOverlapMetrics.iou(mask_a, mask_b)
    dice = MaskOverlapMetrics.dice(mask_a, mask_b)
    expected_dice = (2 * iou) / (1 + iou)

    assert dice == pytest.approx(expected_dice)


# ----------------------------------------------------------------------
# Pixel accuracy
# ----------------------------------------------------------------------


def test_pixel_accuracy_identical_masks_is_one():
    mask = square_mask(box=(2, 6, 2, 6))
    assert MaskOverlapMetrics.pixel_accuracy(mask, mask) == pytest.approx(1.0)


def test_pixel_accuracy_fully_wrong_is_zero():
    mask_a = square_mask(size=4, box=(0, 4, 0, 4))  # all True
    mask_b = square_mask(size=4)  # all False
    assert MaskOverlapMetrics.pixel_accuracy(mask_a, mask_b) == pytest.approx(0.0)


def test_pixel_accuracy_shape_mismatch_raises():
    mask_a = square_mask(size=4)
    mask_b = square_mask(size=6)
    with pytest.raises(ValueError):
        MaskOverlapMetrics.pixel_accuracy(mask_a, mask_b)


# ----------------------------------------------------------------------
# mean_iou / mean_dice over multiple pairs
# ----------------------------------------------------------------------


def test_mean_iou_averages_correctly():
    perfect_pair = (square_mask(box=(0, 3, 0, 3)), square_mask(box=(0, 3, 0, 3)))
    zero_pair = (square_mask(box=(0, 3, 0, 3)), square_mask(box=(7, 10, 7, 10)))

    mean_iou = MaskOverlapMetrics.mean_iou([perfect_pair, zero_pair])
    assert mean_iou == pytest.approx(0.5)


def test_mean_iou_empty_list_raises():
    with pytest.raises(ValueError):
        MaskOverlapMetrics.mean_iou([])


def test_mean_dice_empty_list_raises():
    with pytest.raises(ValueError):
        MaskOverlapMetrics.mean_dice([])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
