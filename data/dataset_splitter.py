"""
Responsibility (SRP):
    Deterministically split a list of ids into two named subsets given
    a ratio and a random seed. Nothing dataset-specific here - this
    works for VOC ids, COCO ids, or any other list of strings/ints.

    This is the general-purpose splitter we discussed early in the
    project (originally deferred because VOC ships its own official
    train/val split). We now use it to further subdivide VOC's official
    "train" list into our own train/val, while VOC's official "val"
    list is kept untouched as our held-out test set.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitResult:
    first_ids: List[str]
    second_ids: List[str]


class DatasetSplitter:
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: fixed random seed, so the same input list + ratio
                always produces the same split - critical for comparing
                experiments across runs.
        """
        self.seed = seed

    def split(self, ids: List[str], first_ratio: float) -> SplitResult:
        """
        Args:
            ids: full list of ids to split.
            first_ratio: fraction (0 < first_ratio < 1) assigned to
                `first_ids`. The remainder goes to `second_ids`.

        Returns:
            SplitResult with two non-overlapping, order-shuffled lists
            whose union (as a set) equals the input ids exactly.
        """
        if not ids:
            raise ValueError("Cannot split an empty id list.")
        if not (0.0 < first_ratio < 1.0):
            raise ValueError(f"first_ratio must be between 0 and 1, got {first_ratio}")

        shuffled = list(ids)
        rng = random.Random(self.seed)
        rng.shuffle(shuffled)

        split_index = round(len(shuffled) * first_ratio)
        # Guard against a degenerate split swallowing every id when the
        # list is tiny (e.g. 3 items at ratio 0.99).
        split_index = max(1, min(split_index, len(shuffled) - 1))

        return SplitResult(
            first_ids=shuffled[:split_index],
            second_ids=shuffled[split_index:],
        )
