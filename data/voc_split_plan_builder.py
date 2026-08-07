"""
Responsibility (SRP):
    Build the final {"train": [...], "val": [...], "test": [...]}
    image-id plan for VOC, encoding the project's agreed policy:

        - VOC's official "train" list is further split (ourselves,
          deterministically) into our own train + val. "val" here is
          used for monitoring training and for hyperparameter tuning.
        - VOC's official "val" list is kept COMPLETELY UNTOUCHED and
          becomes our held-out "test" set - it is never used for
          training or tuning, only for a final, unbiased evaluation.

    This keeps the "which ids go where" POLICY in one place, separate
    from VOCSplitReader (which only knows how to read VOC's raw split
    files) and DatasetSplitter (which only knows how to divide a list).
"""

from dataclasses import dataclass
from typing import Dict, List

from data.dataset_splitter import DatasetSplitter
from data.voc_split_reader import VOCSplitReader


@dataclass
class SplitPlan:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    def as_dict(self) -> Dict[str, List[str]]:
        return {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}


class VOCSplitPlanBuilder:
    def __init__(
        self,
        split_reader: VOCSplitReader,
        dataset_splitter: DatasetSplitter,
        train_ratio: float = 0.85,
    ):
        self.split_reader = split_reader
        self.dataset_splitter = dataset_splitter
        self.train_ratio = train_ratio

    def build(self) -> SplitPlan:
        official_train_ids = self.split_reader.read_split("train")
        official_val_ids = self.split_reader.read_split("val")

        split_result = self.dataset_splitter.split(
            official_train_ids, first_ratio=self.train_ratio
        )

        return SplitPlan(
            train_ids=split_result.first_ids,
            val_ids=split_result.second_ids,
            test_ids=official_val_ids,  # untouched: held out for final eval only
        )
