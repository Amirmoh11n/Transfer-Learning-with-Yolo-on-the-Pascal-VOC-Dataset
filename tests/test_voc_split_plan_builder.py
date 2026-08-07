from unittest.mock import MagicMock

from data.dataset_splitter import DatasetSplitter
from data.voc_split_plan_builder import VOCSplitPlanBuilder


def make_fake_split_reader(official_train_ids, official_val_ids):
    fake_reader = MagicMock()

    def read_split(split_name):
        if split_name == "train":
            return list(official_train_ids)
        if split_name == "val":
            return list(official_val_ids)
        raise ValueError(f"unexpected split name: {split_name}")

    fake_reader.read_split.side_effect = read_split
    return fake_reader


def test_official_val_becomes_test_untouched():
    official_train_ids = [f"train_img_{i}" for i in range(100)]
    official_val_ids = [f"val_img_{i}" for i in range(20)]

    reader = make_fake_split_reader(official_train_ids, official_val_ids)
    builder = VOCSplitPlanBuilder(
        split_reader=reader,
        dataset_splitter=DatasetSplitter(seed=42),
        train_ratio=0.85,
    )

    plan = builder.build()

    assert sorted(plan.test_ids) == sorted(official_val_ids)


def test_official_train_is_split_into_our_train_and_val():
    official_train_ids = [f"train_img_{i}" for i in range(100)]
    official_val_ids = [f"val_img_{i}" for i in range(20)]

    reader = make_fake_split_reader(official_train_ids, official_val_ids)
    builder = VOCSplitPlanBuilder(
        split_reader=reader,
        dataset_splitter=DatasetSplitter(seed=42),
        train_ratio=0.85,
    )

    plan = builder.build()

    assert len(plan.train_ids) == 85
    assert len(plan.val_ids) == 15
    assert set(plan.train_ids) | set(plan.val_ids) == set(official_train_ids)
    assert set(plan.train_ids).isdisjoint(set(plan.val_ids))


def test_our_val_never_overlaps_with_test():
    official_train_ids = [f"train_img_{i}" for i in range(100)]
    official_val_ids = [f"val_img_{i}" for i in range(20)]

    reader = make_fake_split_reader(official_train_ids, official_val_ids)
    builder = VOCSplitPlanBuilder(
        split_reader=reader,
        dataset_splitter=DatasetSplitter(seed=42),
        train_ratio=0.85,
    )

    plan = builder.build()

    assert set(plan.val_ids).isdisjoint(set(plan.test_ids))
    assert set(plan.train_ids).isdisjoint(set(plan.test_ids))


def test_as_dict_has_expected_keys():
    official_train_ids = ["t1", "t2", "t3", "t4", "t5"]
    official_val_ids = ["v1", "v2"]

    reader = make_fake_split_reader(official_train_ids, official_val_ids)
    builder = VOCSplitPlanBuilder(
        split_reader=reader,
        dataset_splitter=DatasetSplitter(seed=42),
        train_ratio=0.8,
    )

    plan_dict = builder.build().as_dict()

    assert set(plan_dict.keys()) == {"train", "val", "test"}
