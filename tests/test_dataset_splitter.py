from data.dataset_splitter import DatasetSplitter


def test_split_produces_no_overlap_and_covers_all_ids():
    ids = [f"img_{i}" for i in range(100)]
    splitter = DatasetSplitter(seed=42)

    result = splitter.split(ids, first_ratio=0.85)

    assert set(result.first_ids).isdisjoint(set(result.second_ids))
    assert set(result.first_ids) | set(result.second_ids) == set(ids)


def test_split_ratio_is_approximately_correct():
    ids = [f"img_{i}" for i in range(200)]
    splitter = DatasetSplitter(seed=42)

    result = splitter.split(ids, first_ratio=0.85)

    assert len(result.first_ids) == 170
    assert len(result.second_ids) == 30


def test_same_seed_gives_reproducible_split():
    ids = [f"img_{i}" for i in range(50)]

    result_a = DatasetSplitter(seed=7).split(ids, first_ratio=0.8)
    result_b = DatasetSplitter(seed=7).split(ids, first_ratio=0.8)

    assert result_a.first_ids == result_b.first_ids
    assert result_a.second_ids == result_b.second_ids


def test_different_seed_gives_different_split():
    ids = [f"img_{i}" for i in range(50)]

    result_a = DatasetSplitter(seed=1).split(ids, first_ratio=0.8)
    result_b = DatasetSplitter(seed=2).split(ids, first_ratio=0.8)

    assert result_a.first_ids != result_b.first_ids


def test_empty_id_list_raises():
    splitter = DatasetSplitter(seed=42)
    try:
        splitter.split([], first_ratio=0.8)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_ratio_raises():
    splitter = DatasetSplitter(seed=42)
    ids = ["a", "b", "c"]
    for bad_ratio in (0.0, 1.0, -0.1, 1.5):
        try:
            splitter.split(ids, first_ratio=bad_ratio)
            assert False, f"expected ValueError for ratio={bad_ratio}"
        except ValueError:
            pass


def test_tiny_list_never_produces_an_empty_side():
    ids = ["a", "b", "c"]
    splitter = DatasetSplitter(seed=42)

    result = splitter.split(ids, first_ratio=0.99)

    assert len(result.first_ids) >= 1
    assert len(result.second_ids) >= 1
