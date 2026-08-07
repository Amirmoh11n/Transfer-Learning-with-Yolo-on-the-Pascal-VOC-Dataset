import pytest

from training.hyperparameter_grid import HyperparameterGridBuilder


def test_build_returns_full_cartesian_product():
    grid = {"epochs": [10, 20], "batch_size": [4, 8]}
    combinations = HyperparameterGridBuilder.build(grid)

    assert len(combinations) == 4
    assert {"epochs": 10, "batch_size": 4} in combinations
    assert {"epochs": 10, "batch_size": 8} in combinations
    assert {"epochs": 20, "batch_size": 4} in combinations
    assert {"epochs": 20, "batch_size": 8} in combinations


def test_build_single_parameter_single_value():
    grid = {"epochs": [10]}
    combinations = HyperparameterGridBuilder.build(grid)
    assert combinations == [{"epochs": 10}]


def test_build_raises_on_empty_grid():
    with pytest.raises(ValueError):
        HyperparameterGridBuilder.build({})


def test_build_raises_on_empty_value_list():
    with pytest.raises(ValueError):
        HyperparameterGridBuilder.build({"epochs": []})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
