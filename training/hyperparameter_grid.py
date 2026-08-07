"""
Responsibility (SRP):
    Turn a hyperparameter grid, e.g.
        {"epochs": [10, 20], "batch_size": [4, 8]}
    into every combination as a list of override dicts:
        [{"epochs": 10, "batch_size": 4}, {"epochs": 10, "batch_size": 8},
         {"epochs": 20, "batch_size": 4}, {"epochs": 20, "batch_size": 8}]

    Pure data transformation. Knows nothing about ModelConfig, training,
    or evaluation - just cartesian products of dicts-of-lists.
"""

import itertools
from typing import Any, Dict, List


class HyperparameterGridBuilder:
    @staticmethod
    def build(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter.")

        for param_name, values in param_grid.items():
            if not values:
                raise ValueError(
                    f"Parameter '{param_name}' has an empty list of values."
                )

        keys = list(param_grid.keys())
        value_lists = list(param_grid.values())

        combinations = []
        for values in itertools.product(*value_lists):
            combinations.append(dict(zip(keys, values)))

        return combinations
