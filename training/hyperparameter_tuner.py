"""
Responsibility (SRP):
    Run a grid search over hyperparameters: for each combination, train
    a fresh model and evaluate it, then report which combination scored
    best on a chosen metric.

    Critical policy this class enforces by construction:
        - Each trial calls model.train(data_yaml_path), which trains on
          the "train" split defined in that data.yaml.
        - Each trial is then scored via SegmentationEvaluator.evaluate(),
          which runs ultralytics val() on the "val" split of that same
          data.yaml.
        - The "test" split is NEVER referenced anywhere in this file.
          Final unbiased reporting on test is a separate, deliberate
          step the caller takes once, after tuning is done - not
          something this class can accidentally do.

    Model/evaluator creation is injectable (model_factory /
    evaluator_factory) so this class can be unit-tested with lightweight
    fakes instead of real ultralytics training runs.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from evaluation.evaluate import SegmentationEvaluator
from evaluation.metrics import SegmentationEvaluationResult
from models.model_config import ModelConfig
from models.yolo import YOLOSegmentationModel
from training.hyperparameter_grid import HyperparameterGridBuilder

ModelFactory = Callable[[ModelConfig], YOLOSegmentationModel]
EvaluatorFactory = Callable[[YOLOSegmentationModel], SegmentationEvaluator]


@dataclass
class TrialResult:
    trial_index: int
    config_overrides: Dict[str, Any]
    report: SegmentationEvaluationResult

    def metric(self, metric_path: str) -> float:
        """
        metric_path is "branch.field", e.g. "mask.map50_95" or "box.map50".
        Kept as a small string path (rather than a hardcoded metric)
        so the tuner can be re-scored on a different metric later
        without re-running training.
        """
        branch_name, field_name = metric_path.split(".")
        branch = getattr(self.report, branch_name)
        return float(getattr(branch, field_name))


@dataclass
class HyperparameterSearchResult:
    trials: List[TrialResult] = field(default_factory=list)
    selection_metric: str = "mask.map50_95"

    @property
    def best_trial(self) -> TrialResult:
        if not self.trials:
            raise ValueError("No trials to select a best result from.")
        return max(self.trials, key=lambda t: t.metric(self.selection_metric))

    def summary(self) -> str:
        best = self.best_trial
        lines = [
            f"Hyperparameter search ({len(self.trials)} trials, "
            f"selection metric = {self.selection_metric})",
            "-" * 60,
        ]
        for trial in self.trials:
            score = trial.metric(self.selection_metric)
            marker = "  <-- BEST" if trial is best else ""
            lines.append(
                f"  Trial {trial.trial_index}: {trial.config_overrides} "
                f"-> {self.selection_metric}={score:.4f}{marker}"
            )
        return "\n".join(lines)


class HyperparameterTuner:
    def __init__(
        self,
        base_config: ModelConfig,
        data_yaml_path: str,
        model_factory: Optional[ModelFactory] = None,
        evaluator_factory: Optional[EvaluatorFactory] = None,
        selection_metric: str = "mask.map50_95",
    ):
        self.base_config = base_config
        self.data_yaml_path = data_yaml_path
        self.model_factory = model_factory or (lambda cfg: YOLOSegmentationModel(cfg))
        self.evaluator_factory = evaluator_factory or (
            lambda model: SegmentationEvaluator(model)
        )
        self.selection_metric = selection_metric

    def run(self, param_grid: Dict[str, List[Any]]) -> HyperparameterSearchResult:
        overrides_list = HyperparameterGridBuilder.build(param_grid)

        trials: List[TrialResult] = []
        for index, overrides in enumerate(overrides_list):
            trial_config = replace(self.base_config, **overrides)

            model = self.model_factory(trial_config)
            model.load_pretrained()
            model.train(self.data_yaml_path)  # trains on data.yaml's "train" split

            evaluator = self.evaluator_factory(model)
            report = evaluator.evaluate(self.data_yaml_path)  # scores on "val" split only

            trials.append(
                TrialResult(trial_index=index, config_overrides=overrides, report=report)
            )

        return HyperparameterSearchResult(
            trials=trials, selection_metric=self.selection_metric
        )
