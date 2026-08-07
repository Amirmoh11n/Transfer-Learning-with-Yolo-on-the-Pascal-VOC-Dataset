"""
Responsibility (SRP):
    Orchestrate inference: call YOLOSegmentationModel.predict(), parse
    each raw result via UltralyticsResultParser, and optionally save
    annotated output images. Contains no ultralytics-attribute
    knowledge itself (that's UltralyticsResultParser's job) and no
    model-loading knowledge (that's YOLOSegmentationModel's job).
"""

from pathlib import Path
from typing import List, Optional

from inference.inference_codes.prediction_result import ImagePrediction
from inference.inference_codes.result_parser import UltralyticsResultParser
from models.yolo import YOLOSegmentationModel


class InferenceRunner:
    def __init__(
        self,
        model: YOLOSegmentationModel,
        result_parser: Optional[UltralyticsResultParser] = None,
    ):
        self.model = model
        self.result_parser = result_parser or UltralyticsResultParser()

    def run(
        self,
        source: str,
        confidence_threshold: float = 0.25,
        save_annotated: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[ImagePrediction]:
        """
        Args:
            source: image path, folder path, or video path (anything
                ultralytics' predict() accepts).
            confidence_threshold: minimum detection confidence to keep.
            save_annotated: if True, save a visualized (boxes+masks
                drawn) copy of each image to output_dir.
            output_dir: where annotated images go. Defaults to
                inference/inference_output.
        """
        raw_results = self.model.predict(source, conf=confidence_threshold)

        predictions = []
        for raw_result in raw_results:
            source_path = str(getattr(raw_result, "path", source) or source)
            prediction = self.result_parser.parse(source_path, raw_result)
            predictions.append(prediction)

            if save_annotated:
                self._save_annotated(raw_result, source_path, output_dir)

        return predictions

    @staticmethod
    def _save_annotated(raw_result, source_path: str, output_dir: Optional[str]) -> None:
        import cv2  # local import: only needed when actually saving images

        output_dir_path = Path(output_dir or "inference/inference_output")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        annotated_image = raw_result.plot()  # ultralytics' own visualization (reuse-first)
        filename = Path(source_path).stem + "_annotated.jpg"
        cv2.imwrite(str(output_dir_path / filename), annotated_image)
