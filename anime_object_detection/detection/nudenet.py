from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.detect.nudenet import detect_with_nudenet, _LABELS

from .base import ObjectDetection


class NudeNetDetection(ObjectDetection):
    def _get_default_model(self) -> str:
        return 'Default'

    def _list_models(self) -> List[str]:
        return ['Default']

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        return 0.45, 0.25

    def _get_labels(self, model_name: str) -> List[str]:
        return _LABELS

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) -> \
            List[Tuple[Tuple[float, float, float, float], str, float]]:
        return detect_with_nudenet(image=image, iou_threshold=iou_threshold, score_threshold=score_threshold)
