from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.detect.booru_yolo import detect_with_booru_yolo, _DEFAULT_MODEL
from imgutils.generic.yolo import _open_models_for_repo_id

from .base import DeepGHSObjectDetection


class BooruYOLODetection(DeepGHSObjectDetection):
    def __init__(self):
        DeepGHSObjectDetection.__init__(self, repo_id='deepghs/booru_yolo')

    def _get_default_model(self) -> str:
        return _DEFAULT_MODEL

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        return 0.7, 0.25

    def _get_labels(self, model_name: str) -> List[str]:
        _, _, labels = _open_models_for_repo_id(self._repo_id)._open_model(model_name)
        return labels

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) -> \
            List[Tuple[Tuple[float, float, float, float], str, float]]:
        return detect_with_booru_yolo(image=image, model_name=model_name,
                                      iou_threshold=iou_threshold, conf_threshold=score_threshold)
