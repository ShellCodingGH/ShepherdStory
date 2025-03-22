import re
from typing import List, Tuple

from imgutils.data import ImageTyping
from imgutils.detect.face import detect_faces

from .base import DeepGHSObjectDetection

def _parse_model_name(model_name: str):
    matching = re.fullmatch(r'^face_detect_(?P<version>[\s\S]+?)_(?P<level>[\s\S]+?)$', model_name)
    return matching.group('version'), matching.group('level')


class FaceDetection(DeepGHSObjectDetection):
    def __init__(self):
        DeepGHSObjectDetection.__init__(self, 'deepghs/anime_face_detection')

    def _get_default_model(self) -> str:
        return 'face_detect_v1.4_s'

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        return 0.7, 0.25

    def _get_labels(self, model_name: str) -> List[str]:
        return ['face']

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) -> \
            List[Tuple[Tuple[float, float, float, float], str, float]]:
        version, level = _parse_model_name(model_name)
        return detect_faces(image, level=level, version=version,
                            conf_threshold=score_threshold, iou_threshold=iou_threshold)