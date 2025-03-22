import os.path
import re
from typing import List, Tuple

from hfutils.operate import get_hf_fs
from hfutils.utils import hf_fs_path, parse_hf_fs_path
from imgutils.data import ImageTyping
from imgutils.detect import detect_person

from .base import ObjectDetection

_VERSIONS = {
    '': 'v0',
    'plus_': 'v1',
    'plus_v1.1_': 'v1.1',
}


def _parse_model_name(model_name: str):
    matching = re.fullmatch(r'^person_detect_(?P<content>[\s\S]+?)best_(?P<level>[\s\S]+?)$', model_name)
    return _VERSIONS[matching.group('content')], matching.group('level')


class PersonDetection(ObjectDetection):
    def __init__(self):
        self.repo_id = 'deepghs/imgutils-models'

    def _get_default_model(self) -> str:
        return 'person_detect_plus_v1.1_best_m'

    def _list_models(self) -> List[str]:
        hf_fs = get_hf_fs()
        return [
            os.path.splitext(os.path.basename(parse_hf_fs_path(path).filename))[0]
            for path in hf_fs.glob(hf_fs_path(
                repo_id=self.repo_id,
                repo_type='model',
                filename='person_detect/*.onnx',
            ))
        ]

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        return 0.5, 0.3

    def _get_labels(self, model_name: str) -> List[str]:
        return ['person']

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) -> \
            List[Tuple[Tuple[float, float, float, float], str, float]]:
        version, level = _parse_model_name(model_name)
        return detect_person(image=image, level=level, version=version,
                             iou_threshold=iou_threshold, conf_threshold=score_threshold)
