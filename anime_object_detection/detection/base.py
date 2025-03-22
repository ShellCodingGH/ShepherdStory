import os.path
from functools import lru_cache
from typing import List, Tuple

import gradio as gr
from hbutils.color import rnd_colors
from hfutils.operate import get_hf_fs
from hfutils.utils import hf_fs_path, parse_hf_fs_path
from imgutils.data import ImageTyping


def _v_fix(v):
    return int(round(v))


def _bbox_fix(bbox):
    return tuple(map(_v_fix, bbox))


class ObjectDetection:
    @lru_cache()
    def get_default_model(self) -> str:
        return self._get_default_model()

    def _get_default_model(self) -> str:
        raise NotImplementedError

    @lru_cache()
    def list_models(self) -> List[str]:
        return self._list_models()

    def _list_models(self) -> List[str]:
        raise NotImplementedError

    @lru_cache()
    def get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        return self._get_default_iou_and_score(model_name)

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        raise NotImplementedError

    @lru_cache()
    def get_labels(self, model_name: str) -> List[str]:
        return self._get_labels(model_name)

    def _get_labels(self, model_name: str) -> List[str]:
        raise NotImplementedError

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) \
            -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        raise NotImplementedError

    def _gr_detect(self, image: ImageTyping, model_name: str,
                   iou_threshold: float = 0.7, score_threshold: float = 0.25) \
            -> gr.AnnotatedImage:
        labels = self.get_labels(model_name=model_name)
        _colors = list(map(str, rnd_colors(len(labels))))
        _color_map = dict(zip(labels, _colors))
        return gr.AnnotatedImage(
            value=(image, [
                (_bbox_fix(bbox), label) for bbox, label, _ in
                self.detect(image, model_name, iou_threshold, score_threshold)
            ]),
            color_map=_color_map,
            label='Labeled',
        )

    def make_ui(self):
        with gr.Row():
            with gr.Column():
                default_model_name = self.get_default_model()
                model_list = self.list_models()
                gr_input_image = gr.Image(type='pil', label='Original Image')
                gr_model = gr.Dropdown(model_list, value=default_model_name, label='Model')
                with gr.Row():
                    iou, score = self.get_default_iou_and_score(default_model_name)
                    gr_iou_threshold = gr.Slider(0.0, 1.0, iou, label='IOU Threshold')
                    gr_score_threshold = gr.Slider(0.0, 1.0, score, label='Score Threshold')

                gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                gr_output_image = gr.AnnotatedImage(label="Labeled")

            gr_submit.click(
                self._gr_detect,
                inputs=[
                    gr_input_image,
                    gr_model,
                    gr_iou_threshold,
                    gr_score_threshold,
                ],
                outputs=[gr_output_image],
            )


class DeepGHSObjectDetection(ObjectDetection):
    def __init__(self, repo_id: str):
        self._repo_id = repo_id

    def _get_default_model(self) -> str:
        raise NotImplementedError

    def _list_models(self) -> List[str]:
        hf_fs = get_hf_fs()
        return [
            os.path.dirname(parse_hf_fs_path(path).filename)
            for path in hf_fs.glob(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='model',
                filename='*/model.onnx'
            ))
        ]

    def _get_default_iou_and_score(self, model_name: str) -> Tuple[float, float]:
        raise NotImplementedError

    def _get_labels(self, model_name: str) -> List[str]:
        raise NotImplementedError

    def detect(self, image: ImageTyping, model_name: str,
               iou_threshold: float = 0.7, score_threshold: float = 0.25) \
            -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        raise NotImplementedError
