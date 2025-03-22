import os

import gradio as gr
from diffusers.utils import load_image


from detection import EyesDetection, FaceDetection, HeadDetection, PersonDetection, HandDetection, CensorDetection, \
    HalfBodyDetection, NudeNetDetection, BooruYOLODetection

_GLOBAL_CSS = """
.limit-height {
    max-height: 55vh;
}
"""
import time

def measure_execution_time(func, *args, **kwargs):
    """
    Measures and prints the execution time of a function.

    Parameters:
        func (callable): The function to measure.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function execution.
    """
    start_time = time.time()  # Start time
    result = func(*args, **kwargs)  # Call the function
    end_time = time.time()  # End time

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    return result

img_path = "/workspace/reference_image0.png"
image = load_image(img_path)
model_name = 'face_detect_v1.4_s'
print(FaceDetection().detect(image, model_name, ))
print(measure_execution_time(FaceDetection().detect, image, model_name, ))

# if __name__ == '__main__':
#     with gr.Blocks(css=_GLOBAL_CSS) as demo:
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML('<h2 style="text-align: center;">Object Detections For Anime</h2>')
#                 gr.Markdown('This is the online demo for detection functions of '
#                             '[imgutils.detect](https://dghs-imgutils.deepghs.org/main/api_doc/detect/index.html). '
#                             'You can try them yourselves with `pip install dghs-imgutils`.')

#         with gr.Row():
#             with gr.Tabs():
#                 with gr.Tab('Face Detection'):
#                     FaceDetection().make_ui()
#                 with gr.Tab('Head Detection'):
#                     HeadDetection().make_ui()
#                 with gr.Tab('Person Detection'):
#                     PersonDetection().make_ui()
#                 with gr.Tab('Half Body Detection'):
#                     HalfBodyDetection().make_ui()
#                 with gr.Tab('Eyes Detection'):
#                     EyesDetection().make_ui()
#                 with gr.Tab('Hand Detection'):
#                     HandDetection().make_ui()
#                 with gr.Tab('Censor Point Detection'):
#                     CensorDetection().make_ui()
#                 with gr.Tab('NudeNet'):
#                     NudeNetDetection().make_ui()
#                 with gr.Tab('BooruYOLO'):
#                     BooruYOLODetection().make_ui()

#     demo.queue(os.cpu_count()).launch()
