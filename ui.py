import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from transformers import CLIPProcessor, CLIPModel
import time
import numpy as np

import re
import json
import os
import random

from PIL import Image
from PIL import ImageOps
import gradio as gr
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from huggingface_hub import login

from model_base import Txt2imgModel
from model_base import Img2poseModel
from model_base import ChatbotModel
from model_base import IPModel
from my_utils.my_utils import extract_names_without_rare_tokens
from my_utils.my_utils import generate_string_hash
from my_utils.my_utils import preprocess_pose_prompt
from my_utils.my_utils import preprocess_pose_image
from my_utils.my_utils import add_padding
from my_utils.my_utils import get_most_similar_sentence
from my_utils.my_utils import get_control_image
from my_utils.my_utils import get_image_path
from my_utils.my_utils import get_image_paths
from my_utils.my_utils import adapt_portrait
from my_utils.my_utils import concat_images
from my_utils.my_utils import get_character_name
from my_utils.my_utils import combine_pose_image_multiple_characters
from my_utils.my_utils import preprocess_pose_image_single_characters
from my_utils.my_utils import save_character_to_database
from my_utils.my_utils import crop_sides
from my_utils.my_utils import modify_character
from my_utils.my_utils import extract_rare_tokens
from my_utils.my_utils import get_most_similar_sentence
from my_utils.my_utils import exist_name
from my_utils.my_utils import populate_dropdowns
from my_utils.my_utils import extract_character_name_by_image
from my_utils.my_utils import draw_text_bubble_on_image
from my_utils.my_utils import save_pdf
from my_utils.my_utils import get_character_prototype
from evaluate import calculate_similarities
from api_infer import queue_prompt


HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)

save_pdf()
working_folder = "/root"
txt2img_prompt_prefix = "(animestyle, anime)1.5, (full body portrait, full body portrait, full body portrait, full body portrait, full body portrait, full body portrait)1.5, "
prompt_suffix = ", perfection style, perfection, perfect, midjourneyv6.1, max details, Photo, product photo, (expressive)1.7, masterpiece, raw quality, best quality, HD, extremely detailed, high definition, stunning beautiful, soft features, masterpiece, raw quality, best quality, extremely detailed, stunning beautiful, high definition, HD,(masterpiece)1.2, (best quality)1.2, (ultra-detailed)1.2, (unity 8k wallpaper)1.2, (illustration)1.1, (anime style)1.1, intricate, fluid simulation, sharp edges, (glossy)1.2, (Smooth)1.2, (detailed eyes)1.2"
no_person_negative_prefix = "(person, human, girl, boy, person)1.5, (person, human being, woman, man, person)1.4"
negative_prompt_suffix = ",  (round face, fat face, fat, obese, chubby)1.2, (blushing, red face, round face, wide face, big head, huge head)1.7, (thin face, tall face, wide face, big head, huge head)1.2, dark color, dark, dark lighting, Stable_Yogis_Animetoon_Negatives, Stable_Yogis_Animetoon_Negatives-neg, DeepNegative_xl_v1, ac_neg1, SimplePositiveXLv2, unaestheticXL_cbp62, unaestheticXL_cbp62 -neg, negativeXL_D, lowres, worst quality, low quality, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad face, bad anatomy, disfigured, poorly drawn,deformed, mutation, malformation, deformed, mutated, disfigured, (deformed eyes)1.1, (bad face)1.2, bad hands, poorly drawn hands, malformed hands, (extra arms)1.2, (extra legs)1.2, (Fused body)1.1, (Fused hands)1.1, Fused legs+, missing arms, missing limb, (extra digit)1.1, fewer digits, floating limbs, disconnected limbs, inaccurate limb, bad fingers, missing fingers, ugly face, (long body)1.2"
pose_skeleton_prompt_prefix = "(full body portrait)2.0, full body portrait, (shorts)1.5, (t-shirts)1.5, a person "
res_folder = "/root/comfy/ComfyUI/output/refer"
txt2img_res_folder = "/root/comfy/ComfyUI/output/txt2img"
txt2img_prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 230222432303012,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "ddim",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "AtomicXL.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "width": 992,
      "height": 1280,
      "batch_size": 4
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": "(full body portrait)1.5, a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, ",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "txt2img/ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
"""
txt2img_prompt_query = json.loads(txt2img_prompt_text)



prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 598003559864927,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "ddim",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "25",
        0
      ],
      "positive": [
        "17",
        0
      ],
      "negative": [
        "23",
        1
      ],
      "latent_image": [
        "19",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "AtomicXL.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "6": {
    "inputs": {
      "text": "(full body portrait)1.5, an elegant high school girl, very long purple hair, purple blazer, white inner shirt, long purple school uniform trousers, , full-body portrait of Yuki leaping into action, her long hair trailing like a banner, landing a fierce spinning heel kick to a bully’s chest, a sunlit school courtyard, cherry blossoms drifting, perfection style, perfection, perfect, midjourneyv6.1, max details, Photo, product photo, (expressive)1.7, masterpiece, raw quality, best quality, HD, extremely detailed, high definition, stunning beautiful, soft features, masterpiece, raw quality, best quality, extremely detailed, stunning beautiful, high definition, HD,(masterpiece)1.2, (best quality)1.2, (ultra-detailed)1.2, (unity 8k wallpaper)1.2, (illustration)1.1, (anime style)1.1, intricate, fluid simulation, sharp edges, (glossy)1.2, (Smooth)1.2, (detailed eyes)1.2",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "(yellow, light color, bright, yellow)1.5, text, watermark",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "refer/ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "13": {
    "inputs": {
      "image": "komi1 (1).png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "16": {
    "inputs": {
      "control_net_name": "controlnet_sdxl_union_promax.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "17": {
    "inputs": {
      "strength": 0.8000000000000002,
      "conditioning": [
        "23",
        0
      ],
      "control_net": [
        "16",
        0
      ],
      "image": [
        "21",
        0
      ]
    },
    "class_type": "ControlNetApply",
    "_meta": {
      "title": "Apply ControlNet (OLD)"
    }
  },
  "19": {
    "inputs": {
      "pixels": [
        "13",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "21": {
    "inputs": {
      "image": "pose1.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "22": {
    "inputs": {
      "image": [
        "13",
        0
      ],
      "vae": [
        "4",
        2
      ],
      "latent_size": [
        "19",
        0
      ]
    },
    "class_type": "ACN_ReferencePreprocessor",
    "_meta": {
      "title": "Reference Preproccessor 🛂🅐🅒🅝"
    }
  },
  "23": {
    "inputs": {
      "strength": 0.7000000000000002,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "control_net": [
        "24",
        0
      ],
      "image": [
        "22",
        0
      ]
    },
    "class_type": "ACN_AdvancedControlNetApply_v2",
    "_meta": {
      "title": "Apply Advanced ControlNet 🛂🅐🅒🅝"
    }
  },
  "24": {
    "inputs": {
      "reference_type": "reference_adain",
      "style_fidelity": 0.5000000000000001,
      "ref_weight": 0.7000000000000002
    },
    "class_type": "ACN_ReferenceControlNet",
    "_meta": {
      "title": "Reference ControlNet 🛂🅐🅒🅝"
    }
  },
  "25": {
    "inputs": {
      "lora_name": "sdxl_lightning_4step_lora.safetensors",
      "strength_model": 1.0000000000000002,
      "model": [
        "4",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  }
}
"""
prompt_query = json.loads(prompt_text)
custom_css = """
/* Overall container: white background, no default rounding. */

footer {visibility: hidden}


.gradio-container {
    border-radius: 0px !important;
    border: 1px solid #000000 !important; 
    background-color: #f9f9f9;
    font-family: "Segoe UI", Tahoma, sans-serif;
    width: 850px!important;
    max-height: 850px;
    overflow: auto !important;

}


/* SHARP CORNERS FOR IMAGES: Remove rounding on gr.Image components. */
.gradio-container .gr-image img {
    border-radius: 0px !important; 
    border: 1px solid #000000 !important; 
}

/* No extra margin/padding for this particular element. */
#sharp_corner {
    border-radius: 0px !important;
    border: 1px solid #000000 !important;
    margin: 0.5px !important;   
    padding: 0px !important;
    # position: relative;       /* So absolutely positioned children are placed relative to this container */
    # display: inline-block;
}


#gallery {
    overflow-y: auto; !important;
}

#gallery .gallery-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    
}
#gallery .gallery-item img {
    max-width: 100%;
    height: auto;
    margin-bottom: 8px;
}
#gallery .gallery-item-caption {
    font-size: 14px;
    color: #555;
    text-align: center;
}


#overlap-container {
    position: relative;       /* So absolutely positioned children are placed relative to this container */
    display: inline-block;    /* Keeps container sized to the image */
}

/* The textbox that overlaps the image */
#top-text-big1 {
    position: absolute;
    top: 30%;                 /* Move to ~50% down from top of container */
    left: 50%;                /* Move to ~50% from left of container */
    transform: translate(-50%, -50%); /* Center the textbox horizontally and vertically */
    background-color: rgba(255, 0, 0, 1.); /* Semi-transparent background */
    transition: opacity 0.5s ease-in-out;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    width: 95%;             /* Fixed width, adjust as desired */
    text-align: center;
    z-index: 9999;
    visibility: visible;
    
}

#top-text-small1 {
    position: absolute;
    top:86%;                 /* Move to ~50% down from top of container */
    left: 50%;                /* Move to ~50% from left of container */
    transform: translate(-50%, -50%); /* Center the textbox horizontally and vertically */
    background-color: rgba(255, 0, 0, 1.); /* Semi-transparent background */
    transition: opacity 0.5s ease-in-out;
    padding: 10px;
    # border: 1px solid #ccc;
    # border-radius: 5px;
    width: 95%;             /* Fixed width, adjust as desired */
    height: 200px;
    text-align: center;
    z-index: 9999;
    visibility: visible;

}

#top-text-small2 {
    position: absolute;
    top: 23%;                 /* Move to ~50% down from top of container */
    left: 50%;                /* Move to ~50% from left of container */
    transform: translate(-50%, -50%); /* Center the textbox horizontally and vertically */
    background-color: rgba(255, 0, 0, 1.); /* Semi-transparent background */
    transition: opacity 0.5s ease-in-out;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    width: 95%;             /* Fixed width, adjust as desired */
    height: 200px;
    text-align: center;
    z-index: 9999;
    visibility: visible;
}

#top-text-big2 {
    position: absolute;
    top: 70%;                 /* Move to ~50% down from top of container */
    left: 50%;                /* Move to ~50% from left of container */
    transform: translate(-50%, -50%); /* Center the textbox horizontally and vertically */
    background-color: rgba(255, 0, 0, 1.); /* Semi-transparent background */
    transition: opacity 0.5s ease-in-out;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    width: 95%;             /* Fixed width, adjust as desired */
    text-align: center;
    z-index: 9999;
    visibility: visible;
    
}

#container {
    position: relative;
    width: 100%;
    height: 777px;  /* Adjust height as needed */
}

#container .component {
    width: 100%;
    height: 100%;
}
#container .component img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
#textbox {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10;
}

"""
pipe_chatbot, pipe_txt2img, pipe_pose, pipe_ip = ChatbotModel(), Txt2imgModel(), Img2poseModel(), IPModel()
pipe_chatbot.multi_thread_load_model()
# pipe_txt2img.multi_thread_load_model()
# pipe_pose.multi_thread_load_model()
pipe_ip.multi_thread_load_model()
# heights = [1000, 904] # A
# widths = [904, 880] # B
selected_image = False
height, width = 800, 560 # height, width

width_txt2img, height_txt2img = 992, 1280
width1, height1 = 992, 1280
width2, height2 = 1008, 800
rare_tokens_file = "rare_tokens.txt"
database_file = "database.txt"
selected_image_path = ""
character_name_value = ""
rare_token = ""
show_prompt_toggle = True
poses = ["dance", "flexing", "jumping", "laying", "sitting", "standing", ]
llm_pose_prompt_prefix = """
Given a user’s prompt, either split this prompt into 4 sub-prompts suitable to Stable Diffusion that describes 4 scenes; OR extend user’s prompt(if it’s too short) into 4 prompts that describes 4 images. Pay SPECIAL ATTENTION to the mentioning of any camera angle/motion in user's prompt, if you see a camera angle/motion, you should treat the parts of the user's prompt relevant to this camera angle/motion as a single outpur prompt(separated by a semicolon(;)). 
 some camera angle/motions could be the below, split with a semicolon(;) at these camera angle/motions:
 - camera angles: e.g. side-view, front-view, closeup, bird's eye view
 - facial expressions: e.g. smiling, crying, frowning
 - body movement: e.g. standing, doing a leg kick, jumping
 - body pose: e.g. sitting, standing, lying down
 
Each of your 4 output prompts needs to be separated by a semicolon(;). It is VERY IMPORTANT that you make sure you have 3 and 3 only semicolons(;), so in total 4 prompts. you need to preserve the dialogue encompassed by double quotation marks (""), as well as preserve the brackets and numbers(e.g.  (side-view)1.5 )
Each of the 4 sub-prompts needs to be different in some way, don’t repeat for the 4 sub-prompts. It is VERY IMPORTANT that you make sure you don’t include my instruction prompt. And you must follow this:
- If the user’s prompt doesn’t indicate an image that include a person’s face or some body parts, DO NOT include anything that describes a person, person’s face and these body parts, but only describe the parts of the person’s other body parts / clothings etc that the user indicates in his prompt; similarly, if it DOES indicate a person’s face or any of the person’s body parts, include them. Also make good attention in your output prompt of different views, e.g. side-view, front-view, close-up. It is VERY IMPORTANT to reflect these view in your prompt. It is VERY IMPORTANT that you extend user’s prompt(if it’s too short) into 4 prompts that describes 4 images. 
===========================================================================================
Examples: 
Example 1:
    - User’s prompt: 
full body portrait, an elegant girl in dark blue school uniform eating ice-cream with (side view)1.5 saying "what's up", very long hair, very long dark purple hair, very long dark purple hair, full body portrait, beautiful eyes, sparklinhg eyes, short dark blue school uniform skirt, full body portrait, sleepy eyes, smiling, standing on the ground saying "How are you doing?", long dark purple hair, very long hair, short dark blue school uniform skirt
    - Your output:
full body portrait, an elegant girl in dark blue school uniform eating ice cream with (side view)1.5 saying "what's up"; full body portrait, an elegant girl in dark blue school uniform smiling; full body portrait, an elegant girl in dark blue school uniform standing on the ground saying "How are you doing?"; full body portrait, an elegant girl in dark blue school uniform walks towards you with (close-up shot)1.5

Explanation: 
here the 4 sub-prompts are separated by 3 semicolons(;)in all 4 prompts in your suggested output(separated by a semicolon) has the words “full body portrait”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; The dialogue indicated by double quotation marks("")(e.g. "what's up", "How are you doing?") and brackets and numbers(e.g.(close-up shot)1.5) are preserved here.
-------------------------------------------------------------------------------------------
Example 2:
    - User’s prompt: 
full body shot, (side-view)1.5, A bustling high school courtyard filled with students laughing and chatting, Among them is Yuki, a high school teenager, sitting among the students, chatting with them, her very long black hair flowing behind her like a dark river, catching the afternoon sun, Her black school uniform fits her perfectly, the skirt swaying as she moves, upper body portrait, Yuki’s beautiful, sparkling eyes glimmer with warmth as she smiles at her friends, but a flicker of something deeper—resolve—passes through them as she glances toward a group practicing martial arts nearby, Full-body portrait: Yuki does a high kick in her black school uniform, hands clutching her school bag, Her long black hair cascades past her waist, a striking contrast to the crisp uniform, as she turns away from the martial arts group, hiding her interest, upper body shot, (side-view)1.5, As she walks past, her reflection in a classroom window reveals a steely determination, hinting at skills she keeps buried beneath her ordinary high school life
    - Your output:
full body shot, (side-view)1.5, A bustling high school courtyard filled with students laughing and chatting, Among them is Yuki, a high school teenager, sitting among the students, chatting with them, her very long black hair flowing behind her like a dark river, catching the afternoon sun, Her black school uniform fits her perfectly, the skirt swaying as she moves; upper body portrait, Yuki’s beautiful, sparkling eyes glimmer with warmth as she smiles at her friends, but a flicker of something deeper—resolve—passes through them as she glances toward a group practicing martial arts nearby; Full-body portrait, Yuki does a high kick in her black school uniform, hands clutching her school bag, Her long black hair cascades past her waist, a striking contrast to the crisp uniform, as she turns away from the martial arts group, hiding her interest; upper body shot, (side-view)1.5, As she walks past, her reflection in a classroom window reveals a steely determination, hinting at skills she keeps buried beneath her ordinary high school life

Explanation: 
here the 4 sub-prompts are separated by 3 semicolons(;). in the 1st prompts in your suggested output(separated by a semicolon) has the words “full body shot”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 2nd prompts in your suggested output(separated by a semicolon) has the words “upper body portrait”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 3rd prompts in your suggested output(separated by a semicolon) has the words “Full-body portrait”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 4th prompts in your suggested output(separated by a semicolon) has the words “upper body shot”, which suggests that you should include the anime girl’s face, and her entire body in your prompt. The brackets and numbers(e.g.(close-up shot)1.5) are preserved here.
-------------------------------------------------------------------------------------------
Example 3:
    - User’s prompt: 
Wide shot, A quiet street at dusk, Yuki walking alone, her very long black hair swaying with each step, the black school uniform skirt brushing her knees as she adjusts her bag, Upper-body close-up, She pauses, her sparkling eyes widening as shouts echo from a nearby alley, curiosity and concern warring on her face, Side-view, Yuki peeks around the corner, her long hair spilling over her shoulder, and sees a street fight—a classmate of hers, struggling against a tougher opponent, Close-up, Her classmate’s face, bruised and desperate, flashes in the dim light, his opponent looming over him as Yuki’s breath catches.
    - Your output:
Wide shot, A quiet street at dusk, Yuki walking alone, her very long black hair swaying with each step, the black school uniform skirt brushing her knees as she adjusts her bag; Upper-body close-up, She pauses, her sparkling eyes widening as shouts echo from a nearby alley, curiosity and concern warring on her face; Side-view, Yuki peeks around the corner, her long hair spilling over her shoulder, and sees a street fight—a classmate of hers, struggling against a tougher opponent; Close-up, Her classmate’s face, bruised and desperate, flashes in the dim light, his opponent looming over him as Yuki’s breath catches.

Explanation: 
here the 4 sub-prompts are separated by 3 semicolons(;). in the 1st prompts in your suggested output(separated by a semicolon) has the words “Wide shot,”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 2nd prompts in your suggested output(separated by a semicolon) has the words “Upper-body close-up”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 3rd prompts in your suggested output(separated by a semicolon) has the words “Side-view”, which suggests that you may or may not include the anime girl’s face, and her entire body in your prompt, meaning your portrait type can be all of “full body portrait”, “lower body portrait”, or “upper body portrait” ; in the 4th prompts in your suggested output(separated by a semicolon) has the words “Close-up”, which suggests that you should include the anime girl’s face, and her entire body in your prompt. The brackets and numbers(e.g.(close-up shot)1.5) are preserved here.
-------------------------------------------------------------------------------------------
Example 4:
    - User’s prompt: 
an elegant anime girl wearing military uniform long pants, short blonde hair, dark green military cap, dark green military uniform long pants, dark green military uniform, dark alley background, does a (sidekick)1.5 with full body portrait saying "Tz...", (punches)1.5 forward with (side-view)1.5 full body portrait saying "Shoot... Gonna die here...", does a (sidekick)1.75 with full body portrait, got stabbed from the back by a knife with shocked facial expression and (side-view)2.0 with upper body portrait, 
    - Your output:
dark alley background, an elegant anime girl wearing military uniform long pants does a (sidekick)1.5 with full body portrait saying "Tz…"; dark alley background, an elegant anime girl wearing military uniform long pants (punches)1.5 forward with (side-view)1.5 full body portrait saying "Shoot... Gonna die here…"; dark alley background, does a (sidekick)1.75 with lower body portrait; dark alley background, an elegant anime girl got stabbed from the back by a knife with shocked facial expression and (side-view)2.0 with upper body portrait
Explanation: 
here the 4 sub-prompts are separated by 3 semicolons(;). in the 1st and 2nd prompt in your suggested output(separated by a semicolon) has the words “full body portrait”, which suggests that you should include the anime girl’s face, and her entire body in your prompt; in the 3rd prompt in your suggested output(separated by a semicolon) has the words “lower body portrait”, which which suggests that you should NOT include the anime girl’s face, and her entire body in your prompt, only the lower body part and the Lower body clothings;  in the 4th prompt in your suggested output(separated by a semicolon) has the words “upper body portrait”, which which suggests that you should NOT include the anime girl’s lower body parts, and her entire body in your prompt, only the upper body part and the upper body clothings
-------------------------------------------------------------------------------------------
Example 5:
    - User’s prompt: 
a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, Upper body portrait, the girl bumped into her (middle-aged female teacher)1.4 at the door, the girl (looks up)1.5, (scared and shocked)1.5 (close-up)1.5 upper body portrait
    - Your output:
upper body portrait, the girl bumped into her middle-aged female teacher at the door; the girl (looks up)1.5, (scared and shocked)1.5 (close-up)1.5 upper body portrait; a cute high school girl (cries out for help)1.2, full body portrait; a cute high school girl falls down on the ground, side-view, full body portrait
Explanation: 
Here this prompt is too short for 4 images and does not have enough motions to split into 4 images, thus you need to extend. Here the user’s prompt only suggests 2 images that can be split into, you need to extend for 2 more. One image suggested by user’s prompt is “the girl bumped into her middle-aged female teacher at the door”, the other ones is “the girl (looks up)1.5, (scared and shocked)1.5”. here the 4 sub-prompts are separated by 3 semicolons(;).  in the 1st and 2nd prompt in your suggested output(separated by a semicolon) has the words “upper body portrait”, which suggests that you should include the anime girl’s face, and her upper body in your prompt; in the 3rd and 4th prompt in your suggested output(separated by a semicolon), which are the extended prompts, has the words “full body portrait”, which suggests that you should include the anime girl’s face, and her entire body in your prompt.

===========================================================================================
Now output 4 prompts, DO NOT give me any explanations, ONLY the 4 prompts separated by semicolon(;). You MUST make sure you pay SPECIAL ATTENTION to the mentioning of any camera angle/motion in user's prompt, if you see a camera angle/motion, you should treat the parts of the user's prompt relevant to this camera angle/motion as a single output prompt(separated by a semicolon(;)). It is VERY IMPORTANT that you make sure you have 3 and 3 only semicolons(;), so in total 4 prompts. You MUST make sure you pay SPECIAL ATTENTION to the mentioning of any camera angle/motion in user's prompt, if you see a camera angle/motion, you should treat the parts of the user's prompt relevant to this camera angle/motion as a single output prompt(separated by a semicolon(;)). It is VERY IMPORTANT that you make sure you have 3 and 3 only semicolons(;), so in total 4 prompts. Make sure also each of the 4 sub-prompts needs to be different in some way, don’t repeat for the 4 sub-prompts.  MAKE SURE you do these also, it is VERY IMPORTANT: If the user’s prompt doesn’t indicate an image that include a person’s face or some body parts, DO NOT include anything that describes a person, person’s face and these body parts, but only describe the parts of the person’s other body parts / clothings etc that the user indicates in his prompt; similarly, if it DOES indicate a person’s face or any of the person’s body parts, include them. ALSO MAKE SURE you make good attention in your output prompt of different views, e.g. side-view, front-view, close-up. It is VERY IMPORTANT to reflect these view in your prompt.It is VERY IMPORTANT that you make sure you don’t include my instruction prompt, which starts from the beginning of my input which is "Given a user’s prompt, " and ends at "My user’s prompt is: ".
some camera angle/motions could be the below, split with a semicolon(;) at these camera angle/motions:
 - camera angles: e.g. side-view, front-view, closeup, bird's eye view
 - facial expressions: e.g. smiling, crying, frowning
 - body movement: e.g. standing, doing a leg kick, jumping
 - body pose: e.g. sitting, standing, lying down
My user’s prompt is: 
"""
max_size = (300, 300)

premade_pose = ["dance", "flexing", "jumping", "laying", "sitting", "standing", "standing in T-shape"]
portrait_corpus = ["full body portrait", "upper body portrait", "lower body portrait"]
portrait_adjust_dict = {
    "full body portrait": 1.,
    "upper body portrait": 0.6,
    "lower body portrait": 0.6
}
pose_image_folder = "poses"
device = "cuda"
rare_tokens = extract_rare_tokens(rare_tokens_file)
character_names = ["None"] + extract_names_without_rare_tokens(database_file)
image_paths = get_image_paths()
# clip_model_name = "openai/clip-vit-base-patch32"
# # Load CLIP for text/image similarity
# clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
# clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# # Load DreamSim for perceptual similarity (install via: `pip install dreamsim`)
# from dreamsim import dreamsim
# dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)


def text_to_anime(prompt, negative_prompt, character_name, is_regenerate=False, num_images=4):
    global pipe_txt2img, character_name_value, rare_token
    
    current_list_size = len(os.listdir(txt2img_res_folder))
    
    if exist_name(character_name) and not is_regenerate:
        raise gr.Error("Name already in database. Please enter another unique name.")

    
    if character_name == "" and not is_regenerate:
        raise gr.Error("Please provide a character name.")

    
    if character_name == "None" and not is_regenerate:
        raise gr.Error("Character names cannot be 'None', please choose another character name.")
    
    character_name_value = character_name
    
    if is_regenerate and not selected_image:
        raise gr.Error("Please select a character first.")

    if not rare_tokens:
        print("No rare tokens found. Exiting.")
        return
    
    with open(database_file, "r") as file:
        num_lines = sum(1 for _ in file)

    # Choose a rare token (e.g., the first available token)
    rare_token = rare_tokens[-1 - num_lines + 1]  # You can add logic to choose a different token

    

    # load model
    # if (pipe_txt2img.pipe is None):
    #     pipe_txt2img.multi_thread_load_model()   
    
    # add prompt prefix to generate full body portrait for prototyping
    prompt = txt2img_prompt_prefix + prompt + prompt_suffix
    # print("txt2img prompt:", prompt)
    # infer 
    txt2img_prompt_query["6"]["inputs"]["text"] = prompt
    txt2img_prompt_query["7"]["inputs"]["text"] = negative_prompt
    txt2img_prompt_query["5"]["inputs"]["width"] = width_txt2img
    txt2img_prompt_query["5"]["inputs"]["height"] = height_txt2img
    txt2img_prompt_query["5"]["inputs"]["batch_size"] = 4
    txt2img_prompt_query["3"]["inputs"]["seed"] = random.randint(0, 1000000)
    
    
    queue_prompt(txt2img_prompt_query)
    while len(os.listdir(txt2img_res_folder)) < current_list_size + 4:
        pass
    time.sleep(2)
    res_item = os.listdir(txt2img_res_folder)[-4:]
    res = []
    for item in res_item:
        res_item_path = os.path.join(txt2img_res_folder, item)
        res.append(res_item_path)
    return res
    # res = pipe_txt2img.infer(prompt=prompt, negative_prompt=negative_prompt, height=height_txt2img, width=width_txt2img, 
    #                         num_images=num_images)
    # return res
      

def llm_extract(prompt_prefix, prompt):

    # load model
    if (pipe_chatbot.pipe is None):
        pipe_chatbot.multi_thread_load_model()   
    
    # # add prompt prefix to generate full body portrait for prototyping
    prompt = prompt_prefix + prompt
    # infer 
    res = pipe_chatbot.infer(prompt=prompt)
    print("llm res:", res)
    return res
      

def generate_pose_skeleton(pose, index):
    try:
        # load model
        # if (pipe_txt2img.pipe is None):
        #     pipe_txt2img.multi_thread_load_model()
        current_list_size = len(os.listdir(txt2img_res_folder))
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

        # add prompt prefix to generate full body portrait for prototyping
        # for pose in poses.split(","):
        prompt = pose_skeleton_prompt_prefix + pose

        # pipe_txt2img.unload_lora()
        # pipe_txt2img.multi_thread_load_lora(pipe_txt2img, [0., 0., 0.8, 0., 0., 0., 0., 0.])
        # infer 
        # res = pipe_txt2img.infer(prompt=prompt, negative_prompt=negative_prompt_suffix, height=height, width=width, 
        #                         num_images=1)
        txt2img_prompt_query["6"]["inputs"]["text"] = prompt
        txt2img_prompt_query["7"]["inputs"]["text"] = negative_prompt_suffix
        # txt2img_prompt_query["5"]["inputs"]["width"] = width1 if index % 2 == 0 else width2
        # txt2img_prompt_query["5"]["inputs"]["height"] = height1 if index % 2 == 0 else height2
        txt2img_prompt_query["5"]["inputs"]["width"] = width
        txt2img_prompt_query["5"]["inputs"]["height"] = height
        txt2img_prompt_query["5"]["inputs"]["batch_size"] = 1
        txt2img_prompt_query["3"]["inputs"]["seed"] = random.randint(0, 1000000)
        
        
        queue_prompt(txt2img_prompt_query)
        while len(os.listdir(txt2img_res_folder)) < current_list_size + 1:
            pass
        time.sleep(1)
        res_item = os.listdir(txt2img_res_folder)[-1]
        res = os.path.join(txt2img_res_folder, res_item)
        res = load_image(res)
        pose_skeleton = openpose(res)
        pose_skeleton_folder_path = "pose_skeletons"
        pose_skeleton_path = "pose_skeleton{}.png".format(index)
        pose_skeleton.save(pose_skeleton_path)
        return pose_skeleton_path
      
    except RuntimeError as e:
        if 'out of memory' in str(e):
            raise gr.Error("GPU out of memory. Please delete some models.") 

  
def pose_to_anime(prompt, negative_prompt, input_img_path, pose_skeleton, input_image, control_image, height, width, portrait_portion=[355, 280],):
    global pipe_pose
    
    current_list_size = len(os.listdir(res_folder))
    
    if input_img_path != "" and pose_skeleton != "":
        prompt = prompt.replace(txt2img_prompt_prefix, "")
        preprocess_pose_image(prompt, pose_skeleton, input_img_path, portrait_portion=portrait_portion)
    
    prompt, negative_prompt = prompt, negative_prompt
        
    # try: 
    # load model
    # if (pipe_pose.pipe is None):
    #     pipe_pose.multi_thread_load_model()  
    # print("input img:", input_image)
    # print("control img:", control_image)
    # print("prompt:", prompt)
    # infer
    # res = pipe_pose.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image, 
    #                     height=height, width=width, control_image=control_image)
    prompt_query["6"]["inputs"]["text"] = prompt
    prompt_query["7"]["inputs"]["text"] = negative_prompt
    prompt_query["13"]["inputs"]["image"] = input_image
    prompt_query["21"]["inputs"]["image"] = control_image
    prompt_query["3"]["inputs"]["seed"] = random.randint(0, 1000000)
    
    queue_prompt(prompt_query)
    while len(os.listdir(res_folder)) < current_list_size + 1:
        pass
    time.sleep(1)
    res_item = os.listdir(res_folder)[-1]
    res = os.path.join(res_folder, res_item)
    res = load_image(res)
    return res

    # except RuntimeError as e:
    #     if 'out of memory' in str(e):
    #         raise gr.Error("GPU out of memory. Please delete some models.") 


def extract_quotes(sentence):
    """
    Extract all content between double quotation marks in a sentence.

    Parameters:
        sentence (str): The input string.

    Returns:
        list: A list of strings found between double quotation marks.
    """
    return re.findall(r'"(.*?)"', sentence)  


def ip_generate(prompt, negative_prompt, height=height1, width=width1, input_img="komi2.png"):
    global pipe_ip
    prompt = txt2img_prompt_prefix + prompt + prompt_suffix
    # print("txt2img prompt:", prompt)
    # infer 
    res = pipe_ip.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_img, 
                        height=height, width=width, )
    return res

def postprocess_image(portrait_type, image_path, output_path, is_control_image=False, portrait_portion=[292, 219]):
    with Image.open(image_path) as img:
        width, height = img.size
        
        if portrait_type == "full body portrait":
            add_padding(image_path, output_path, portrait_portion)

        elif portrait_type == "upper body portrait":
            # add_padding(image_path, output_path, portrait_portion)
            
            new_height = height * 0.5
            new_width = new_height / portrait_portion[1] * portrait_portion[0]
            # if is_control_image:
            #     new_height = height * 0.6
            #     new_width = new_height / portrait_portion[1] * portrait_portion[0]
            # new_width = width * portrait_portion
            left = (width - new_width) // 2
            upper = 0
            right = left + new_width
            lower = new_height
                
            # Crop the image to the new height (keep the full width)
            cropped_img = img.crop((left, upper, right, lower))

            # Save the cropped image
            cropped_img.save(output_path)
        elif portrait_type == "lower body portrait":
            # add_padding(image_path, output_path, portrait_portion)
            
            new_height = height * 0.5
            new_width = new_height / portrait_portion[1] * portrait_portion[0]
            # if is_control_image:
            #     new_height = height * 0.6
            #     new_width = width * 0.58 if portrait_portion == [440, 457] else width * 0.82
                
            left = (width - new_width) // 2
            upper = height - new_height
            right = left + new_width
            lower = height

            # Crop the image to the new height (keep the full width)
            cropped_img = img.crop((left, upper, right, lower))

            # Save the cropped image
            cropped_img.save(output_path)
            

def generate_comfyui_filename(number):
  """
  Generates a filename string in the format ComfyUI_XXXXX_.png.

  Args:
    number: An integer representing the sequence number (starting from 1).

  Returns:
    A string formatted as ComfyUI_XXXXX_.png with the number padded
    to five digits with leading zeros.

  Raises:
    ValueError: If the input number is not a positive integer.
  """
  if not isinstance(number, int) or number < 1:
    raise ValueError("Input 'number' must be a positive integer starting from 1.")

  # Format the number to be 5 digits wide, padded with leading zeros
  padded_number = f"{number:05d}"

  # Construct the filename
  filename = f"ComfyUI_{padded_number}_.png"
  return filename


def create_manga(character1, character2, character3, character4, prompt, negative_prompt, ip=False, four_prompt=False, layout_regenerate_index=-1, 
                 modify_prompt=None, portrait_portion=[355, 280]):
    # characters can be of lennth 1 with a list of 2 characters each or length 4 with a list of 2 characters each
    # characters can be 1 or 2 characters' names
    print("image generate start")
    character_names = [character1, character2, character3, character4]
    # if character_name == "None":
    #     input_img_path = ""
    # else:
    #     input_img_path = get_image_path(character_name)
    #     character_prototype = get_character_prototype(character_name)
    # others = llm_extract(llm_others_prompt_prefix, prompt)
    
    #  adapted pose skeleton's image path
    output_path_pose = "preprocessed_pose.png"
    output_path_pose = os.path.join(working_folder, output_path_pose)
    
    # adapted input character prototype's image path
    output_path_portrait = "preprocessed_portrait.png"
    output_path_portrait = os.path.join(working_folder, output_path_portrait)
    # characters = [[character1, character2]]
    
    images = []
    four_prompt_file = "four_prompts.txt"
    if layout_regenerate_index != -1:
        if character_names[layout_regenerate_index] == "None":
            input_img_path = ""
            character_prototype = ""
            
        else:
            input_img_path = get_image_path(character_names[layout_regenerate_index])
            character_prototype = get_character_prototype(character_names[layout_regenerate_index])
        
        if layout_regenerate_index == 0 or layout_regenerate_index == 3:
            portrait_portion = [354, 457]
        
        height = height1 if portrait_portion == [354, 457] else height2
        width = width1 if portrait_portion == [354, 457] else width2
        # with open(four_prompt_file, 'r') as file:
        #     lines = file.readlines()
        # lines[layout_regenerate_index] = (modify_prompt + prompt_suffix) if modify_prompt != "" else lines[layout_regenerate_index]
        # # Step 3: Write the updated lines back to the file
        # with open(four_prompt_file, 'w') as file:
        #     file.writelines(lines)
        with open(four_prompt_file, "r") as file:
            lines = file.readlines()
            pose_skeleton = "pose_skeleton{}.png".format(layout_regenerate_index)
            # pose_skeleton = os.path.join(working_folder, pose_skeleton)
            prompt = txt2img_prompt_prefix + lines[layout_regenerate_index].strip()
            # pose_skeleton = generate_pose_skeleton(prompt)
            # print("portrait portion:", portrait_portion)
            # get the selected character's prototype image
            dialogue = ' '.join(extract_quotes(prompt))
            prompt = prompt.replace(dialogue, "")
            negative_prompt += negative_prompt_suffix
            if ip:
                reference_image = ip_generate(prompt, negative_prompt, height, width, input_img_path)
            else:
                pose_skeleton = generate_pose_skeleton(prompt, layout_regenerate_index)
                # portrait_portion = [354, 457] if (index == 0 or index == 3) else [355, 280]
                # height = height1 if portrait_portion == [354, 457] else height2
                # width = width1 if portrait_portion == [354, 457] else width2
                # print("portrait portion:", portrait_portion)
                # get the selected character's prototype image
                # prompt = character_prototype + pmpt + prompt_suffix
                # prompt = prompt[:846] if len(prompt) > 846 else prompt
                
                # file.write(f"{prompt}\n")
                # dialogue = ' '.join(extract_quotes(prompt))
                
                # prompt = prompt.replace(dialogue, "")
                # negative_prompt += negative_prompt_suffix
                reference_image = pose_to_anime(prompt, negative_prompt, 
                                                input_img_path, pose_skeleton,
                                                output_path_portrait, output_path_pose, 
                                                height, width, portrait_portion=portrait_portion
                                                )
            
            save_path = "reference_image{}.png".format(layout_regenerate_index)
            reference_image.save(save_path)
            if dialogue != "":
                reference_image = draw_text_bubble_on_image(save_path, dialogue, max_size, save_path)
            portrait_prompt = prompt.replace(txt2img_prompt_prefix, "")
            portrait_type = get_most_similar_sentence(prompt, portrait_corpus)
            
            res_folder_length = len(os.listdir(res_folder))
            output_path = generate_comfyui_filename(res_folder_length + 1)
            postprocess_image(portrait_type, save_path, output_path, portrait_portion=portrait_portion)
            # print("ref img generated")
            return reference_image
    
    # if four_prompt:
    #     height = height1 if portrait_portion == [354, 457] else height2
    #     width = width1 if portrait_portion == [354, 457] else width2
    #     # poses = poses.split(",")
    #     # poses = poses[:1] if len(poses) > 1 else poses
    #     # poses += [random.choice(premade_pose) for _ in range(1 - len(poses))]
    #     # pose = poses[0]
    #     # pose_skeleton = generate_pose_skeleton(prompt)
    #     # print("portrait portion:", portrait_portion)
    #     # get the selected character's prototype image
       
        
    #     prompt += prompt_suffix
    #     # with open(four_prompt_file, "w") as file:
    #     #     pass  # Do nothing, just open and close
    #     # with open(four_prompt_file, "a") as file:
    #     #     file.write(f"{prompt}\n")

    #     negative_prompt += negative_prompt_suffix
    #     pose_skeleton = "pose_skeleton{}.png".format(layout_regenerate_index)
    #     reference_image = pose_to_anime(prompt, negative_prompt, 
    #                                             input_img_path, pose_skeleton,
    #                                             output_path_portrait, output_path_pose, 
    #                                             height, width, portrait_portion=portrait_portion
    #                                             )
        
    #     dialogue = ' '.join(extract_quotes(prompt))
    #     save_path = "reference_image_six_prompt.png"
    #     reference_image.save(save_path)
    #     if dialogue != "":
    #         reference_image = draw_text_bubble_on_image(save_path, dialogue, max_size, save_path)
        
    #     # print("ref img generated")
    #     return reference_image
    if not ip:
        four_prompts = llm_extract(llm_pose_prompt_prefix, prompt)
    
    # poses = poses.split(",")
    # poses = poses[:6] if len(poses) > 6 else poses
    # poses = (poses * (6 // len(poses) + 1))[:6]
        four_prompts = four_prompts.split(";")[:4]
    else:
        with open(four_prompt_file, "r") as file:
            lines = file.readlines()
            four_prompts = [line.strip() for line in lines][:4]
    # poses += [random.choice(premade_pose) for _ in range(6 - len(poses))]
    # now poses have exactly 6 elements
    
    with open(four_prompt_file, "w") as file:
        pass  # Do nothing, just open and close
    with open(four_prompt_file, "w") as file:
        for index, pmpt in enumerate(four_prompts):
            if character_names[index] == "None":
                input_img_path = ""
                character_prototype = ""
            else:
                input_img_path = get_image_path(character_names[index])
                character_prototype = get_character_prototype(character_names[index])
                
            if ip:
                
                # print("pose:", pmpt, "index:", index)
                # pose_skeleton = generate_pose_skeleton(pmpt, index)
                portrait_portion = [354, 457] if (index == 0 or index == 3) else [355, 280]
                height = height1 if portrait_portion == [354, 457] else height2
                width = width1 if portrait_portion == [354, 457] else width2
                # print("portrait portion:", portrait_portion)
                # get the selected character's prototype image
                prompt = txt2img_prompt_prefix + character_prototype + pmpt + prompt_suffix
                # prompt = prompt[:846] if len(prompt) > 846 else prompt
                
                file.write(f"{prompt}\n")
                dialogue = ' '.join(extract_quotes(prompt))
                prompt = prompt.replace(dialogue, "")
                negative_prompt += negative_prompt_suffix
            
                reference_image = ip_generate(prompt, negative_prompt, height, width, input_img_path)
                # uncomment later
                save_path = "reference_image_ip{}.png".format(four_prompts.index(pmpt))
                # save_path = "reference_image{}.png".format(four_prompts.index(pmpt))
                
            else:
                # print("pose:", pmpt, "index:", index)
                pose_skeleton = generate_pose_skeleton(pmpt, index)
                portrait_portion = [354, 457] if (index == 0 or index == 3) else [355, 280]
                height = height1 if portrait_portion == [354, 457] else height2
                width = width1 if portrait_portion == [354, 457] else width2
                # print("portrait portion:", portrait_portion)
                # get the selected character's prototype image
                prompt = character_prototype + pmpt + prompt_suffix
                # prompt = prompt[:846] if len(prompt) > 846 else prompt
                
                file.write(f"{prompt}\n")
                dialogue = ' '.join(extract_quotes(prompt))
                
                prompt = prompt.replace(dialogue, "")
                negative_prompt += negative_prompt_suffix
            
                reference_image = pose_to_anime(prompt, negative_prompt, 
                                                input_img_path, pose_skeleton,
                                                output_path_portrait, output_path_pose, 
                                                height, width, portrait_portion=portrait_portion
                                                )
                save_path = "reference_image{}.png".format(four_prompts.index(pmpt))
            
            # dialogue = ' '.join(extract_quotes(pmpt))
            reference_image.save(save_path)
            if dialogue != "":
                reference_image = draw_text_bubble_on_image(save_path, dialogue, max_size, save_path)
            
            res_folder_length = len(os.listdir(res_folder))
            output_path = generate_comfyui_filename(res_folder_length + 1)
            portrait_prompt = prompt.replace(txt2img_prompt_prefix, "")
            portrait_prompt = portrait_prompt.replace(character_prototype, "")
            portrait_type = get_most_similar_sentence(prompt, portrait_corpus)
            
            postprocess_image(portrait_type, save_path, output_path, portrait_portion=portrait_portion)
            
            reference_image = load_image(output_path)
            images.append(reference_image)
            
    return images


# Benchmark function
def benchmark_model(prompt, num_inference_steps=25, is_ip=False):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    


    images = create_manga("komi", "komi", "komi", "komi", prompt, negative_prompt, ip=is_ip)

        
        
    inference_time = time.time() - start_time
    memory_usage = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
    print("is ip:", is_ip, "total inference time:", inference_time, "memory usage:", memory_usage)
    
    return images[0], num_inference_steps, memory_usage, inference_time

# # Calculate similarity scores
# def calculate_similarities(img1, img2, prompt):
#     img1 = load_image(img1)
#     img2 = load_image(img2)
#     # CLIP-T (Text-Image)
#     inputs_base = clip_processor(text=[prompt], images=img1, return_tensors="pt", padding=True).to(device)
#     inputs_ip = clip_processor(text=[prompt], images=img2, return_tensors="pt", padding=True).to(device)
    
#     with torch.no_grad():
#         clip_t_base = clip_model(**inputs_base).logits_per_image.item()
#         clip_t_ip = clip_model(**inputs_ip).logits_per_image.item()
    
#     # CLIP-I (Image-Image)
#     inputs = clip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         features = clip_model.get_image_features(**inputs)
#         clip_i = torch.cosine_similarity(features[0], features[1], dim=0).item()
    
#     # DreamSim
#     img1_tensor = dreamsim_preprocess(img1).unsqueeze(0).to(device)
#     img2_tensor = dreamsim_preprocess(img2).unsqueeze(0).to(device)
#     with torch.no_grad():
#         dreamsim_score = dreamsim_model(img1_tensor, img2_tensor).item()
        
#     return clip_t_base, clip_t_ip, clip_i, dreamsim_score

def create_manga_four_prompt(character1, character2, character3,character4, prompt1, negative_prompt1, 
                              prompt2, negative_prompt2, prompt3, negative_prompt3, prompt4, negative_prompt4,
                              ):
    # characters can be of lennth 1 with a list of 2 characters each or length 4 with a list of 2 characters each
    # characters can be 1 or 2 characters' names
    print("image generate start")
    reference_image1 = create_manga(character1, prompt1, negative_prompt1, four_prompt=True, portrait_portion=[354, 457])
    reference_image2 = create_manga(character2, prompt2, negative_prompt2, four_prompt=True, portrait_portion=[355, 280])
    reference_image3 = create_manga(character3, prompt3, negative_prompt3, four_prompt=True, portrait_portion=[355, 280])
    reference_image4 = create_manga(character4, prompt4, negative_prompt4, four_prompt=True, portrait_portion=[354, 457])
    # reference_image5 = create_manga(character5, prompt5, negative_prompt5, four_prompt=True, portrait_portion=[355, 220])
    # reference_image6 = create_manga(character6, prompt6, negative_prompt6, four_prompt=True, portrait_portion=[354, 457])
    # return reference_image1, reference_image2, reference_image3, reference_image4, reference_image5, reference_image6
    return reference_image1, reference_image2, reference_image3, reference_image4
    
# transport selected image in gallery to other tabs in app
def get_select_image(evt: gr.SelectData, character_prompt, regenerate_character_name, regenerate_bool=False):
    global character_name_value
    global selected_image_path
    regenerate_character_name = regenerate_character_name.strip()
    character_name_value = regenerate_character_name
    # selected_image_path = evt # testing purpose
    selected_image_path = evt.value["image"]["path"]
    character_prompt = character_prompt.strip()
    character_prompt += " , "
    modified = False
    # if regenerate_bool:
    #     regenerate_character_name = get_character_name(selected_image_path)
    #     modify_character(regenerate_character_name, selected_image_path, character_prompt)
    #     return gr.update(value=image_paths)
    with open(database_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            if regenerate_character_name == line.split(";")[1]:
                print("modifying character")
                modified = True
                modify_character(regenerate_character_name, selected_image_path, character_prompt)
                break
    if not modified:
        print("saving new character")
        save_character_to_database(character_name_value, rare_token, selected_image_path, character_prompt, database_file)

    gr.Info("Saved as character")
    image_paths = get_image_paths()
    print("regen gallery selected character:", regenerate_character_name)
    dropdown1, dropdown2, dropdown3, dropdown4 = populate_dropdowns()
    return gr.update(value=image_paths), dropdown1, dropdown2, dropdown3, dropdown4

def show_gallery(evt: gr.SelectData, ):
    image_paths = get_image_paths()
    return gr.update(value=image_paths)

def get_select_character(evt: gr.SelectData):
    global character_name_value, selected_image
    image_path = evt.value["image"]["path"]
    name = get_character_name(image_path)
    character_name_value = name
    selected_image = True
    print("char gallery selected character:", name)
    return name
    
def show_prompts():
    four_prompt_file = "four_prompts.txt"
    textboxes = []
    with open(four_prompt_file, "r") as file:
        lines = file.readlines()
        for line in lines[:4]:
            textboxes.append(gr.Textbox(label="Prompt", value=line.replace(prompt_suffix, ""), visible=True, interactive=True, ))
    return textboxes

def open_link():
    return "https://wj.qq.com/s2/19618426/9b30/"
    
def save_prompts(modify_prompt1, modify_prompt2, modify_prompt3, modify_prompt4):
    four_prompt_file = "four_prompts.txt"

    with open(four_prompt_file, 'r') as file:
        lines = file.readlines()
    lines[0] = (modify_prompt1.strip() + prompt_suffix + "\n")
    lines[1] = (modify_prompt2.strip() + prompt_suffix + "\n")
    lines[2] = (modify_prompt3.strip() + prompt_suffix + "\n")
    lines[3] = (modify_prompt4.strip() + prompt_suffix + "\n")
    
    # Step 3: Write the updated lines back to the file
    with open(four_prompt_file, 'w') as file:
        file.writelines(lines)
    return [gr.Textbox(visible=False)] * 4


with gr.Blocks(css=custom_css, elem_id="overlap-container", title="Shepherd Story") as demo:
    
    
    with gr.Tab("Create Prototype"):
        gr.Markdown("""
                    Please select a full body portrait including eveything from the head to feet of your character for best quality.
                    """)
        # get inputs for creating character prototype
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
            
            # height = gr.Slider(512, 1960, label="Height", step=8, value=1024)
        with gr.Row():
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
            # width = gr.Slider(512, 1960, label="Width", step=8, value=800)
        with gr.Row(equal_height=True):
            txt2img_gen_btn = gr.Button(value="Generate With Text")
            character_name = gr.Textbox(label="Character Name", placeholder="Enter a character name", lines=1)
        with gr.Row():
            # gallery to show generated images
            gallery = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            , columns=[4], rows=[1], object_fit="contain", height=480, allow_preview=True, format="png")
        # with gr.Row():
            # input_img = gr.Image(label="Upload your own image", type='filepath', height=480)
        

    with gr.Tab("Manage Characters") as character_database:
        gr.Markdown("""
                    Please select a full body portrait including eveything from the head to feet of your character for best quality.
                    """)
        is_regenerate_checkbox = gr.Checkbox(visible=False, value=True)
        regenerate_prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
        regenerate_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
        regenerate_character_name = gr.Textbox(visible=False)
        # gallery to show first character
        character_gallery = gr.Gallery(
            value=image_paths,
            label="Characters", show_label=True, elem_id="gallery", allow_preview=False
        , rows=1, format="png", columns=[2], object_fit="contain")
        
        # button to regenerated the 1st selected character
        regenerate_bool = gr.Checkbox(visible=False, value=True)
        character_regenerate_btn = gr.Button(value="Re-generate selected character")
        regenerate_gallery = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            , columns=[4], rows=[1], object_fit="contain", height=480, allow_preview=True, format="png")
    
    with gr.Tab("Create Manga"):
        not_four_prompt = gr.Checkbox(visible=False, value=False)
        four_prompt = gr.Checkbox(visible=False, value=True)
        layout_regenerate_index0 = gr.Number(visible=False, value=0)
        layout_regenerate_index1 = gr.Number(visible=False, value=1)
        layout_regenerate_index2 = gr.Number(visible=False, value=2)
        layout_regenerate_index3 = gr.Number(visible=False, value=3)
        # layout_regenerate_index4 = gr.Number(visible=False, value=4)
        # layout_regenerate_index5 = gr.Number(visible=False, value=5)
        # Give the row a unique elem_id, e.g. "no_gap_row"
        gr.Markdown("""
                    Please describe your manga story, and choose a character or None for each cell of the manga. The sequence of the manga is: upper left -> lower left -> upper right -> lower right.
                    If you're describing the content in each individual image, it's good to describe an image on a new line. Please use double quotation marks to indicate any dialogue.
                    """)
        with gr.Row():
            with gr.Tab("Single Prompt Mode"):
                manag_prompt_sub1 = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
                with gr.Row(equal_height=True):
                    manag_negative_prompt_sub1 = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
                with gr.Column():
                    character_dropdown_sub1 = gr.Dropdown(
                        character_names, value="None", label="Choose character 1", interactive=True
                    )
                    character_dropdown_sub2 = gr.Dropdown(
                        character_names, value="None", label="Choose character 2", interactive=True
                    )
                with gr.Column():
                    character_dropdown_sub3 = gr.Dropdown(
                        character_names, value="None", label="Choose character 3", interactive=True
                    )
                    character_dropdown_sub4 = gr.Dropdown(
                        character_names, value="None", label="Choose character 4", interactive=True
                    )
                # seed = gr.Number()
                # with gr.Group(visible=False):
                #     manag_prompt_sub2 = gr.Textbox(label="Prompt", placeholder="Enter a prompt for character 2", lines=3)
                #     with gr.Row(equal_height=True):
                #         manag_negative_prompt_sub2 = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image) for character 2", lines=3)
                #         character_dropdown_sub2 = gr.Dropdown(
                #             character_names, value="None", label="Choose character 2", interactive=True
                #         )  
                with gr.Row():  
                    with gr.Column():    
                        manga_button_single_prompt = gr.Button(value="Create Manga", scale=1)
                        ip_adapter_button = gr.Button(value="Baseline", scale=1)
                        
                    with gr.Column():
                        show_prompt_btn = gr.Button(value="Modify Prompt", scale=1)
                        save_prompt_btn = gr.Button(value="Save Prompt", scale=1)
                    ip_adapter_bool = gr.Checkbox(visible=False, value=True)
                    false_ip_adapter_bool = gr.Checkbox(visible=False, value=False)
                with gr.Row():
            
                    with gr.Column():
                        modify_prompt1 = gr.Textbox(label="Prompt", placeholder="Enter a prompt", visible=False, elem_id="top-text-big1" )
                        layout_image1 = gr.Image(
                            show_label=False, elem_id="sharp_corner", height=457, interactive=False, type='pil', format='png'
                        )
                        modify_prompt2 = gr.Textbox(label="Prompt", placeholder="Enter a prompt", visible=False, elem_id="top-text-small1")
                        
                        layout_image2 = gr.Image(
                            show_label=False, elem_id="sharp_corner", height=280, interactive=False, type='pil', format='png'
                        )
                        
                        
                    with gr.Column():
                        
                        modify_prompt3 = gr.Textbox(label="Prompt", placeholder="Enter a prompt", visible=False, elem_id="top-text-small2")
                        
                        layout_image3 = gr.Image(
                            show_label=False, elem_id="sharp_corner", height=280, interactive=False, type='pil', format='png'
                        )
                        modify_prompt4 = gr.Textbox(label="Prompt", placeholder="Enter a prompt", visible=False, elem_id="top-text-big2")
                    
                        layout_image4 = gr.Image(
                            show_label=False, elem_id="sharp_corner", height=457, interactive=False, type='pil', format='png'
                        )
                save_pdf_button = gr.Button(value="Save PDF")
                pdf_output = gr.File(visible=True)
                
                link_button = gr.Button("Survey Link", variant="primary", link="https://wj.qq.com/s2/19618426/9b30/")
                link_output = gr.Textbox(visible=False)

                
        character_database.select(show_gallery, [], [character_gallery])
        character_image_path_placeholder = gr.Textbox(visible=False)
        # gallery.select(fn=populate_dropdowns, inputs=[], outputs=[])
        gallery.select(get_select_image, [prompt, character_name], [character_gallery, character_dropdown_sub1, character_dropdown_sub2,character_dropdown_sub3, character_dropdown_sub4 ]) 

        character_gallery.select(get_select_character, [], [regenerate_character_name]) 
        character_regenerate_btn.click(fn=text_to_anime, inputs=[regenerate_prompt, regenerate_negative_prompt, regenerate_character_name, is_regenerate_checkbox], outputs=[regenerate_gallery])
        regenerate_gallery.select(get_select_image, [regenerate_prompt, regenerate_character_name, regenerate_bool], [character_gallery, character_dropdown_sub1, character_dropdown_sub2,character_dropdown_sub3, character_dropdown_sub4 ]) 
        
        txt2img_gen_btn.click(fn=text_to_anime, inputs=[prompt, negative_prompt, character_name], outputs=[gallery])

        # character_regenerate_btn.click(fn=text_to_anime, inputs=[prompt, negative_prompt, character_name], outputs=[gallery])
        
        # character_dropdowns_single_prompt = [[character_dropdown_sub1, character_dropdown_sub2]]
        # character_dropdowns_four_prompt = [[character_dropdown1, character_dropdown1_sub1], [character_dropdown2, character_dropdown2_sub1], [character_dropdown3, character_dropdown3_sub1], [character_dropdown4, character_dropdown4_sub1]]
        manga_button_single_prompt.click(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1,], outputs=[layout_image1, layout_image2, layout_image3, layout_image4, ])
        ip_adapter_button.click(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1, ip_adapter_bool], outputs=[layout_image1, layout_image2, layout_image3, layout_image4, ])
        show_prompt_btn.click(fn=show_prompts, inputs=[], outputs=[modify_prompt1, modify_prompt2, modify_prompt3, modify_prompt4])
        save_prompt_btn.click(fn=save_prompts, inputs=[modify_prompt1, modify_prompt2, modify_prompt3, modify_prompt4], outputs=[modify_prompt1, modify_prompt2, modify_prompt3, modify_prompt4])
        layout_image1.select(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1, false_ip_adapter_bool, not_four_prompt,layout_regenerate_index0, modify_prompt1], outputs=[layout_image1])
        layout_image2.select(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1, false_ip_adapter_bool, not_four_prompt,layout_regenerate_index1, modify_prompt2], outputs=[layout_image2])
        layout_image3.select(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1, false_ip_adapter_bool, not_four_prompt,layout_regenerate_index2, modify_prompt3], outputs=[layout_image3])
        layout_image4.select(fn=create_manga, inputs=[character_dropdown_sub1, character_dropdown_sub2, character_dropdown_sub3, character_dropdown_sub4, manag_prompt_sub1, manag_negative_prompt_sub1, false_ip_adapter_bool, not_four_prompt,layout_regenerate_index3, modify_prompt4], outputs=[layout_image4])
        save_pdf_button.click(save_pdf, [], [pdf_output])
        link_button.click(fn=open_link, outputs=link_output)

demo.launch(share=True, favicon_path="favicon.png")
# prompt = """

# a cute high school girl with very long purple hair leaps sideways, front-view, full-body portrait, dodging a punch, purple blazer flapping, white shirt wrinkled, dynamic motion blur, alleyway backdrop, neon lights flickering, closeup of her fist colliding with a thug’s jaw, lower angle, upper body portrait, sweat spraying, purple hair whipping across her determined face, side-view of her spinning kick, purple skirt swirling, calf socks glinting, opponent crumpling mid-air, debris scattering, rear-view, full-body portrait, sprinting toward a chain-link fence, graffiti-covered walls blurred in motion
# """
# negative_prompt = ""
# evaluate_prompt_file = "evaluate_prompt.txt"
# clip_t_base_all, clip_t_ip_all, clip_i_base_all, clip_i_ip_all, dreamsim_score_base_all, dreamsim_score_ip_all = [], [], [], [], [], []

# with open(evaluate_prompt_file, "r") as file:
#     lines = file.readlines()
#     for prompt in lines:
#         img_base, steps_base, mem_base, time_base = benchmark_model(prompt)

#         img_ip, steps_ip, mem_ip, time_ip = benchmark_model(prompt, is_ip=True)

#         for i in range(4):
#             my_img_path1 = "/root/reference_image{}.png".format(i)
#             ip_img_path1 = "/root/reference_image_ip{}.png".format(i)
            
#             # get the line from four_prompts.txt
#             with open("/root/four_prompts.txt", "r") as f:
#                 lines = f.readlines()
#                 prompt = lines[i].strip()
#             clip_t_base, clip_t_ip, clip_i_base, clip_i_ip, dreamsim_score_base, dreamsim_score_ip = calculate_similarities(my_img_path1, ip_img_path1, prompt)
#             clip_t_base_all.append(clip_t_base)
#             clip_t_ip_all.append(clip_t_ip)
#             clip_i_base_all.append(clip_i_base)
#             clip_i_ip_all.append(clip_i_ip)
#             dreamsim_score_base_all.append(dreamsim_score_base)
#             dreamsim_score_ip_all.append(dreamsim_score_ip)
#             print("-----MINE vs. IPAdapter metrics------")
#             print(f"{'Metric':<20} | Image_{i} | Image_{i}")
#             print(f"{'-'*60}")
#             print(f"{'CLIP-T Score':<20} | {clip_t_base:<20.4f} | {clip_t_ip:<20.4f}")
#             print(f"{'CLIP-I Score':<20} | {clip_i_base:<20.4f} | {clip_i_ip:<20.4f}")
#             print(f"{'DreamSim Score':<20} | {dreamsim_score_base:<20.4f} | {dreamsim_score_ip:<20.4f}")
# print("--------MEAN SCORES---------")

# clip_t_base_all = np.array(clip_t_base_all)
# clip_t_ip_all = np.array(clip_t_ip_all)
# clip_i_base_all = np.array(clip_i_base_all)
# clip_i_ip_all = np.array(clip_i_ip_all)
# dreamsim_score_base_all = np.array(dreamsim_score_base_all)
# dreamsim_score_ip_all = np.array(dreamsim_score_ip_all)

# print(f"{'CLIP-T Score':<20} | {np.mean(clip_t_base_all):<20.8f} | {np.mean(clip_t_ip_all):<20.8f}")
# print(f"{'CLIP-I Score':<20} | {np.mean(clip_i_base_all):<20.8f} | {np.mean(clip_i_ip_all):<20.8f}")
# print(f"{'DreamSim Score':<20} | {np.mean(dreamsim_score_base_all):<20.8f} | {np.mean(dreamsim_score_ip_all):<20.8f}")
# Print results
   
# test_prompt = """
# a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, afternoon at a school background, warm afternoon sunlight, a girl (marking her exam papers)1.5 upper body portrait, then said "Phew..." with eyes closed looking tired upper body portrait, then (packed the practice exam papers into her school bag)1.5, then walked out of the classroom door with (back shot)1.6 full body portrait, 
# """
# prompt = """
# animestyle, a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, 
# """
# text_to_anime(prompt, "", "q", )


# create_manga("komi", test_prompt, "", four_prompt=False, portrait_portion=[354, 457])
