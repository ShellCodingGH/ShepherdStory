

#!/usr/bin/env python
# coding: utf-8

# file: model_base.py
# -*- coding: utf-8 -*-

# importing the modules
import os
import concurrent.futures
   
import numpy as np
import torch
from threading import Thread
from safetensors.torch import load_file
import cv2
import gradio as gr
from PIL import Image
import huggingface_hub
import onnxruntime as rt
from controlnet_aux import OpenposeDetector
import transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from stable_diffusion_xl_controlnet_reference import StableDiffusionXLControlNetReferencePipeline
# from stable_diffusion_reference import StableDiffusionReferencePipeline
# from stable_diffusion_xl_reference import StableDiffusionXLReferencePipeline
# from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers import DiffusionPipeline
from diffusers import ControlNetModel
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import UniPCMultistepScheduler
from diffusers import DDIMScheduler
from diffusers import LCMScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image
from diffusers.utils import export_to_gif
from diffusers.utils import export_to_video
from diffusers.models.attention_processor import AttnProcessor2_0
from compel import Compel
from compel import ReturnedEmbeddingsType
import deepspeed
from huggingface_hub import hf_hub_download

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
# set gpu device
device = "cuda" if torch.cuda.is_available() else "cpu"
# options for input to print_msg function to decide what to print on screen
LOAD_MODEL_TEXT = "load model"
FINISH_MODEL_LOADING_TEXT = "finish model loading"

# some hidden text to enchance quality and reduce defects
hidden_booster_text = ", masterpiece-anatomy-perfect, dynamic, dynamic colors, bright colors, high contrast, excellent work, extremely elaborate picture description, 8k, obvious light and shadow effects, ray tracing, obvious layers, depth of field, best quality, RAW photo, best quality, highly detailed, intricate details, HD, 4k, 8k, high quality, beautiful eyes, sparkling eyes, beautiful face, masterpiece,best quality,ultimate details,highres,8k,wallpaper,extremely clear,"
hidden_negative = ", internal-organs-outside-the-body, internal-organs-visible, anatomy-description, unprompted-nsfw ,worst-human-external-anatomy, worst-human-hands-anatomy, worst-human-fingers-anatomy, worst-detailed-eyes, worst-detailed-fingers, worst-human-feet-anatomy, worst-human-toes-anatomy, worst-detailed-feet, worst-detailed-toes, camera, smartphone, worst-facial-details, ugly-detailed-fingers, ugly-detailed-toes,fingers-in-worst-possible-shape, worst-detailed-eyes, undetailed-eyes, undetailed-fingers, undetailed-toes, "

MAX_INPUT_TOKEN_LENGTH = 4096
MAX_NEW_TOKENS = 1024
# base = "checkpoints/envy_2D.safetensors"
base = "checkpoints/AtomicXL.safetensors"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!
adapter_names=["sdxl_lightning", "jeweled_eyes", "epic_fantasy", "aidma_upgrader", "sdxl_enhance", "aidma_mj", "handXL"]
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16)


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return
    

class Model():
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.pipe = None
        
    def load_chatbot(self,):
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,  # replace with "mps" to run on a Mac device
        )


    def load_txt2img(self,):
        # Load model.
        self.pipe = StableDiffusionXLPipeline.from_single_file(base, 
                                                          torch_dtype=torch.bfloat16, 
                                                          vae=vae,
                                                          variant="fp16").to(device)
    
    def load_pose(self):
        self.controlnet_conditioning_scale = 0.35
        # load controlnet
        controlnet = ControlNetModel.from_single_file(
            "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/blob/main/diffusion_pytorch_model_promax.safetensors",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.pipe = StableDiffusionXLControlNetReferencePipeline.from_single_file(base, 
                                                                            torch_dtype=torch.bfloat16, 
                                                                            controlnet=controlnet, 
                                                                            vae=vae,  
                                                                            variant="fp16",

                                                                            )
    
    def load_rmbg(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
        self.pipe = rt.InferenceSession(model_path, providers=providers)
        
    def load_lora(self, adapter_weights, chibi=False):
        self.pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        self.pipe.fuse_lora()
        print("loaded lightning_xl")
        # if isinstance(self.pipe, StableDiffusionXLControlNetReferencePipeline):
        #     self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm_xl",low_cpu_mem_usage=True)
        #     self.pipe.fuse_lora()
        #     print("loaded lcm_xl")
        return
    
    def unload_lora(self, ):
        print("unloaded lora")
        
        
    def multi_thread_unload_lora(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.unload_lora,)
            future.result()

    
    def multi_thread_load_chatbot(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_chatbot,)
            future.result()

    
    def multi_thread_load_txt2img(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_txt2img,)
            future.result()

    
    def multi_thread_load_pose(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_pose,)
            future.result()

    
    def multi_thread_load_rmbg(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_rmbg,)
            future.result()

    
    def multi_thread_load_lora(self, adapter_weights=[1., 0., 0.3, 0.5, 0.5, 0.5, 1.], chibi=False):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_lora, adapter_weights, chibi)
            future.result()

    
    def load_ti(self, ):
        # load textual inversions
        state_dict1 = load_file("textual_inversions/unaestheticXL_Alb2.safetensors")
        self.pipe.load_textual_inversion(state_dict1["clip_g"], token="unaestheticXL_Alb2", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        self.pipe.load_textual_inversion(state_dict1["clip_l"], token="unaestheticXL_Alb2", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)

        state_dict2 = load_file("textual_inversions/negativeXL_D.safetensors")
        self.pipe.load_textual_inversion(state_dict2["clip_g"], token="negativeXL_D", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        self.pipe.load_textual_inversion(state_dict2["clip_l"], token="negativeXL_D", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)

        state_dict3 = load_file("textual_inversions/Animetoon_Negatives.safetensors")
        self.pipe.load_textual_inversion(state_dict3["clip_g"], token="Animetoon_Negatives", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        self.pipe.load_textual_inversion(state_dict3["clip_l"], token="Animetoon_Negatives", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
        print("loaded Animetoon_Negatives")

        
        state_dict5 = load_file("textual_inversions/DeepNegative_xl_v1.safetensors")
        self.pipe.load_textual_inversion(state_dict5["clip_g"], token="DeepNegative_xl_v1", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        self.pipe.load_textual_inversion(state_dict5["clip_l"], token="DeepNegative_xl_v1", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
        print("loaded DeepNegative_xl_v1")

        print("TI loaded!")
    
    def multi_thread_load_ti(self, ):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_ti, )
            future.result()

    
    def load_model(self):
        gr.Info(LOAD_MODEL_TEXT)
        
        match self.model_name:
            case "chatbot":
                self.multi_thread_load_chatbot()

            case "txt2img":
                # load pipe
                self.multi_thread_load_txt2img()
                # load LoRAs
                self.multi_thread_load_lora()
                # # load textual inversions
                self.multi_thread_load_ti()

            case "img2pose":
                self.multi_thread_load_pose()
                # load LoRAs
                self.multi_thread_load_lora(adapter_weights=[1., 0.2, 0.3, 0.5, 0.5, 0.5, 1.])
                # load textual inversions
                self.multi_thread_load_ti()

            case "rmbg":
                # load pipe
                self.multi_thread_load_rmbg()
                
            
        if self.model_name != "chatbot" and self.model_name != "rmbg":
            
            # self.pipe.unet.to(memory_format=torch.channels_last)
            # self.pipe.vae.to(memory_format=torch.channels_last)
            # self.pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
            # self.pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
            self.pipe.fuse_qkv_projections()
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True, timestep_spacing="trailing")
            # self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_attention_slicing()
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.to(device)
            # init deepspeed inference engine
            # deepspeed.init_inference(
            #     model=getattr(pipe,"model", pipe),      # Transformers models
            #     mp_size=1,        # Number of GPU
            #     dtype=torch.bfloat16, # dtype of the weights (fp16)
            #     replace_method="auto", # Lets DS autmatically identify the layer to replace
            #     replace_with_kernel_inject=False, # replace the model with the kernel injector
            # )
            # print("DeepSpeed Inference Engine initialized")
        
        # gr.Info(FINISH_MODEL_LOADING_TEXT)
  
        
    def multi_thread_load_model(self):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_model)
            future.result()

    
    def check_input_img(self, image):
        if image is None:
            raise gr.Error("Please provide a input image.")
        
    def check_height_width(self, height, width):
        if height % 8:
            raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
            
        if width % 8:
            raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
        
    def compel_prompts(self, prompt, negative_prompt):
        # prompt weighter to add weights to prompts
        
        compel_proc = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2] ,
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
            )
  
        # positive prompt with weights
        prompt = prompt + hidden_booster_text 

        prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)

        # negative prompt with weights
        negative_prompt = negative_prompt + hidden_negative
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel_proc(negative_prompt)
        
        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds

    def gen_batch_img(self, height, width, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, num_inference_steps=4, input_img=None, control_image=None, num_images=4, guidance_scale=0.):
        res = []
        for x in range(num_images):
            i = 0
            res_image = Image.fromarray(np.zeros((64, 64)))
            while not res_image.getbbox() and i < 30:
                if isinstance(self.pipe, StableDiffusionXLControlNetReferencePipeline):
                   
                    res_image = self.pipe(
                        height=height, width=width, 
                        prompt_embeds=prompt_embeds, 
                        pooled_prompt_embeds=pooled_prompt_embeds, 
                        negative_prompt_embeds=negative_prompt_embeds, 
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                        image=control_image,
                        ref_image=input_img,
                        reference_attn=True,
                        reference_adain=False,
                        style_fidelity=1.,
                        guidance_scale=guidance_scale,

                        clip_skip=2,
                    ).images[0]
                else:
                    res_image = self.pipe(height=height, width=width, 
                                    prompt_embeds=prompt_embeds, 
                                    pooled_prompt_embeds=pooled_prompt_embeds, 
                                    negative_prompt_embeds=negative_prompt_embeds, 
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, 
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps,
                                    clip_skip=2,
                                    
                                    ).images[0]
                    res_image.save("prototype{}.png".format(x))
                i += 1
            res.append(res_image)
        return res[0] if num_images == 1 else res
    
    def delete_model(self):
        # delete models to release memory
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        else:
            raise gr.Error("No model to delete.")
            
        # print message when finished deleting models
        gr.Info("Model deleted.")
     
     
class ChatbotModel(Model):
    def __init__(self):
        super().__init__("chatbot")
        
    def infer(self, **kwargs):
        prompt = kwargs["prompt"]
        

        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipe(messages, max_new_tokens=256)
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
        # print(assistant_response)
        return assistant_response


class Txt2imgModel(Model):
    def __init__(self):
        super().__init__("txt2img")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, height, width, num_images = kwargs["prompt"], kwargs["negative_prompt"], kwargs["height"], kwargs["width"], kwargs["num_images"]
        
        # check valid height and width
        self.check_height_width(height, width)
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.compel_prompts(prompt, negative_prompt)
        # generate result image(s)
        res = self.gen_batch_img( height, width, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, num_images=num_images)

        return res
    
    
class Img2poseModel(Model):
    def __init__(self):
        super().__init__("img2pose")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img, height, width, control_image = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"], kwargs["height"], kwargs["width"], kwargs["control_image"]
        self.check_input_img(input_img)
        
        # load image
        image = load_image(input_img)
        control_image = load_image(control_image)
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.compel_prompts(prompt, negative_prompt)

        # generating 4 result images
        res1 = self.gen_batch_img(height, width, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, 4, image, control_image=control_image, num_images=1,)

        return res1


class RMBGModel(Model):
    def __init__(self):
        super().__init__("rmbg")
    
    # get a mask for background removal
    def get_mask(self, img, s=1024):
        img = (img / 255).astype(np.float32)
        h, w = h0, w0 = img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        mask = self.pipe.run(None, {'img': img_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
        return mask
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        
        input_img = kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        input_img = input_img.convert("RGB")
        # remove background
        img = np.array(input_img)
        mask = self.get_mask(img)
        img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
        mask = mask.repeat(3, axis=2)
        img = Image.fromarray(img)
        return img
    
