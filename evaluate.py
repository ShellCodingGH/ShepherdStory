import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from transformers import CLIPProcessor, CLIPModel
import time

from diffusers.utils import load_image

device = "cuda"
# Load CLIP for text/image similarity
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load DreamSim for perceptual similarity (install via: `pip install dreamsim`)
from dreamsim import dreamsim
dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)
steps_base = 4
steps_ip = 30
# prompt = """
# a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, afternoon at a school background, warm afternoon sunlight, a girl (marking her exam papers)1.5 upper body portrait, then said "Phew..." with eyes closed looking tired upper body portrait, then (packed the practice exam papers into her school bag)1.5, then walked out of the classroom door with (back shot)1.6 full body portrait, 
# """

# prompt = """
# an elegant high school girl, very long purple hair, purple blazer, white inner shirt, long purple school uniform trousers,a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt, afternoon at a school background, warm afternoon sunlight, a girl (marking her exam papers)1.5 upper body portrait, perfection style, perfection, perfect, midjourneyv6.1, max details, Photo, product photo, (expressive)1.7, masterpiece, raw quality, best quality, HD, extremely detailed, high definition, stunning beautiful, soft features, masterpiece, raw quality, best quality, extremely detailed, stunning beautiful, high definition, HD,(masterpiece)1.2, (best quality)1.2, (ultra-detailed)1.2, (unity 8k wallpaper)1.2, (illustration)1.1, (anime style)1.1, intricate, fluid simulation, sharp edges, (glossy)1.2, (Smooth)1.2, (detailed eyes)1.2
# """


negative_prompt = ""
prototype_image = "/root/komi1.png"

def calculate_similarities(img1, img2, prompt, prototype_image=prototype_image, ):
    img1 = load_image(img1)
    img2 = load_image(img2)
    prototype_prompt = "a cute high school girl, very long purple hair, purple blazer, white inner shirt, purple calf socks, purple school uniform skirt,"
    prompt = prompt.replace(prototype_prompt, "")
    print("evalute prompt:", prompt)
    prototype_image = load_image(prototype_image)
    
    # CLIP-T
    text_inputs = clip_processor(
        text=[prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    def get_clip_t_score(image):
        image_inputs = clip_processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        # Cosine similarity (range [-1, 1]) then normalize to [0, 1]
        similarity = torch.matmul(text_features, image_features.T).item()
        return similarity  # Convert to 0-1 range

    clip_t_base = get_clip_t_score(img1)
    clip_t_ip = get_clip_t_score(img2)

    # CLIP-I (Image-Image)
    inputs1 = clip_processor(images=[prototype_image, img1], return_tensors="pt", padding=True).to(device)
    inputs2 = clip_processor(images=[prototype_image, img2], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features1 = clip_model.get_image_features(**inputs1)
        features2 = clip_model.get_image_features(**inputs2)
        clip_i_base = torch.cosine_similarity(features1[0], features1[1], dim=0).item()
        clip_i_ip = torch.cosine_similarity(features2[0], features2[1], dim=0).item()
    
    # DreamSim
    prototype_tensor = dreamsim_preprocess(prototype_image).to(device)
    img1_tensor = dreamsim_preprocess(img1).to(device)
    img2_tensor = dreamsim_preprocess(img2).to(device)
    with torch.no_grad():
        dreamsim_score_base = dreamsim_model(prototype_tensor, img1_tensor).item()
        dreamsim_score_ip = dreamsim_model(prototype_tensor, img2_tensor).item()
    
    return clip_t_base, clip_t_ip, clip_i_base, clip_i_ip, dreamsim_score_base, dreamsim_score_ip


# for i in range(4):
#     my_img_path1 = "/workspace/reference_image{}.png".format(i)
#     ip_img_path1 = "/workspace/reference_image_ip{}.png".format(i)
    
#     # get the line from four_prompts.txt
#     with open("/workspace/four_prompts.txt", "r") as f:
#         lines = f.readlines()
#         prompt = lines[i].strip()
#     clip_t_base, clip_t_ip, clip_i_base, clip_i_ip, dreamsim_score_base, dreamsim_score_ip = calculate_similarities(my_img_path1, ip_img_path1, prompt)
#     print("-----MINE vs. IPAdapter metrics------")
#     print(f"{'Metric':<20} | Image_{i} | Image_{i}")
#     print(f"{'-'*60}")
#     print(f"{'CLIP-T Score':<20} | {clip_t_base:<20.4f} | {clip_t_ip:<20.4f}")
#     print(f"{'CLIP-I Score':<20} | {clip_i_base:<20.4f} | {clip_i_ip:<20.4f}")
#     print(f"{'DreamSim Score':<20} | {dreamsim_score_base:<20.4f} | {dreamsim_score_ip:<20.4f}")
