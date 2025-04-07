# ShepherdStory

# 部署 (Install)
1. ```pip install comfy-cli```
2. ```comfy install```
3. download this repo as zip, extract all files in this repo into your /root folder or a folder of your choice. *For 深睿: 请把my_utils.py文件中的 ```working_folder```改为你的 working directory*
4. download the zip for these custom nodes and put into the ```custom_nodes``` folder: https://github.com/SeargeDP/SeargeSDXL, https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes, https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
5. use your CivitAI token to download these base model weights and put into the ```models/checkpoints``` folder: https://civitai.com/models/370529/atomix-anime-xl

```
wget "https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=YOUR_CIVITAI_TOKEN" -O comfy/ComfyUI/models/checkpoints/AtomicXL.safetensors
```
6. download these LoRA weights and put into the ```models/loras``` folder: [[https://civitai.com/models/370529/atomix-anime-xl]](https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16)(https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true)

```
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true -O comfy/ComfyUI/models/loras/sdxl_lightning_4step_lora.safetensors
```

8. download these ControlNet weights and put into the ```models/controlnet``` folder: [[[https://civitai.com/models/370529/atomix-anime-xl]](https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16)(https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true)](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true)

```
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true -O comfy/ComfyUI/models/controlnet/controlnet_sdxl_union_promax.safetensors
```

9. put comfyui_colab.ipynb *in this repo* into ```notebook``` folder,  run this notebook. Note in future running, only need to re-run cell with title "Run ComfyUI with cloudflared (Recommended Way)"
10. in terminal, run the following commands:
```
apt install nvidia-cuda-toolkit -y
pip install -r requirements.txt -U
apt-get install build-essential -y
pip install accelerate==1.2.1 -U
export HF_TOKEN=REPLACE_WITH_YOUR_HUGGING_FACE_TOKEN
python ui.py 
```
7. open the printed link in terminal and start using the app.
