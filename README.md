# ShepherdStory

# 部署 (Install)
1. ```pip install comfy-cli```
2. ```comfy install```
3. download the zip for these custom nodes and put into the ```custom_nodes``` folder: https://github.com/SeargeDP/SeargeSDXL, https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes, https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
4. download these base model weights and put into the ```models/checkpoints``` folder: https://civitai.com/models/370529/atomix-anime-xl

```
wget "https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=5f4a57bd4db9e5da2f7e86f0fb6f9237" -O comfy/ComfyUI/models/checkpoints/AtomicXL.safetensors
```
6. download these LoRA weights and put into the ```models/loras``` folder: [[https://civitai.com/models/370529/atomix-anime-xl]](https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16)(https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true)

```
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true -O comfy/ComfyUI/models/loras/sdxl_lightning_4step_lora.safetensors
```

8. download these ControlNet weights and put into the ```models/controlnets``` folder: [[[https://civitai.com/models/370529/atomix-anime-xl]](https://civitai.com/api/download/models/413937?type=Model&format=SafeTensor&size=pruned&fp=fp16)(https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true)](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true)

```
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true -O comfy/ComfyUI/models/controlnets/controlnet_sdxl_union_promax.safetensors
```

9. go to notebooks/comfyui_colab.ipynb, run this notebook. Note in future running, only need to re-run cell with title "Run ComfyUI with cloudflared (Recommended Way)"
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
