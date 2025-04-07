# ShepherdStory

# 部署 (Install)
1. ```pip install comfy-cli```
2. ```comfy install```
3. download the zip for these custom nodes and put into the ```custom_nodes``` folder: https://github.com/SeargeDP/SeargeSDXL, https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes, https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
5. go to notebooks/comfyui_colab.ipynb, run this notebook. Note in future running, only need to re-run cell with title "Run ComfyUI with cloudflared (Recommended Way)"
6. in terminal, run the following commands:
```
apt install nvidia-cuda-toolkit -y
pip install -r requirements.txt -U
apt-get install build-essential -y
pip install accelerate==1.2.1 -U
export HF_TOKEN=REPLACE_WITH_YOUR_HUGGING_FACE_TOKEN
python ui.py 
```
7. open the printed link in terminal and start using the app.
