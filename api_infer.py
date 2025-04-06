import json
from urllib import request, parse
import random
import pandas as pd


#This is the ComfyUI api prompt format.

#If you want it for a specific workflow you can "enable dev mode options"
#in the settings of the UI (gear beside the "Queue Size: ") this will enable
#a button on the UI to save workflows in api format.

#keep in mind ComfyUI is pre alpha software so this format will change a bit.

#this is the one for the default workflow

prompt_file = "four_prompts.txt"
with open("url.txt", 'r') as file:
  # lines = file.readlines()
  # url = lines[0].strip()
  
  raw_url = file.readline()
  if not raw_url:
      raise ValueError("url.txt is empty")

  # Clean the URL
  base_url = raw_url.strip() 
  full_url = f"{base_url}/prompt" 
  print(f"Read and stripped URL: '{base_url}'") # For debugging
# search_text_file = "/root/svd_folder/ComfyUI/input/search_texts.txt"
# prompt = json.loads(prompt_text)
# excel_file = '/root/svd_folder/ComfyUI/test.xlsx'  # Replace with your Excel file path
# position_texts = [
#     "画面下方写着","画面左侧写着","画面右侧写着",
#     "画面中央写着","画面左下角写着",
#     "画面右下角写着"
#     ]

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(full_url, data=data, )
    request.urlopen(req)

# def extract_query_texts(excel_file):
#     # Read the Excel file
#     df = pd.read_excel(excel_file, engine='openpyxl')

#     # Extract the 'B' column
#     column_b = df['query']

#     # Convert to a list and print each cell's text
#     texts = column_b.tolist()

    

#     return texts
  


# def generate_random_position_text():
#     return random.choice(position_texts)


# texts = extract_query_texts(excel_file)


# for text in texts:
#     # position_text = generate_random_position_text()
#     pmpt = position_text + '"' + text + '"'
#     print("prompt:", pmpt)
#     search_text = text
#     prompt["53"]["inputs"]["string"] = pmpt.strip()
#     prompt["115"]["inputs"]["search_text"] = search_text.strip()
#     queue_prompt(prompt)


#set the text prompt for our positive CLIPTextEncode
# with open(prompt_file, 'r') as prompt_f, open(search_text_file, 'r') as search_f:
#     prompts = prompt_f.readlines()
#     search_texts = search_f.readlines()

# for index in range(len(prompts)):
#     print("pmpt:", prompts[index], "\t\tsrch txt: ", search_texts[index])
#     prompt["53"]["inputs"]["string"] = prompts[index].strip()
#     prompt["115"]["inputs"]["search_text"] = search_texts[index].strip()
#     queue_prompt(prompt)



#set the seed for our KSampler node
# prompt["3"]["inputs"]["seed"] = 5




