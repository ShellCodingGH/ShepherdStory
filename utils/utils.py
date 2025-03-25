import re
import os
import hashlib
import uuid
import random
import textwrap
from threading import Thread

from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
import gradio as gr
from diffusers.utils import load_image
from sentence_transformers.cross_encoder import CrossEncoder
import cv2

from model_base import RMBGModel
from anime_object_detection.detection import FaceDetection

device = "cuda"
img_path = "/workspace/reference_image0.png"
# image = load_image(img_path)
model_name = 'face_detect_v1.4_s'
database_file = "database.txt"
portrait_corpus = ["full body portrait", "upper body portrait", "lower body portrait"]
similarity_model = CrossEncoder("cross-encoder/stsb-distilroberta-base", device=device)
rmbg_model = RMBGModel()
prompt_suffix = ", masterpiece++, best quality++, ultra-detailed+ +, unity 8k wallpaper+, illustration+, anime style+, intricate, fluid simulation, sharp edges. glossy++, Smooth++, detailed eyes++"
    

def generate_string_hash():
    # Generate a random UUID
    random_uuid = uuid.uuid4()
    
    # Create a SHA-256 hash of the random UUID
    hash_object = hashlib.sha256(random_uuid.bytes)
    
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()


def get_image_paths(file_path="database.txt"):
    image_paths = []
    try:
        with open(file_path, "r") as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line by whitespace
                parts = line.strip().split(";")
                if parts:  # Ensure the line is not empty
                    # Get the last item and add it to the array
                    image_paths.append((parts[2], parts[1]))
        return image_paths
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []
    
def get_image_path(character_name, file_path="database.txt"):
    """
    Retrieves the image path for a given character name from the database file.

    Args:
        character_name (str): The name of the character to search for.
        file_path (str): The path to the database file (default is "database.txt").

    Returns:
        str: The image path if found, or None if no match is found.
    """
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Split the line by whitespace
                parts = line.strip().split(";")
                # Ensure the line has at least three parts (rare token, character name, image path)
                if len(parts) >= 3 and parts[1] == character_name:
                    return parts[2]  # Return the image path
        return None  # No match found
    except FileNotFoundError:
        print(f"Error: File '{file_path}' does not exist.")
        return None

def extract_names_without_rare_tokens(file_path):
    names = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Split the line to separate the rare token and name
                parts = line.strip().split(";")  # Split at the first space
                _, name, _1, _2 = parts  # Exclude the first part (rare token)
                names.append(name)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return names

def extract_character_name_by_image(image_path, file_path="database.txt"):
    # Open the file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components (rare_token + character_name + image_path)
            parts = line.strip().split(";")
            
            # Ensure the line has the correct format (3 parts)
            if len(parts) == 3:
                rare_token, character_name, current_image_path, prompt = parts
                
                # Check if the current image path matches the provided image path
                if current_image_path == image_path:
                    return character_name  # Return the character_name if it matches
    
    # Return None if no match is found
    return None


# Extract rare tokens
def extract_rare_tokens(file_path):
    """
    Extract rare tokens from a given file where each line contains a token after ': '.
    """
    rare_tokens = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                token = line.split(": ", 1)[1].split("->")[0].strip()
                rare_tokens.append(token)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return rare_tokens

    
def populate_dropdowns():
    global character_names
    
    if (os.path.getsize(database_file) == 0):
        return [gr.update(choices=["None"], value="None") for _ in range(10)]
    character_names = extract_names_without_rare_tokens(database_file)
    return [gr.update(choices=["None"] + character_names, value=character_names[0]) for _ in range(10)]

def exist_name(character_name, file_path="database.txt"):
    try:
        with open(file_path, "r") as file:
            # Read all lines from the file
            lines = file.readlines()
            if os.path.getsize(file_path) == 0:
                return False
            # Check if the input string exists in the file
            for line in lines:
                if character_name.strip() == line.split(";")[1].strip():
                    return True  # Found the string
        return False  # String not found
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    

def get_character_prototype(character_name, file_path="database.txt"):

    with open(file_path, "r") as file:
        # Read all lines from the file
        lines = file.readlines()
        if os.path.getsize(file_path) == 0:
            return None
        # Check if the input string exists in the file
        for line in lines:
            if character_name.strip() == line.split(";")[1].strip():
                return line.split(";")[3].strip()  # Found the string
      


def save_character_to_database(character_name, rare_token, image_path, character_prompt, database_file="database.txt"):
    """
    Save the combination of character name and rare token to a database file.
    """
    # save_character_embed(pipe, character_name)
    character_info = rare_token + ";" + character_name + ";" + image_path + ";" + character_prompt
    with open(database_file, "a") as db_file:
        db_file.write(f"{character_info}\n")
    print(f"Saved: {character_info}")
    

def get_most_similar_sentence(query, corpus):
    query = query.replace(prompt_suffix, "")
    # print("prompt query:", query)
    ranks = similarity_model.rank(query, corpus)

    max_rank = max(ranks, key=lambda x: x["score"])
    max_sentence = corpus[max_rank['corpus_id']]
    # print("max rank:", max_rank, "max sentence:", max_sentence)
    return max_sentence if not (max_sentence == "standing in T-shape") else "tpose"


def get_control_image(pose, pose_image_folder):
    png_files = [f for f in os.listdir(pose_image_folder) if f.startswith(pose) and f.endswith(".png")]
    return os.path.join(pose_image_folder, random.choice(png_files))


def add_padding(image_path, output_path, portrait_portion):
    with Image.open(image_path) as img:
        target_width = round(img.height / portrait_portion[1] * portrait_portion[0])
        total_padding = target_width - img.width
        left, right = total_padding // 2, total_padding - (total_padding // 2)
        ImageOps.expand(img, border=(left, 0, right, 0), fill='black').save(output_path)

# def add_padding_vertical(image_path, output_path, portrait_portion, portrait_type):
#     with Image.open(image_path) as img:
#         target_height = round(img.width / 219 * 292) if portrait_portion == [292, 219] else round(img.width / 457 * 440)
#         upper, lower = 0, 0
#         if portrait_type == "upper body poratrit":
#             upper = target_height - img.height
#         if portrait_type == "lower body poratrit":
#             lower = target_height - img.height
#         ImageOps.expand(img, border=(0, upper, 0, lower), fill='white').save(output_path)



def adapt_portrait(portrait_type, image_path, output_path, is_control_image=False, portrait_portion=[292, 219]):
    with Image.open(image_path) as img:
        width, height = img.size
        
        # if portrait_portion is None:
        #     if portrait_type == "full body portrait":
        #         img.save(output_path)
                
        #         return
        #     elif portrait_type == "upper body portrait":
        #         add_padding_vertical(image_path, output_path, portrait_portion, portrait_type)
        #         new_height = height * 0.5
        #         cropped_img = img.crop((0, 0, width, new_height))
        #         # Save the cropped image
        #         cropped_img.save(output_path)
        #     elif portrait_type == "lower body portrait":
        #         add_padding_vertical(image_path, output_path, portrait_portion, portrait_type)
                
        #         new_height = height * 0.5
        #         cropped_img = img.crop((0, height - new_height, width, new_height))
        #         # Save the cropped image
        #         cropped_img.save(output_path)
        #     return
        
        # if portrait_portion is False:
        #     if portrait_type == "full body portrait":
        #         img.save(output_path)
                
        #         return
        #     elif portrait_type == "upper body portrait":
        #         add_padding_vertical(image_path, output_path, portrait_portion, portrait_type)
        #         # new_height = height * 0.5
        #         # cropped_img = img.crop((0, 0, width, new_height))
        #         # # Save the cropped image
        #         # cropped_img.save(output_path)
        #     elif portrait_type == "lower body portrait":
        #         add_padding_vertical(image_path, output_path, portrait_portion, portrait_type)
                
        #         # new_height = height * 0.5
        #         # cropped_img = img.crop((0, height - new_height, width, new_height))
        #         # # Save the cropped image
        #         # cropped_img.save(output_path)
        #     return
        
        if portrait_type == "full body portrait":
            add_padding(image_path, output_path, portrait_portion)

        elif portrait_type == "upper body portrait":
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
            
            


def preprocess_pose_prompt(query, image_path, output_path_pose, output_path_portrait, portrait_portion, is_full_portrait):
    pose = get_most_similar_sentence(query, pose_corpus)
    # if is_full_portrait:
    #     portrait_type = "full body portrait"
    # else:
    portrait_type = get_most_similar_sentence(query, portrait_corpus)
        
    control_image_path = get_control_image(pose, pose_image_folder)
    adapt_portrait(portrait_type, control_image_path, output_path_pose, is_control_image=True, portrait_portion=portrait_portion)
    adapt_portrait(portrait_type, image_path, output_path_portrait, portrait_portion=portrait_portion)
    portrait_image = load_image(output_path_portrait)
    rmbg_img = rmbg_fn(portrait_image)
    rmbg_img.save(output_path_portrait)
    
     
def preprocess_pose_image(prompt, pose_skeleton_path, input_image_path,  portrait_portion, index="", is_full_portrait=False):
    # if characters[0] == "None" or characters[1] == "None":
        # user chose 1 character       
    if index is None:
        return None
    
    #  adapted pose skeleton's image path
    output_path_pose = "preprocessed_pose{}.png".format(index)
    
    # adapted input character prototype's image path
    output_path_portrait = "preprocessed_portrait{}.png".format(index)
    
    # adapted control pose skeleton image
    # pose = get_most_similar_sentence(prompt, pose_corpus)
    # if is_full_portrait:
    #     portrait_type = "full body portrait"
    # else:
    portrait_type = get_most_similar_sentence(prompt, portrait_corpus)
    # print("sentence tf porttrait type:", portrait_type)
    # control_image_path = get_control_image(pose, pose_image_folder)
    adapt_portrait(portrait_type, pose_skeleton_path, output_path_pose, is_control_image=True, portrait_portion=portrait_portion)
    adapt_portrait(portrait_type, input_image_path, output_path_portrait, portrait_portion=portrait_portion)
    portrait_image = load_image(output_path_portrait)
    rmbg_img = rmbg_fn(portrait_image)
    rmbg_img.save(output_path_portrait)
    # preprocess_pose_prompt(prompt, input_image, output_path_pose, output_path_portrait, portrait_portion=portrait_portion, is_full_portrait=is_full_portrait)


def concat_images(image1_path, image2_path, output_path, padding=0):
    # Open the two images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    # Calculate the dimensions of the new image
    new_height = max(img1.height, img2.height)
    # new_width = new_height * 1.33 if portrait_portion == [292, 219] else new_height * 1.04
    new_width = img1.width + img2.width + padding
    
    # Create a new blank image with a white background
    new_image = Image.new("RGB", (int(new_width), int(new_height)), (255, 255, 255))
    
    # Paste the first image onto the new image
    new_image.paste(img1, (0, (new_height - img1.height) // 2))
    
    # Paste the second image with a 10px padding
    new_image.paste(img2, (img1.width + padding, (new_height - img2.height) // 2))
    
    # Save the concatenated image
    new_image.save(output_path)
        
        
def preprocess_pose_image_single_characters(character_name, prompt, index=""):
    output_path_portrait = "preprocessed_portrait_single_character{}.png".format(index)
    output_pose_portrait = "preprocessed_pose_single_character{}.png".format(index)
    portrait_type = get_most_similar_sentence(prompt, portrait_corpus)
    
    input_img_path = get_image_path(character_name)
    adapt_portrait(portrait_type, input_img_path, output_path_portrait, portrait_portion=None)
    crop_sides(output_path_portrait, output_path_portrait)
    
    pose = get_most_similar_sentence(prompt, pose_corpus)
    control_image_path = get_control_image(pose, pose_image_folder)
    adapt_portrait(portrait_type, control_image_path, output_pose_portrait, portrait_portion=None)
    crop_sides(output_pose_portrait, output_pose_portrait)
    return portrait_type
    
    

def combine_pose_image_multiple_characters(image_paths, type, portrait_type, portrait_portion=[292, 219]):
    concat_output_path = "preprocessed_{}_concat.png".format(type)
    concat_images(image_paths[0], image_paths[1], concat_output_path)
    adapt_portrait(portrait_type, concat_output_path, concat_output_path, portrait_portion=False)
    

# def concat_images()  
def rmbg_fn(input_image):
    global rmbg_model
    try: 
        # load model
        if (rmbg_model.pipe is None):
            rmbg_model.multi_thread_load_model()  
        
        # infer
        res = rmbg_model.infer(input_img=input_image)

        return res
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            raise gr.Error("GPU out of memory. Please delete some models.") 

def crop_sides(image_path, output_path):
    with Image.open(image_path) as img:
        # Get original dimensions
        original_width, original_height = img.size
        
        # Calculate crop dimensions
        crop_width = original_width * 0.2  # 20% of the original width
        left_crop = crop_width / 2  # 10% from each side
        right_crop = original_width - left_crop
        
        # Perform the crop
        cropped_img = img.crop((left_crop, 0, right_crop, original_height))
        
        # Save the cropped image
        cropped_img.save(output_path)

    
    
def modify_character(regenerate_character_name, image_path, character_prompt, database_file="database.txt"):
    with open(database_file, "r") as file:
        lines = file.readlines()
    print("character name:", regenerate_character_name)
    updated_lines = []
    for line in lines:
        if regenerate_character_name in line:
            print('character in line')
            line = (";".join(line.strip().split(";")[:2] + [image_path] + [character_prompt]) + "\n") 
            
        updated_lines.append(line)
    
    print(updated_lines)
    if any(regenerate_character_name in line for line in lines):
        with open(database_file, "w") as file:
            file.writelines(updated_lines)

def detect_faces(image_path):
    text_bubble = Image.open("/workspace/text_bubble.png")
    detector = create_detector('yolov3')
    image = cv2.imread(image_path)
    preds = detector(image)
    print(preds)
    
    face_bbox = random.choice(preds)["bbox"]
    return face_bbox[:4]
    background_image = Image.open(image_path).convert("RGB")
    # if have face
    left_gap = face_bbox[0]
    right_gap = image.shape[1] - face_bbox[2]
    padding = 10
    if left_gap > right_gap:
        text_bubble = text_bubble.transpose(Image.FLIP_LEFT_RIGHT)
        position = (face_bbox[0] - padding, face_bbox[1] + padding)
        # background_image = background_image.transpose(Image.FLIP_LEFT_RIGHT).
    
    # if doesnt have face
    

def paste_text_bubble_safe(background_path, text_bubble_path, face_box):
    """
    Paste a text bubble near the bounding box of a detected face, ensuring
    it does not cover the face and remains fully within the background image.

    Parameters:
        background_path (str): Path to the background image.
        text_bubble_path (str): Path to the text bubble image.
        face_box (tuple): Bounding box of the face (left, top, right, lower).

    Returns:
        Image: The modified image with the text bubble.
    """
    # Open the images
    background = Image.open(background_path)
    text_bubble = Image.open(text_bubble_path)
    
    # Get dimensions of the images
    bg_width, bg_height = background.size
    bubble_width, bubble_height = text_bubble.size
    
    # Unpack the face bounding box
    left, top, right, lower = face_box
    face_width = right - left
    face_height = lower - top
    
    # Define potential positions for the bubble (relative to the face)
    potential_positions = [
        (right + 10, top),  # Right of the face
        (left - bubble_width - 10, top),  # Left of the face
        (left, lower + 10),  # Below the face
        (left, top - bubble_height - 10)  # Above the face
    ]
    
    # Find the first position that keeps the bubble entirely within the image
    for bubble_x, bubble_y in potential_positions:
        if (0 <= bubble_x <= bg_width - bubble_width) and (0 <= bubble_y <= bg_height - bubble_height):
            position = (bubble_x, bubble_y)
            break
    else:
        # If no suitable position is found, default to (0, 0)
        position = (0, 0)
    
    # Paste the text bubble on the background
    background.paste(text_bubble, position, text_bubble if text_bubble.mode == "RGBA" else None)
    background.save("test_text_bubble.png")
    return background

def detect_faces(image_path, model_name="face_detect_v1.4_s"):
    image = load_image(image_path)
    preds = FaceDetection().detect(image, model_name, )
    if len(preds) > 0:
        bbox = preds[0][0]
    print("face bbox:", bbox)
    
    return bbox

def draw_text_bubble_on_image(
    background_path: str,
    text: str,
    max_size: (int, int),
    output_path: str = "output_with_text_bubble.jpg"
):
    """
    Draw multiline, wrapped text into the *circular portion* of an existing 
    bubble image ("/workspace/text_bubble.png"), then paste that bubble
    near the face bounding box (if any) in the background image.

    Procedure:
      1. get the detected face bounding box
      2. if there is a face detected: calculate the left boundary to image left boundary distance,
         the right boundary to image right boundary distance => then put the text bubble at the
         larger-distance side
      3. if there is no face detected: place it on middle a bit left / middle a bit right with 
         a 0.5 random probability

    Parameters:
        background_path (str): Path to the background image.
        text (str): The text to place inside the bubble.
        max_size (tuple): (max_width, max_height) for bubble resizing.
        output_path (str): Where to save the final image. Default: 'output_with_text_bubble.jpg'.

    Returns:
        PIL.Image.Image: The modified background image with the pasted text bubble.
    """
    # 1. Get face bounding box (if any)
    face_bbox = detect_faces(background_path)  # (left, top, right, lower) or None

    # ---------------------------
    # 1. Load background & bubble
    # ---------------------------
    bg_image = Image.open(background_path).convert("RGBA")
    bubble_img = Image.open("/workspace/text_bubble.png").convert("RGBA")
    
    bubble_w, bubble_h = bubble_img.size
    max_w, max_h = max_size

    # Resize bubble if it's bigger than max_size
    ratio = min(max_w / bubble_w, max_h / bubble_h)
    if ratio < 1.0:
        new_w = int(bubble_w * ratio)
        new_h = int(bubble_h * ratio)
        bubble_img = bubble_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bubble_w, bubble_h = bubble_img.size

    # ---------------------------------------------------
    # 2. Define the circle portion's bounding box (no tail)
    #    Tweak these values to match your bubble image.
    # ---------------------------------------------------
    circle_left   = int(0.10 * bubble_w)  # 10% from left
    circle_right  = int(0.90 * bubble_w)  # 90% from left
    circle_top    = int(0.0 * bubble_h)   # e.g. 0% from top
    circle_bottom = int(0.80 * bubble_h)  # 80% from top

    circle_width  = circle_right - circle_left
    circle_height = circle_bottom - circle_top
    
    # ----------------------------------------------------------------------
    # 3. Find a font size that allows the text to wrap & fit in the circle
    # ----------------------------------------------------------------------
    font_path = "/workspace/fonts/font1.ttf"  # Update with your actual font path
    font_size = 34
    min_font_size = 10
    padding = 10
    line_spacing_mult = 1.2  # Spacing between lines

    if face_bbox:
        # Face detected
        left, top, right, lower = face_bbox
        distance_left  = left  # distance from face's left boundary to image's left
        distance_right = bg_image.width - right  # distance from face's right boundary to image's right
        
        # Choose side with the larger distance
        if distance_left > distance_right:
            # Place bubble to the left of the face
            paste_x = left - bubble_w - 10
            bubble_img = bubble_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # Place bubble to the right of the face
            paste_x = right + 10

        # Align vertically near the top of face
        paste_y = top

        # Boundary checks
        if paste_x < 0:
            paste_x = 0
        if paste_x + bubble_w > bg_image.width:
            paste_x = bg_image.width - bubble_w

        if paste_y + bubble_h > bg_image.height:
            paste_y = bg_image.height - bubble_h
        if paste_y < 0:
            paste_y = 0

    else:
        # No face detected
        # We place the bubble in the middle, offset left or right at random
        center_x = (bg_image.width - bubble_w) // 2
        center_y = (bg_image.height - bubble_h) // 2

        # 0.5 chance for left vs right offset
        if random.random() < 0.5:
            # "middle a bit left" => shift slightly negative in x
            paste_x = center_x - 20
        else:
            # "middle a bit right" => shift slightly positive in x
            paste_x = center_x + 20

        paste_y = center_y

        # Boundary checks
        if paste_x < 0:
            paste_x = 0
        if paste_x + bubble_w > bg_image.width:
            paste_x = bg_image.width - bubble_w

        if paste_y < 0:
            paste_y = 0
        if paste_y + bubble_h > bg_image.height:
            paste_y = bg_image.height - bubble_h
    def wrap_text_lines(current_font: ImageFont.FreeTypeFont):
        """
        Wraps the text into multiple lines that fit within the horizontal boundary 
        defined by `circle_left` and `circle_right` and the font's pixel width.
        """
        draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines = []
        words = text.split()
        current_line = []

        max_line_width = circle_width - 2 * padding

        for word in words:
            test_line = " ".join(current_line + [word])
            line_w = draw_temp.textlength(test_line, font=current_font)
            if line_w <= max_line_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def does_text_fit_given_font(current_font: ImageFont.FreeTypeFont):
        """
        Wrap the text by measuring each candidate line's pixel width.
        Then check if the total height of all lines (+ padding) 
        exceeds the circle's height.
        """
        wrapped_lines = wrap_text_lines(current_font)

        # Measure total height
        total_height = 0
        for ln in wrapped_lines:
            line_h = font_size
            total_height += int(line_h * line_spacing_mult)

        # Add top & bottom padding
        total_height += 2 * padding

        return total_height <= circle_height

    # Iteratively reduce the font size until text fits or min_font_size is reached
    while font_size >= min_font_size:
        test_font = ImageFont.truetype(font_path, font_size)
        if does_text_fit_given_font(test_font):
            break
        font_size -= 2

    final_font = ImageFont.truetype(font_path, font_size)

    # -------------------------------------------------------------------
    # 4. Wrap the text (again) using the *final* font, and measure lines
    # -------------------------------------------------------------------
    wrapped_lines = wrap_text_lines(final_font)

    # Measure total wrapped height
    bubble_draw = ImageDraw.Draw(bubble_img)  # We'll draw on the bubble
    total_height = 0
    line_heights = []

    for ln in wrapped_lines:
        line_h = font_size
        line_heights.append(line_h)
        total_height += int(line_h * line_spacing_mult)

    # Where to start drawing so it's vertically centered
    y_offset = circle_top + (circle_height - total_height) // 2

    # Draw each line horizontally centered
    for ln in wrapped_lines:
        line_w = bubble_draw.textlength(ln, font=final_font)
        x_offset = circle_left + (circle_width - line_w) // 2
        bubble_draw.text((x_offset, y_offset), ln, fill=(0, 0, 0, 255), font=final_font)
        y_offset += int(font_size * line_spacing_mult)

    # ------------------------------------------------
    # 5. Position bubble according to the procedure
    # ------------------------------------------------
    

    # ---------------------------------------
    # 6. Composite bubble onto background
    # ---------------------------------------
    bg_image.alpha_composite(bubble_img, (paste_x, paste_y))

    # 7. Save the final result
    bg_image.convert("RGB").save(output_path)
    # print(f"Saved output to {output_path}")
    return bg_image
def extract_quotes(sentence):
    """
    Extract all content between double quotation marks in a sentence.

    Parameters:
        sentence (str): The input string.

    Returns:
        list: A list of strings found between double quotation marks.
    """
    return re.findall(r'"(.*?)"', sentence)  


def save_pdf(pathA="/workspace/reference_image0.png", 
             pathB="/workspace/reference_image1.png", 
             pathC="/workspace/reference_image2.png", 
             pathD="/workspace/reference_image3.png", output_pdf="layout_2x2.pdf", border_size=2, border_color=(0, 0, 0), padding=10):
    """
    Opens two images (pathA, pathB) and places them side by side 
    in a single row, matching their heights.
    Returns a Pillow Image object containing the combined row.
    """
    imgA = Image.open(pathA).convert("RGB")
    imgB = Image.open(pathB).convert("RGB")
    imgC = Image.open(pathC).convert("RGB")
    imgD = Image.open(pathD).convert("RGB")
 
    # Resize image B
    imgA_resized = imgA.resize((708, 914))
    ImageOps.expand(imgA_resized, border=border_size, fill=border_color)
    
    imgB_resized = imgB.resize((710, 560))
    ImageOps.expand(imgB_resized, border=border_size, fill=border_color)
    
    imgC_resized = imgC.resize((710, 560))
    ImageOps.expand(imgC_resized, border=border_size, fill=border_color)
    
    imgD_resized = imgD.resize((710, 914))
    ImageOps.expand(imgD_resized, border=border_size, fill=border_color)
    
    
    wA, hA = imgA_resized.size
    wB, hB = imgB_resized.size
    
    # The total width is the sum of widths
    combined_width = wA + wB + 3 * padding
    new_height = hA + hB + 3 * padding
    
    # Create a new blank canvas
    combined = Image.new(mode="RGB", size=(combined_width, new_height), color=(255, 255, 255))
    
    # Paste image A at (0,0)
    combined.paste(imgA_resized, (padding, padding))
    # Paste image B at (wA, 0)
    combined.paste(imgB_resized, (2 * padding + wA, padding))
    combined.paste(imgC_resized, (padding, 2 * padding + hA))
    combined.paste(imgD_resized, (2 * padding + wA, 2 * padding + hB))
    combined.save(output_pdf, "PDF")
    # return combined

# pose = '"you are cute!", cute, ponytail, "come play with me!"'
# dialogue = ' '.join(extract_quotes(pose))
# print(dialogue)
# draw_text_bubble_on_image("/workspace/prototype0.png", "hello i am Kaijo Tsukunawa. Nice to not meet you.", (400,400), )
# paste_text_bubble_safe("/workspace/reference_image0.png", "/workspace/text_bubble.png", (0, 0, 0, 0))
