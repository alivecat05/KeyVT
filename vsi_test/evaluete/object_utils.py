import os
import numpy as np
from PIL import Image

import cv2
import re
import string
import torch
def query_transform(dataset,query):
    rules = {
        "arkitscenes": {
            "tv_monitor": "tv"
        },
        "scannet": {
            "object": "discard_object",
            "desk": "table",
            "trash can": "trash bin",
            "recycling bin": "trash bin",
            "television": "tv",
            "trash bag": "trash bin",
            "office chair": "chair",
            "dining table": "table",
            "mouse": "computer mouse",
            "armchair": "chair",
            "couch": "sofa",
            "kitchen counter": "counter",
            "beanbag chair": "chair",
            "desk lamp": "lamp",
            "lamp base": "lamp",
            "night lamp": "lamp",
        },
        "scannetpp": {
            "object": "discard_object",
            "ceiling lamp": "ceiling light",
            "soap dispenser": "hand soap",
            "office chair": "chair",
            "dining table": "table",
            "dining chair": "chair",
            "coat hanger": "coat rack",
            "mouse": "computer mouse",
            "mouse pad": "computer mouse",
            "armchair": "chair",
        }
    }
    dataset_rules = rules[dataset]
    for s in query.split():
        if s in dataset_rules:
            query = query.replace(s, dataset_rules[s])
    
    return query

def transform_label__1(dataset, instance_info):
    rules = {
        "arkitscenes": {
            "tv_monitor": "tv"
        },
        "scannet": {
            "object": "discard_object",
            "desk": "table",
            "trash can": "trash bin",
            "recycling bin": "trash bin",
            "trash bag": "trash bin",
            "television": "tv",
            "office chair": "chair",
            "dining table": "table",
            "mouse": "computer mouse",
            "armchair": "chair",
            "couch": "sofa",
            "kitchen counter": "counter",
            "beanbag chair": "chair",
            "desk lamp": "lamp",
            "lamp base": "lamp",
            "night lamp": "lamp",
        },
        "scannetpp": {
            "object": "discard_object",
            "ceiling lamp": "ceiling light",
            "soap dispenser": "hand soap",
            "office chair": "chair",
            "dining table": "table",
            "dining chair": "chair",
            "coat hanger": "coat rack",
            "mouse": "computer mouse",
            "mouse pad": "computer mouse",
            "armchair": "chair",
        }
    }
    
    dataset_rules = rules[dataset]

    new = []
    for obj in instance_info:
        if obj in list(dataset_rules.keys()):
            new.append(dataset_rules[obj])

    return new if new!=[] else instance_info
def transform_label(dataset, instance_info):
    rules = {
        "arkitscenes": {
            "tv_monitor": "tv"
        },
        "scannet": {
            "object": "discard_object",
            "desk": "table",
            "trash can": "trash bin",
            "recycling bin": "trash bin",
            "television": "tv",
            "office chair": "chair",
            "dining table": "table",
            "mouse": "computer mouse",
            "armchair": "chair",
            "couch": "sofa",
            "kitchen counter": "counter",
            "beanbag chair": "chair",
            "desk lamp": "lamp",
            "lamp base": "lamp",
            "night lamp": "lamp",
        },
        "scannetpp": {
            "object": "discard_object",
            "ceiling lamp": "ceiling light",
            "soap dispenser": "hand soap",
            "office chair": "chair",
            "dining table": "table",
            "dining chair": "chair",
            "coat hanger": "coat rack",
            "mouse": "computer mouse",
            "mouse pad": "computer mouse",
            "armchair": "chair",
        }
    }
    
    # dataset_rules = rules[dataset]
    # for instance_id, obj in instance_info.items():
    #     label = obj["label"]
    #     if label in dataset_rules:
    #         instance_info[instance_id]["label"] = dataset_rules[label]

    # return instance_info
    dataset_rules = rules[dataset]

    for instance_id, obj in enumerate(instance_info):
        label = obj["label"]
        if label in dataset_rules:
            instance_info[instance_id]["label"] = dataset_rules[label]

    return instance_info

def extract_labels(question, question_type):
    object_list = []
    if 'object_rel_direction' in question_type:##
        pattern = r"I am standing by the (.*?) and facing the (.*?), is the (.*?) to"
        match = re.search(pattern, question)
        if match:
            standing = match.group(1).strip()
            facing = match.group(2).strip()
            target = match.group(3).strip()
            object_list = [standing, facing, target]
    
    elif question_type == 'object_rel_distance':
        list_match = re.search(r"\((.*?)\)", question)
        object_list = [obj.strip() for obj in list_match.group(1).split(',')] if list_match else []
        reference_match = re.search(r"closest to the ([a-zA-Z0-9_ ]+)", question)
        reference_object = reference_match.group(1).strip() if reference_match else None
        if reference_object is not None:
            object_list.append(reference_object)

    elif question_type == 'obj_appearance_order':
        list_match = re.search(r"categories.*?:\s*(.*)", question)
        object_list = [obj.strip().rstrip(string.punctuation) for obj in list_match.group(1).split(',')] if list_match else []

    elif question_type == 'object_counting':
        obj_match = re.search(r"How many (.+?)\(s\)", question)
        if obj_match:
            object_list = [obj_match.group(1)]

    elif question_type == 'object_size_estimation':
        obj_match = re.search(r"of the ([^,]+?), measured", question)
        if obj_match:
            object_list = [obj_match.group(1)]

    elif question_type == 'object_abs_distance':###
        pattern = r"distance between the (.+?) and the (.+?) \(in meters\)\?"
        match = re.search(pattern, question)
        if match:
            src_obj = match.group(1)
            tgt_obj = match.group(2)
            object_list = [src_obj, tgt_obj]

    elif question_type == 'route_planning':###
        location_obj, facing_obj, target_obj = None, None, None
        location_pattern = r"beginning (?:at|by) the (.*?)(?:\s+and\s+facing|\s+facing|,|$)"
        match = re.search(location_pattern, question)
        if match:
            location_obj = match.group(1)
        facing_pattern = r"facing(?:\s+(?:the|to|toward the|into the|out the))?\s+(.*?)(?:\.|\?|$)"
        match = re.search(facing_pattern, question)
        if match:
            facing_obj = match.group(1)
        target_pattern = r"navigate to the (.*?)(?:\.|\?|$)"
        match = re.search(target_pattern, question)
        if match:
            target_obj = match.group(1)
        object_list = [location_obj, facing_obj, target_obj]
        object_list = [x for x in object_list if x]

    return object_list

def filter_objs_with_labels(question_labels, label_to_ids, all_objs):
    selected_labels = []
    for label, ids in label_to_ids.items():
        if label in question_labels and label not in selected_labels:
            selected_labels.append(label)
    selected_objs = [{**{"id": obj['id']}, **obj} for obj in all_objs if obj['label'] in selected_labels]
    return selected_objs

def extract_labels_direct(question,label_to_ids):
    candidate_labels = list(label_to_ids.keys())
    question = question.lower()
    question_labels = []
    # Tokenize the question to get individual words for matching
    question_words = set(re.findall(r'\b\w+\b', question))

    for label in candidate_labels:
        # Check for whole label match first
        if re.search(r'\b' + re.escape(label) + r'\b', question):
            if label not in question_labels:
                question_labels.append(label)
        else:
            # If no whole match, check for partial word matches
            label_parts = label.split()
            for part in label_parts:
                if part in question_words:
                    if label not in question_labels:
                        question_labels.append(label)
                    break  # Move to the next label once a match is found
    return question_labels

def extract_labels_w_lmms(model, processor,query, frames=None):
    with torch.no_grad():
        system_prompt = (
            f"""You are a spatial reasoning assistant. Extract all object names mentioned in the following situation and question that are relevant for spatial reasoning.

            Situation and Question:
            "{query}

            Respond ONLY with a comma-separated list of object names (no articles, no adjectives, no explanations). Use singular form.

            Example:
            Input: "I am standing by the stove and facing the TV. Is the sofa to my left?"
            Output: stove, tv, sofa"""
        )
        messages = [
                {
                    "role": "user",
                    "content": [
                    ],
                }
            ]
        if frames is not None:
            for img in frames: # type: ignore
                messages[0]['content'].append(
                    {"type": "image", "image": img}
                )
        messages[0]['content'].append(
                {"type": "text", "text": system_prompt}
        )

        text_template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if frames is None:
            image_inputs = None
        else:
            image_inputs = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        inputs = processor(images=image_inputs, text=text_template, return_tensors='pt').to(model.device, torch.float16)

        generated_ids = model.generate(**inputs,max_new_tokens=200, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()  
def extract_labels_and_filter_objs(question, question_type, label_to_ids, all_objs):
    question_labels = extract_labels(question, question_type) ## rule based extraction
    return filter_objs_with_labels(question_labels, label_to_ids, all_objs), question_labels

def extract_labels_and_filter_objs_direct(question, question_type):
    question_labels = extract_labels(question,question_type)
    return question_labels

def draw_marks(image, objects, alpha=0.5, base_square_size=36):
    overlay = image.copy()

    # Define a list of vibrant colors for object markers (BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (0, 128, 255),  # Orange
        (128, 0, 128),  # Purple
        (0, 255, 128),  # Spring Green
        (255, 128, 0)   # Light Blue
    ]

    # First, draw all the rectangles on the overlay with unique colors
    for i, obj_data in enumerate(objects):
        obj_id = str(obj_data['id'])
        x, y = map(int, obj_data["image_position"])

        square_width = base_square_size + 10 * (len(obj_id) - 1)
        square_height = base_square_size

        top_left = (x - square_width // 2, y - square_height // 2)
        bottom_right = (x + square_width // 2, y + square_height // 2)
        
        # Assign a unique color from the list for the rectangle
        rect_color = colors[i % len(colors)]
        cv2.rectangle(overlay, top_left, bottom_right, rect_color, -1)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Now, draw the text, choosing a color for contrast
    for i, obj_data in enumerate(objects):
        obj_id = str(obj_data['id'])
        x, y = map(int, obj_data["image_position"])

        font_scale = 0.9
        text = obj_id + ' ' + obj_data['label']
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        # Define the region where text will be drawn
        text_w, text_h = text_size
        text_top_left = (text_x, text_y - text_h)
        text_bottom_right = (text_x + text_w, text_y)

        # Ensure the region is within image bounds
        (tl_x, tl_y) = (max(0, text_top_left[0]), max(0, text_top_left[1]))
        (br_x, br_y) = (min(image.shape[1], text_bottom_right[0]), min(image.shape[0], text_bottom_right[1]))

        # Assign a different color for the text
        font_color = colors[(i + 1) % len(colors)]

        if br_y > tl_y and br_x > tl_x:
            bg_roi = image[tl_y:br_y, tl_x:br_x]
            
            # Calculate the average brightness of the background
            gray_roi = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_roi)
            
            # If background is too bright, use black for better contrast
            if brightness > 180:
                font_color = (0, 0, 0)
        else:
            # Default to black if region is invalid
            font_color = (0, 0, 0)

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 3)

    return image

def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No significant content detected in the image, skipping cropping.")
        return image

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def get_images_from_folder(folder, img_suffix='.png'):
    images = []
    if os.path.exists(folder) and os.path.isdir(folder):
        img_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(img_suffix)])

        for img_file in img_files:
            try:
                with Image.open(img_file) as img:
                    img = img.convert('RGB')
                    images.append(img)  # Pass opened image
            except Exception as e:
                print(f"Error encoding image {img_file}: {e}")
    else:
        print(f"Warning: frame image folder {folder} does not exist.")

    return images