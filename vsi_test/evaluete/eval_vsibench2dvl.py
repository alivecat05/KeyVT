import sys
import json
import os
from aggregate_result import agg_data
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['CURL_CA_BUNDLE'] = ''
sys.path.insert(0,'/root/dws/3D_QA/TStar/cdViews')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/evaluate')
import argparse
import copy
import time
import re
from copy import deepcopy
import ray
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from transformers import logging
import numpy as np
from decord import VideoReader,cpu ,gpu # Use decord for video resolution checking
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from spatial_utils import clean_text, vsi_reward
from typing import List
from PIL import Image
from model import ModelConfig
# from init_retriever import *
from gaussian import sample_video_frames_w_fps
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "multiple choice": " Please answer with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags.",
    "numerical": " Please answer with the only numerical value  within the <answer> </answer> tags.",
    "regression": " Please answer with the only numerical value  within the <answer> </answer> tags.",
    "verbal": " Please answer the question simply within the <answer> </answer> tags",
}



# SFT_TYPE_TEMPLATE = {
#     "multiple choice": " Please answer with the option's letter from the given choices (e.g., A, B, etc.) ",
#     "numerical": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) ",
#     "regression": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) ",
#     "verbal": " Please answer the question simply ",
# }
def get_llavanext_video_model(model_id,device):
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map=device
        # load_in_4bit=True,
    ).eval()
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    processor.patch_size = model.config.vision_config.patch_size
    return model, processor
# def get_llava_model(model_id,device):
#     from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration        
#     model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#         model_id, 
#         torch_dtype=torch.float16, 
#         device_map=device
#         # load_in_4bit=True,
#     ).eval()
#     # model.to(device)
#     processor = AutoProcessor.from_pretrained(model_id)

#     return model, processor
# def llava_predict(model, processor,query, frames=None):
    
#     system_prompt = f"""{query}."""
#     messages = [
#             {
#                 "role": "user",
#                 "content": [
#                 ],
#             }
#         ]
#     for img in frames: # type: ignore
#         messages[0]['content'].append(
#             {"type": "image", "image": img}
#         )
#     messages[0]['content'].append(
#             {"type": "text", "text": system_prompt}
#     )

#     text_template = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
    
#     image_inputs = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
    
    
#     image_inputs = [img.resize((512,384)) for img in image_inputs]
    
    
#     inputs = processor(images=image_inputs, text=text_template, return_tensors='pt').to(model.device, torch.float16)

#     generated_ids = model.generate(
#         **inputs,
#         do_sample=False,          # 确保每次输出相同
#         # max_new_tokens=20,
#         pad_token_id=processor.tokenizer.pad_token_id,
#         eos_token_id=processor.tokenizer.eos_token_id
#     )
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     ans = output_text[0].strip()
#     if ans=='television':
#         ans='tv'
#     return ans


def get_llava_model(model_id,device):
    model_name= 'llava_qwen'
    pretrained = model_id
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device,attn_implementation=None)
    model.eval()
    return model, image_processor,tokenizer

def llava_predict(model, processor,tokenizer,query, frames=None,**kwargs):

    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN*len(frames) + query
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_inputs = processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, torch.half)
    cont = model.generate(
        input_ids,
        images=image_inputs,
        modalities= ["image"],
        do_sample=False,
        temperature=0.0,
        max_new_tokens=20,
        ratio=args.ratio,
        is_compress=args.is_compress,
        is_ot = args.is_ot,
        is_divprune = args.is_divprune,
    )
    ans = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    if ans=='television':
        ans='tv'
    if ans=='rectangle':
        ans = 'rectangular'
    return ans






def get_llava_video_model(model_id,device):
    model_name= 'llava_qwen'
    pretrained = model_id
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device,attn_implementation=None)
    model.eval()
    return model, image_processor,tokenizer

def llava_video_predict(model, processor,tokenizer,query, frames=None,**kwargs):

    # "You are a spatial reasoning assistant",
    #             "Thinking carefully and answer the question",
    #             query,
    #             "\nPlease don't output any 0 and 1!"
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN*len(frames) + "You are a spatial reasoning assistant."+"Thinking carefully and answer the question."+query+"\nPlease don't output any 0 and 1!"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_inputs = processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, torch.half)
    cont = model.generate(
        input_ids,
        images=image_inputs,
        modalities= ["image"],
        do_sample=False,
        temperature=0.0,
        max_new_tokens=20,
        is_compress=args.is_compress,
        ratio=args.ratio,
        is_ot = args.is_ot,
        is_divprune = args.is_divprune,
        # is_geometry = True,
    )
    ans = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    if ans=='television':
        ans='tv'
    if ans=='rectangle':
        ans = 'rectangular'
    return ans

def get_internVL_model(model_id,device):
    sys.path.insert(0,'/root/dws/3D_QA/TStar/cdViews/model')
    from InternVL3_1b.modeling_internvl_chat import InternVLChatModel
    from transformers import AutoModel, AutoTokenizer
    
    model = InternVLChatModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    return model, tokenizer


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB') if not isinstance(image_file,Image.Image) else image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    for i,img in enumerate(images):
        img.save(f'{i}.jpg')
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def internvl_predict(model,tokenizer,query,frames):
    generation_config = dict(max_new_tokens=1024, do_sample=False,temperature=0.0)
    
    pixel_values = [
    load_image(f.resize((480, 480))).to(torch.bfloat16).to(model.device, non_blocking=True)
    for f in frames
    ]   
    
    pixel_values = torch.cat(pixel_values, dim=0).to(model.device)
    question = '<image>\n'+query

    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True,iscompre_on=args.is_compress
                               ,compress_ratio=args.ratio, is_ot = args.is_ot)

    return response


def get_qwen_model(model_id,device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    ).eval()
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor
def qwen_predict(model, processor,query, frames=None):
    
    if args.is_compress:
        print('Remove instructions for compressed qwen inference')
        system_prompt = f"""{query}"""
    elif args.sample_strategy == 'space_aks':
        system_prompt = (
            "You are a spatial reasoning assistant",
            "Thinking carefully and answer the question",
            query,
            "\nPlease don't output any 0 and 1!"
        )
    else:
        system_prompt = f"""{query}."""
    messages = [
            {
                "role": "user",
                "content": [
                ],
            }
        ]
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
    
    image_inputs, video_inputs = process_vision_info(messages) # type: ignore
    inputs = processor(
        text=text_template,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Generate the output.
    generated_ids = model.generate(
        **inputs,
    )
    # Trim the input tokens so that only the generated answer remains.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def load_vsi_evalset():
    file_path = os.path.abspath(__file__)
    vsi_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", "eval_vsibench.json")
    with open(vsi_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_single_message_eval(item, video_root, video_nframes):
    """Prepare message structure for a single eval data point."""
    if item["problem_type"] == "multiple choice":
        question = item["problem"] + "Options:\n"
        for op in item["options"]:
            question += op + "\n"
    else:
        question = item["problem"]

    content = []
    data_type = item["data_type"]
    data_path = os.path.normpath(os.path.join(video_root, item["path"]))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found for path {data_path}")
    
    mm_content = {"type": data_type}
    if data_type == "image":
        mm_content["image"] = data_path
        content.append(mm_content)
    elif data_type == "video":
        mm_content["video"] = data_path
        if video_nframes != -1:
            mm_content["nframes"] = video_nframes
        content.append(mm_content)
    else:
        raise ValueError(f"Unsupported data_type '{data_type}' found for path {data_path}.")

    content.append(
            {
                "type": "text",
                "text": SFT_QUESTION_TEMPLATE.format(Question=question)
                + SFT_TYPE_TEMPLATE[item["problem_type"]]
            }
        )
    msg = [{"role": "user", "content": content}]
    return msg

def retrieval_prepare_single_message_eval(item, video_root, video_nframes):
    """Prepare message structure for a single eval data point."""
    if item["problem_type"] == "multiple choice":
        question = item["problem"] + "Options:\n"
        for op in item["options"]:
            question += op + "\n"
    else:
        question = item["problem"]

    content = []
    data_type = item["data_type"]
    data_path = os.path.normpath(os.path.join(video_root, item["path"]))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found for path {data_path}")
    
    mm_content = {"type": data_type}
    if data_type == "image":
        mm_content["image"] = data_path
        content.append(mm_content)
    elif data_type == "video":
        for f in item['key_frames']:
            mm_content = {"type": 'image'}
            mm_content["image"] = f

            content.append(mm_content)
    else:
        raise ValueError(f"Unsupported data_type '{data_type}' found for path {data_path}.")

    content.append(
            {
                "type": "text",
                "text": SFT_QUESTION_TEMPLATE.format(Question=question)
                + SFT_TYPE_TEMPLATE[item["problem_type"]]
            }
        )
    msg = [{"role": "user", "content": content}]
    return msg

def preprocess_batch(batch_data, processor, model_config, video_root, video_nframes):
    batch_messages = []
    for i, x in enumerate(batch_data):
        if x.get('key_frames') is not None:
            msg = retrieval_prepare_single_message_eval(x, video_root, video_nframes)
        else:
            msg = prepare_single_message_eval(x, video_root, video_nframes)
        batch_messages.append(msg)
    
    prompts_text = [
        processor.apply_chat_template(
            example, tokenize=False, add_generation_prompt=True
        )
        for example in batch_messages
    ]
    prompts_text_for_log = deepcopy(prompts_text)
        
    images_inputs = []
    video_inputs = []
    for example in batch_messages:
        imgs, vids = process_vision_info(example)
        if vids is not None and len(vids) > 0:
            video_inputs.extend(vids)
            images_inputs = None
        else:
            images_inputs.append(imgs)
            video_inputs = None
    batch = processor(
        text=prompts_text,
        images=images_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    
    if "spatial-mllm" in model_config.model_type:
        if video_inputs is not None and len(video_inputs) > 0: # Check if video_inputs is not empty
            video_inputs = torch.stack(video_inputs) / 255.0 # [B, T, C, H, W]
            batch.update({"videos_input": video_inputs})
            
    return batch, prompts_text_for_log

def inference_batch(batch_inputs, model, processor, model_config):
    inputs = {key: val.to(model.device) if isinstance(val, torch.Tensor) else val 
            for key, val in batch_inputs.items()}
        
    # Generate response
    start_time = time.time()
    with torch.no_grad(), torch.amp.autocast(device_type=str(model.device), dtype=model.dtype):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=model_config.max_tokens,
            do_sample=True if model_config.temperature > 0 else False,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            use_cache=True,
        )
        
        end_time = time.time()
        print(f"Time taken for generation: {end_time - start_time} seconds")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def postprocess_batch(batch_data, batch_output_text, prompts_text):
    batch_results = []
    for batch_idx, sample in enumerate(batch_data):
        model_output = batch_output_text[batch_idx]
        
        result_sample = {}
        result_sample['sample'] = sample.copy()
        result_sample["prompt"] = prompts_text[batch_idx]
        result_sample["model_output"] = model_output
        
        # --- if contains <answer> tags, extract answer, else use model_output as answer ---
        clean_ans = clean_text(model_output)
        result_sample["cleaned_model_output"] = clean_ans
        
        # --- get cleaned gt answer ---
        clean_ans_gt = clean_text(sample.get("solution", ""))
        result_sample["cleaned_gt_answer"] = clean_ans_gt
        
        # --- calculate reward ---
        result_sample["reward"] = vsi_reward(clean_ans_gt, clean_ans, sample['problem_type'])
        result_sample["correct"] = result_sample["reward"] == 1.0
        batch_results.append(result_sample)
    return batch_results

def calculate_metrics(results):
    """Calculate metrics from a list of results."""
    mean_acc_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") != "regression" and "reward" in s]
    mean_mra_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") == "regression" and "reward" in s and s.get("prediction") != "error"]

    final_metrics = {"mean_acc": 0.0, "mean_mra": 0.0, "mean_all": 0.0}
    if mean_acc_rewards:
            final_metrics["mean_acc"] = torch.tensor(mean_acc_rewards, dtype=torch.float32).mean().item()
    if mean_mra_rewards:
            final_metrics["mean_mra"] = torch.tensor(mean_mra_rewards, dtype=torch.float32).mean().item()
    if mean_acc_rewards or mean_mra_rewards:
        all_rewards = torch.cat([torch.tensor(mean_acc_rewards, dtype=torch.float32), torch.tensor(mean_mra_rewards, dtype=torch.float32)])
        final_metrics["mean_all"] = all_rewards.mean().item()
    return final_metrics

def save_results(output_path: str, results, final_acc):
    """Save evaluation results to file."""
    try:
        idx = len(results)-1
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if results[idx]['sample'].get('key_frames') is not None:
            # save_dir = os.path.join('/root/dws/3D_QA/Spatial-MLLM-master/eval_results/', 'key_frames_gaussian_clip_116/')
            # save_dir+=str(results[idx]['sample']['problem_id'])+'/'
            # frames = results[idx]['sample']['key_frames']
            # os.makedirs(save_dir, exist_ok=True)
            # for i,f in enumerate(frames):
            #     f.save(os.path.join(save_dir, f'{i}.jpg'))
            results[idx]['sample']['key_frames'] = 'extracted'
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "final_acc": [final_acc]},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Results saved to {output_path}")
        try:
            print(agg_data(output_path))
        except:
            print('Aggregation failed')
        
    except Exception as e:
        print(f"Error writing results to output file: {e}")
def sample_video_frames(visual, fps=1,max_num_frames  = 10) -> list:
    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
        vr = VideoReader(visual)
        image_num = len(vr)
        # sample max_num_frames frame indices from the video
        if image_num < max_num_frames:
            frame_indices = np.arange(image_num)
        else:
            frame_indices = np.linspace(0, image_num - 1, max_num_frames).astype(int)
        # read the frames
        frames = [vr[i].asnumpy() for i in frame_indices]
        frames_list = [
            Image.fromarray(frame).convert("RGB") 
            for frame in frames
        ]
        return frames_list
    return []
        
word_to_num = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    # 可根据需要扩展
}

def extract_number(sentence: str):
    # 匹配阿拉伯数字
    digit_match = re.search(r'\b\d+\b', sentence)
    if digit_match:
        return int(digit_match.group())
    
    # 匹配英文数字单词（仅限已知词汇）
    words = re.findall(r'\b\w+\b', sentence.lower())
    for word in words:
        if word in word_to_num:
            return word_to_num[word]
    
    return None  # 未找到数字


def inference_mllm(batch_data,model, processor,tokenizer):
    item = batch_data[0]
    if item["problem_type"] == "multiple choice":
        question = item["problem"] + "Options:\n"
        for op in item["options"]:
            question += op + "\n"
    else:
        question = item["problem"]
    
    # prompt = question + SFT_TYPE_TEMPLATE[item["problem_type"]]
    prompt = question + '\n' + SFT_TYPE_TEMPLATE[item["problem_type"]]
    print(f'Prompt: {prompt}')
    if args.model_name == 'llava' or args.model_name == 'llavanext':
        batch_output_text=llava_predict(model, processor,tokenizer,prompt, frames=item['key_frames'])
    elif args.model_name == 'llavavideo':
        batch_output_text=llava_video_predict(model, processor,tokenizer,prompt, frames=item['key_frames'])
    elif args.model_name=='qwen':
        batch_output_text = qwen_predict(model,processor,prompt,frames=item['key_frames'])
    elif args.model_name == 'internvl':
        batch_output_text = internvl_predict(model, tokenizer,prompt, frames=item['key_frames'])
    numerical_answer = extract_number(batch_output_text)
    if numerical_answer is not None:
        batch_output_text = str(numerical_answer)
    return [batch_output_text]


@ray.remote(num_gpus=1)
def evaluate_vsibench_r(axis,vsi_data, model_config, output_path, video_root, video_nframes, batch_size):
    start,end = axis
    # --- cache video resolutions to avoid reopening the same file repeatedly ---
    tokenizer = None
    processor = None
    resolution_cache = {}
    def get_resolution(path):
        """Return (width, height) of the video using decord; utilize cache when available."""
        if path in resolution_cache:
            return resolution_cache[path]
        try:
            vr = VideoReader(path, num_threads=1)
            raw_fps = vr.get_avg_fps()
            # Decord frame shape: (H, W, C)
            h, w, _ = vr[0].shape
            resolution_cache[path] = (w, h)
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path} with decord: {e}")
        return resolution_cache[path],raw_fps
    if args.model_name == 'llava':
        llava_model_path = args.llava_model_path
        llava_device = args.mllm_device
        model,processor,tokenizer = get_llava_model(llava_model_path,device=llava_device)
        # tokenizer = None
    elif args.model_name == 'llavanext':
        llavanext_video_model_path = args.llavanext_video_model_path
        llava_device = args.mllm_device
        model,processor = get_llavanext_video_model(llavanext_video_model_path,device=llava_device)
        tokenizer = None
    elif args.model_name == 'llavavideo':
        ln_video_model_path = args.llava_video_model_path
        ln_video_device = args.mllm_device
        model,processor,tokenizer = get_llava_video_model(ln_video_model_path,device=ln_video_device)
    elif args.model_name == 'qwen':
        qwen_path = args.qwen_model_path
        qwen_device = args.mllm_device
        model,processor =get_qwen_model(qwen_path,qwen_device)
    elif args.model_name == 'internvl':
        internvl_model_path = args.internvl_model_path
        internvl_model_device = args.mllm_device
        model,tokenizer = get_internVL_model(internvl_model_path,device=internvl_model_device)
        processor = None
    final_output = []
    # Helper function to process the accumulated batch and flush results
    def handle_batch(batch_data, processed_idx,tokenizer,processor):
        
        """Run inference on one accumulated batch, update metrics & save."""
        nonlocal final_output
        if not batch_data:
            return
        # batch_llm_inputs, prompts_text = preprocess_batch(batch_data, processor, model_config, video_root, video_nframes)
        # batch_output_text = inference_batch(batch_llm_inputs, model, processor, model_config)
        tokenizer  = tokenizer if tokenizer is not None else None
        processor = processor if processor is not None else None
        batch_output_text =inference_mllm(batch_data,model, processor,tokenizer)
        prompts_text = [''] * len(batch_data)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # --- calculate metrics ---
        current_metrics = calculate_metrics(final_output)
        save_results(output_path, final_output, current_metrics)
        processed_count = len(final_output)
        print(
            f"Processed up to overall index {processed_idx}, saved {processed_count} samples."
        )

    current_batch = []
    current_resolution = None  # (w,h) of first video in the current batch

    for idx, item in enumerate(tqdm(vsi_data[start:end], desc="Processing vsibench batches"),start=start):
        video_path = os.path.normpath(os.path.join(video_root, item["path"]))
        video_res,raw_fps = get_resolution(video_path)
        print('Resolution: ',video_res)
        frames_list = sample_video_frames_w_fps(video_path,fps=1/4)
        # uniform_idx = uniform_sampling(frames_list, 8)
        # key_frames = [frames_list[i] for i in uniform_idx]
        # frames_list = sample_video_frames(video_path,fps=1,max_num_frames=1e6)
        
        if len(frames_list)<video_nframes:
            key_frames = frames_list
        elif args.sample_strategy == 'uniform':
            print("Using random sampling for frame selection...")
            total_views = len(frames_list)
            if total_views <= args.nframes:
                return torch.arange(total_views).tolist()
            interval = total_views / args.nframes
            selected_indices = [int(i * interval) for i in range(args.nframes)]

            key_frames = [frames_list[idx] for idx in selected_indices]
        else:
            aks_index = aks_index_storage[idx][str(idx)]
            key_frames = [frames_list[i] for i in aks_index]
            print(f'Video {video_path} key frames indices: {aks_index}  Num frames:{len(key_frames)}')

            
        item['key_frames'] = key_frames
        if not current_batch:
            current_resolution = video_res
        
        # Check if resolution changes OR batch size limit reached
        if video_res != current_resolution or len(current_batch) >= batch_size:
            # Flush the current batch before adding this item
            handle_batch(current_batch, idx - 1,tokenizer,processor)
            current_batch = []
            current_resolution = video_res

        current_batch.append(item)

            # In case the batch exactly reaches BATCH_SIZE after appending, flush now
        if len(current_batch) >= batch_size:
            handle_batch(current_batch, idx,tokenizer,processor)
            current_batch = []
            current_resolution = None
        
    # Flush remaining samples after the loop ends
    if current_batch:
        handle_batch(current_batch, len(vsi_data) - 1)

    return final_output

def main(args):
    

    model_config = ModelConfig(
        model_path=args.model_path,
        model_type=args.model_type,
    )
    output_dir = os.path.join("/root/dws/3D_QA/Spatial-MLLM-master/eval_results/", f"eval_vsibench/{args.model_type}")
    os.makedirs(output_dir, exist_ok=True)
    vsi_data = load_vsi_evalset()
    n_gpu = torch.cuda.device_count()
    # n_gpu = len(gpu_name)
    ray.init(
        num_gpus=n_gpu,
        num_cpus=os.cpu_count(),
        _temp_dir="/root/dws/3D_QA/ray_temp",
        _system_config={"automatic_object_spilling_enabled": False,
                        "metrics_report_interval_ms": 0, },  # 禁止磁盘spill到/tmp
    )
    features = []
    per_gpu_data_length = len(vsi_data) // n_gpu
    for i in range(n_gpu): 
        start = i * per_gpu_data_length
        end = (i + 1) * per_gpu_data_length if i != n_gpu - 1 else len(vsi_data)
        axis = (start,end)
        output_path_gpu = os.path.join(output_dir, f"results_{model_config.model_type}_{i}.json")
        # local_config= deepcopy(model_config)                                                                
        # local_config.device= f"cuda:{gpu_name[i]}"
        features.append(evaluate_vsibench_r.remote(
            axis,
            vsi_data, 
            model_config=model_config, 
            output_path=output_path_gpu,
            video_root=args.video_root,
            video_nframes=args.nframes,
            batch_size=args.batch_size,
        ))
        # features.append(evaluate_vsibench_r(
        #     axis,
        #     vsi_data, 
        #     model_config=model_config, 
        #     output_path=output_path_gpu,
        #     video_root=args.video_root,
        #     video_nframes=args.nframes,
        #     batch_size=args.batch_size,
        # ))
    ret = ray.get(features)
    final_output = []
    for item in ret:
        final_output.extend(item)
        
    #--- calculate final metrics ---
    final_acc_dict = calculate_metrics(final_output)
    save_results(os.path.join(output_dir, f"results_{model_config.model_type}.json"), final_output, final_acc_dict)
    print(f"Finished evaluation for vsibench.")
    print(f"Final Metrics: {final_acc_dict}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on VSIBench dataset.")
    parser.add_argument("--model_name", type=str, default="internvl", choices=['llava', 'qwen','llavavideo','llavanext','internvl'])
    parser.add_argument("--model_path", type=str,  default='/root/dws/3D_QA/VG-LLM-main/data/model/vgllm-qa-vggt-8b',help="Path to the model.")
    parser.add_argument("--video_root", type=str,  default='/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench',help="Root directory for video files.")
    parser.add_argument("--llava_model-path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--internvl_model_path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/InternVL3_8B")
    parser.add_argument("--qwen_model_path", type=str, default="/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--llava_video_model_path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--llavanext_video_model_path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/llava_next_video_7B")
    parser.add_argument("--mllm_device", type=str, default="cuda:0")
    parser.add_argument("--model_type", type=str, default="vgllm_4b_uni", help="Type of the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--nframes", type=int, default=2)
    parser.add_argument("--is_compress", action='store_true')
    parser.add_argument("--ratio", type=float, default=2)
    parser.add_argument('--is_divprune',action='store_true')
    parser.add_argument('--is_ot',action='store_true')
    parser.add_argument('--sample_strategy',type=str,default='space_aks',choices=['space_aks','aks','random','coselect','uniform'])
    args = parser.parse_args()

    
    
    if args.sample_strategy == 'space_aks':
        with open('/root/dws/3D_QA/FastVGGT-main/outputs/selected_frames/vsibench_selected_frames_rest_16_f1_4.json', 'r') as f:
            aks_index_storage = json.load(f)
    elif args.sample_strategy == 'aks':
        with open('/root/dws/3D_QA/Spatial-MLLM-master/scores_frames_storage/vsibench_selected_frames_aks_8_d3.json', 'r') as f:
            aks_index_storage = json.load(f)
    elif args.sample_strategy == 'coselect':
        with open('/root/dws/3D_QA/Spatial-MLLM-master/scores_frames_storage/VSIBench_scores_f1_4_8_coselect_B.json', 'r') as f:
            aks_index_storage = json.load(f)
    main(args)