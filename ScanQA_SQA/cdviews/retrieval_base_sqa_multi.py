import argparse
import sys
import torch
import os
import numpy as np
import json
import copy
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
from matplotlib import pyplot as plt
from qa_utils import load_and_update, get_scanqa, get_sqa, custom_collate_fn
from view_distance_calculation import *
from PIL import Image
import logging
from transformers import AutoProcessor,Blip2ForImageTextRetrieval,AutoModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import ray
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
logging.getLogger().handlers = []
logs_path = '/root/dws/3D_QA/TStar/logs/'
import re

def get_internVL_model(model_id,device):
    sys.path.insert(0,'/root/dws/3D_QA/TStar/cdViews/model')
    from InternVL3_8B.modeling_internvl_chat import InternVLChatModel
    from transformers import AutoModel, AutoTokenizer
    device = "cuda:7"
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
    generation_config = dict(max_new_tokens=10, do_sample=False)

    pixel_values = [
    load_image(f.resize((max(f.size), max(f.size)))).to(torch.bfloat16).to(model.device, non_blocking=True)
    for f in frames
    ]
    
    pixel_values = torch.cat(pixel_values, dim=0).to(model.device)

    question = '<image>\n'+query+'Answer simply'

    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=False,iscompress_on=args.is_compress,compress_ratio=args.ratio)
    
    return response



def get_llava_video_model(model_id,device):
    model_name= 'llava_qwen'
    pretrained = model_id
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device,attn_implementation=None)
    model.eval()
    return model, image_processor,tokenizer
def llava_video_predict(model, processor,tokenizer,query, frames=None,**kwargs):
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN*len(frames) + query +'Answer the question simply.'
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
def qwen_predict(model, processor,query, objects=None,frames=None,device='cuda',ground=False):
    
    system_prompt = f'{query} Answwer the question simply'
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
        text=[text_template],
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
    ans = output_text[0].strip()
    if ans=='television':
        ans='tv'
    return ans
def get_llava_model(model_id,device):
    model_name= 'llava_qwen'
    pretrained = model_id
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device,attn_implementation=None)
    model.eval()
    return model, image_processor,tokenizer
def preprocess_qwen(sources, tokenizer, has_image: bool = False, max_len=2048,
                    system_message: str = "You are a helpful assistant."):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    # if roles[source[0]["from"]] != roles["human"]:
    #     source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def llava_predict(model, processor,tokenizer,query, frames=None,**kwargs):
    line = kwargs.pop('line',[])
    
    line['from'] = 'human'
    question = query
    line['value'] = '<image>' * len(frames) +  question
    input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).to(
        model.device)
    image_inputs = processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, torch.half)
    
    cont = model.generate(
            input_ids,
            images=image_inputs,
            modalities= ["image"],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            is_compress=args.is_compress,
            ratio=args.ratio,
            is_divprune = args.is_divprune,
            is_ot = args.is_ot,
        )
    outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    outputs = outputs.strip().lower()
    ans = outputs.split('\n')[0]
    if ans=='television':
        ans='tv'
    if ans=='rectangle':
        ans = 'rectangular'
    return ans
def get_BLIP2_model(model_id,device):
    # Try using the official pretrained model instead of local checkpoint
    try:
        print(f"Attempting to load BLIP2 model from: {model_id}")
        model = Blip2ForImageTextRetrieval.from_pretrained(
            "/root/dws/3D_QA/TStar/cdViews/model/blip2_vit_g",  # Use official model
            dtype=torch.float16,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained("/root/dws/3D_QA/TStar/cdViews/model/blip2_vit_g")
        print("Successfully loaded official BLIP2 model")
        return model, processor
    except Exception as e:
        print(f"Failed to load official model: {e}")
        # Fallback to local model
        try:
            model = Blip2ForImageTextRetrieval.from_pretrained(
                model_id, 
                dtype=torch.float16,
                device_map=device
            )
            processor = AutoProcessor.from_pretrained(model_id)
            print("Loaded local BLIP2 model")
            return model, processor
        except Exception as e2:
            print(f"Failed to load local model: {e2}")
            # Return None to signal failure
            return None, None
def get_neighbour_df(scene_id):
    distance_df = calculate_view_distance(scene_id, args)
    return distance_df  
def viewNMS(scene_id,list_of_images, num_images, distance_threshold=0.5):
    neighbour_df=get_neighbour_df(scene_id)
    selected_images = []
    remaining_images = list_of_images.copy()

    while len(selected_images) < num_images and remaining_images:
        current_image = remaining_images.pop(0)
        selected_images.append(current_image)

        sorted_distances = neighbour_df.loc[current_image].sort_values()
        filtered_images = sorted_distances[sorted_distances < distance_threshold]
        neighbours_to_remove = set(filtered_images.index.tolist())
        remaining_images = [img for img in remaining_images if img not in neighbours_to_remove]

    return selected_images


def sqa_evaluate(ans,line):
    ans = ans.strip().lower().replace('.','').replace(',','')
    ground_truth = line['answers'][0]
    if ground_truth in ['one','two','three','four','five','six','seven','eight','nine','zero']:
        ground_truth = str(['zero','one','two','three','four','five','six','seven','eight','nine'].index(ground_truth))
    if ans.lower() in ['one','two','three','four','five','six','seven','eight','nine','zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Zero']:
        ans = str(['zero','one','two','three','four','five','six','seven','eight','nine'].index(ans.lower()))
    gt_lower = ground_truth.lower()
    ans_lower = ans.lower()
    is_correct = False
    if gt_lower in ans_lower or ans_lower in gt_lower:
        is_correct = True
    else:
        shorter, longer = (gt_lower, ans_lower) if len(gt_lower) < len(ans_lower) else (ans_lower, gt_lower)
        if len(shorter) > 4: # Only for words with some length
            for i in range(len(shorter), 4, -1):
                if shorter[:i] in longer:
                    is_correct = True
                    break
    return is_correct, ground_truth


def scanqa_evaluate_EM(pred_data,idx2labels):
    data = copy.deepcopy(pred_data)
    n_correct = 0
    res, gts = {}, {}
    for item in data:
        item["sample_id"] = item["sample_id"]+ '_0'
        res[item['sample_id']] = [item['pred_response'].rstrip(".")]
        gts[item['sample_id']] = idx2labels[item['sample_id']]

        if item['pred_response'] in idx2labels[item['sample_id']]:
            n_correct += 1

    is_correct = False
    if data[-1]['pred_response'] in idx2labels[data[-1]['sample_id']]:
        is_correct = True
    
    metrics = {
        "EM": n_correct / len(data)
        }
    return is_correct,metrics



@ray.remote(num_gpus=1)
def eval_model(args,start,end,idx):
    
    frames = None
    
    logs_path = '/root/dws/3D_QA/TStar/logs/'
    sys.path.insert(0,'/root/dws/3D_QA/Spatial-MLLM-master/evaluate')
    from gaussian import Get_frames_prior_multiview, key_frames_retrieval_B
    from coselect import coselect
    sys.path.insert(0,'/root/dws/3D_QA/TStar/Video_3D/llava/eval')
    from caption_eval.bleu.bleu import Bleu
    from caption_eval.rouge.rouge import Rouge
    from caption_eval.meteor.meteor import Meteor
    from caption_eval.cider.cider import Cider
    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()



    if args.geometry:
        from floc import data_param,geo_param
        geo_param.geometry_data = geometry_data
        geo_param.dataset = 'SQA'
        
        
        
        
    eval_tools = (cider, bleu, meteor, rouge)
    # Model
    if args.model_name == 'llava':
        llava_model_path = args.llava_model_path
        llava_device = args.mllm_device
        model,processor,tokenizer = get_llava_model(llava_model_path,device=llava_device)
    elif args.model_name == 'qwen':
        Qwen_model_path = args.qwen_model_path
        qwen_device = args.mllm_device
        model,processor = get_qwen_model(Qwen_model_path,device=qwen_device)    
    elif args.model_name == 'llavavideo':
        ln_video_model_path = args.llava_video_model_path
        ln_video_device = args.mllm_device
        model,processor,tokenizer = get_llava_video_model(ln_video_model_path,device=ln_video_device)

    blip_model_path = args.blip_model_path
    blip_device =args.blip_device
    blip_model,blip_processor=get_BLIP2_model(blip_model_path,device=blip_device)

    
    if args.dataset == 'ScanQA':
        with open("/root/dws/3D_QA/TStar/cdViews/data/qa/ScanQA/scanqa_val_llava_style.json") as f:
            raw_data = json.load(f)
            idx2labels = {}
            for item in raw_data:
                idx2labels[item['id']] = item['metadata']['answers']
    # Data
    print("Test the view selector... ")
    test_mode = ['val'] if args.dataset == 'ScanQA' else ['test', ]


    for mode in test_mode:
        
        print('evaluating with QA for {}'.format(mode))
        if args.dataset == 'ScanQA':
            qa_data = get_scanqa(args, mode=mode)
        elif args.dataset == 'SQA':
            qa_data = get_sqa(args, mode=mode)
            
        logs_path = logs_path+f'{args.dataset}_blip2_9_{args.method_type}/'
        os.makedirs(logs_path, exist_ok=True)
        
        haha = f'{args.dataset}_{mode}_rtb_{args.method_type}_gpu{idx}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            filename= logs_path+haha,
            filemode='w'
        )
        logger = logging.getLogger(__name__)

        accuracy_history = []
        
        
        pred_data = []
        for i,line in tqdm(enumerate(qa_data[start:end],start=start), total=len(qa_data[start:end])):
            torch.cuda.empty_cache()
            scene_id = line['scene_id']
            if args.geometry:
                geo_param.scene_id = line['scene_id']
                
                geo_param.data_id = i            
            
            

            scene_path = args.image_folder + '/' + scene_id + '/color'
            question_id = line['question_id']
            # caption = caption_dict[str(question_id)][0]
            question = line['situation']+'\n'+line['question'] if args.dataset == 'SQA' else line['question']
            question_id = str(line['question_id'])
            # direction = extract_pose_each_view_with_yaw(scene_id, args)
            all_views_path = sorted(os.listdir(scene_path), key=lambda x: int(x.split('.')[0]))

            all_views = [Image.open(scene_path+'/'+view).convert('RGB') for view in all_views_path]
            
            vis_output_dir = f'/root/dws/3D_QA/Spatial-MLLM-master/eval_results/{args.dataset}_blip2_9_{args.method_type}/'
            os.makedirs(vis_output_dir, exist_ok=True)
            vis_output_dir_dir = os.path.join(vis_output_dir, f'{i}_{scene_id}/')
            os.makedirs(vis_output_dir_dir, exist_ok=True)
            
            # cos_idx = cos_indices[i][str(i)] if args.sample_strategy == 'coselect' else []
            # print(f"Using co selected frame indices: {aks_idx}")
            if args.pure_retrieval:
                if args.sample_strategy == 'aks':
                    aks_idx = aks_indices[i][str(i)] if args.sample_strategy == 'aks' else []
                    frames = [all_views[t] for t in aks_idx] 
                    print(f"Selected frames using AKS strategy with indices: {aks_idx})")
                    if len(frames) > args.num_frames:
                        frames,score,topk_indices_ = key_frames_retrieval_B(line['question'],frames,args.num_frames,blip_model,blip_processor,is_sqa = True)
                elif args.sample_strategy == 'space_aks':
                    space_aks_idx = space_aks_indices[i][str(i)] if args.sample_strategy == 'space_aks' else []
                    frames = [all_views[t] for t in space_aks_idx] 
                    # frames,score_storage_,topk_indices_ = key_frames_retrieval_B(question,frames,args.num_frames,blip_model,blip_processor,is_sqa = True)
                    if args.geometry:
                        geo_param.key_frames_idx = topk_indices_
                    # print(f"Using Space-AKS selected frame indices: {topk_indices_}")
                elif args.sample_strategy == 'coselect':
                    coselect_idx = coselect_indices[i][str(i)] if args.sample_strategy == 'coselect' else []
                    print(f"Using Co-Select selected frame indices: {coselect_idx}")
                    frames = [all_views[t] for t in coselect_idx]
                elif args.sample_strategy == 'random':
                    print("Using random sampling for frame selection...")
                    total_views = len(all_views)
                    if total_views <= args.num_frames:
                        frames = all_views
                    else:
                        random_indices = np.random.choice(total_views, args.num_frames, replace=False)
                        random_indices.sort()
                        frames = [all_views[idx] for idx in random_indices]

            num_frames = len(frames)
            if args.model_name == 'llava':
                ans = llava_predict(model, processor, tokenizer,question, frames=frames,line = line)
            elif args.model_name == 'qwen':
                ans = qwen_predict(model, processor, question, frames=frames)
            elif args.model_name == 'llavavideo':
                ans = llava_video_predict(model, processor, tokenizer,question, frames=frames)
            print(f'Answer: {ans}')

            # for j, f in enumerate(frames):
            #     f.save(os.path.join(vis_output_dir_dir, f'_{j}.png'))
            if num_frames > 0:
                fig, axes = plt.subplots(4, 8, figsize=(12, 12))
                fig.suptitle(f'Selected Frames for Scene: {scene_id}\nQuestion: {question[:50]}...', fontsize=14)
                axes = axes.flatten()
                
                for idx, frame in enumerate(frames):
                    if idx < 32:  # Display up to 9 frames
                        axes[idx].imshow(frame)
                        axes[idx].axis('off')
                
                # Hide unused subplots
                for idx in range(num_frames, 32):
                    axes[idx].axis('off')
                
                # Create output directory if it doesn't exist
                vis_output_dir = vis_output_dir_dir
                os.makedirs(vis_output_dir, exist_ok=True)
                # Save visualization
                vis_filename = f'key_frames_{i}.png'
                plt.tight_layout()
                plt.savefig(os.path.join(vis_output_dir, vis_filename), dpi=150, bbox_inches='tight')
                plt.close()

            print(ans)
            logger.info(f"{i}|{len(qa_data)} Scene: {scene_id}")
            logger.info(f"Question ID: {question_id}")
            if args.dataset == 'SQA':
                logger.info(f"Situation: {line['situation']}")
            logger.info(f"Question: {line['question']}")
            logger.info(f'Answer: {ans}')
            if args.dataset == 'ScanQA':
                is_correct,EM = scanqa_evaluate_EM(pred_data,idx2labels)
            elif args.dataset == 'SQA':
                is_correct,ground_truth = sqa_evaluate(ans,line)
            if is_correct:
                logger.info(f"Is_correct :Correct! The ground truth is {line['answers']}")
                if args.dataset == 'ScanQA':
                    logger.info(f'Current metrics: {EM}')
                elif args.dataset == 'SQA':
                    
                    accuracy_history.append(1)
                    acc = sum(accuracy_history) / len(accuracy_history)
                    logger.info(f'Current accuracy: {acc:.4f}')
            else:
                logger.info(f"Is_correct :Wrong! The ground truth is {line['answers']}")
                if args.dataset == 'ScanQA':
                    logger.info(f'Current Metrics: {EM}')
                elif args.dataset == 'SQA':
                    accuracy_history.append(0)
                    acc = sum(accuracy_history) / len(accuracy_history)
                    logger.info(f'Current accuracy: {acc:.4f}')
            logger.info(f'--------------------------------------')
                    
    return accuracy_history    
                    

 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llavavideo", choices=['llava', 'qwen','llavavideo','llavanext'])
    parser.add_argument("--llava_model-path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--qwen_model-path", type=str, default="/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--llava_video_model_path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--blip_model-path", type=str, default="/root/dws/3D_QA/TStar/cdViews/model/blip2-opt-2_7b")   
    parser.add_argument("--cfg_file", type=str, default="/root/dws/3D_QA/TStar/cdViews/cfgs/QA.yaml")
    parser.add_argument("--method_type", type=str, default="gaussian_2")
    parser.add_argument("--mllm_device", type=str, default="cuda:0")
    parser.add_argument("--blip_device", type=str, default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument('--ratio',type=float,default=2)
    parser.add_argument('--is_compress',action='store_true')
    parser.add_argument("--sample_strategy", type=str, default="space_aks",choices=['symmetric','aks','space_aks','coselect','random'])
    parser.add_argument('--pure_retrieval', action='store_true')
    parser.add_argument("--dataset", type=str, default="ScanQA",choices=['ScanQA','SQA'])
    parser.add_argument('--geometry',action='store_true')
    parser.add_argument('--is_divprune',action='store_true')
    parser.add_argument('--is_ot',action='store_true')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    args = load_and_update(args)
    
    
    
    
    if args.dataset == 'SQA' and args.geometry:
        from datasets import load_from_disk
        geometry_data  = load_from_disk('/root/dws/3D_QA/Depth-Anything_3/outputs/SQA_pose_da3_hf')
        geometry_data = geometry_data.with_format("numpy")
    if args.sample_strategy == 'aks':
        with open(f'/root/dws/3D_QA/Spatial-MLLM-master/scores_frames_storage/{args.dataset}_selected_frames_8_d3.json',"r", encoding="utf-8") as f:
            aks_indices= json.load(f)
    elif args.sample_strategy == 'space_aks':
        # with open(f'/root/dws/3D_QA/FastVGGT-main/outputs/selected_frames/{args.dataset}_selected_frames_rest_16.json',"r", encoding="utf-8") as f:
        with open(f'/root/dws/3D_QA/FastVGGT-main/outputs/selected_frames/SQA_real_extr_16.json',"r", encoding="utf-8") as f:
            space_aks_indices= json.load(f)
    elif args.sample_strategy == 'coselect':
        with open(f'/root/dws/3D_QA/Spatial-MLLM-master/scores_frames_storage/{args.dataset}_selected_frames_coselect{args.num_frames}.json',"r", encoding="utf-8") as f:
            coselect_indices= json.load(f)
    
    
    
    n_gpu = torch.cuda.device_count()
    ray.init(
        num_gpus=n_gpu,
        num_cpus=os.cpu_count(),
        _temp_dir="/root/dws/3D_QA/ray_temp",
        _system_config={"automatic_object_spilling_enabled": False,
                        "metrics_report_interval_ms": 0, },  # 禁止磁盘spill到/tmp

    )
    n_gpu = torch.cuda.device_count()
    features = []
    per_gpu_data_length = 3519 // n_gpu
    print(f"Total data length: 3519, Per GPU data length: {per_gpu_data_length}, Number of GPUs: {n_gpu}")
    for i in range(n_gpu): 
        start = i * per_gpu_data_length
        end = (i + 1) * per_gpu_data_length if i != n_gpu - 1 else 3519
        features.append(eval_model.remote(args,start,end,i))

    results = ray.get(features)
    all_pred_data = []
    for pred_data in results:
        all_pred_data.extend(pred_data)
        
    overall_accuracy = sum([sum(chunk) for chunk in results])
    print(f"Overall accuracy across all GPUs: {overall_accuracy / 3519:.4f}")
    with open(logs_path+f'overall_accuracy_{args.dataset}_blip2_9_{args.method_type}.txt','w') as f:
        f.write(f'Overall accuracy: {overall_accuracy / 3519:.4f}\n')
        