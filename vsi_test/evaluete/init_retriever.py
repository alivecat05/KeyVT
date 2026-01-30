import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/evaluate')
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/src/models')
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/src/models')
from transformers import AutoProcessor,Blip2ForImageTextRetrieval
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import json
import gc
import time
from copy import deepcopy
import clip
# add workspace to sys.path
from decord import VideoReader,cpu
from ultralytics import YOLOWorld
import torch
import shutil
from object_utils import extract_labels_and_filter_objs_direct
from LongCLIP.model import longclip
def get_longclip_model(device='cuda'):
    model = longclip.load("/root/dws/3D_QA/Spatial-MLLM-master/LongCLIP/checkpoints/longclip-L.pt",device=device)
    # model.to(device)
    return model
def sample_video_frames_w_fps(path, fps=1) -> list:
    """Sample frames from video at specified fps using decord (corrected)."""
    vr = VideoReader(path, ctx=cpu(0), num_threads=16)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps
    if fps <= 0:
        raise ValueError("fps must be positive")

    sample_times = np.arange(0, duration, 1.0 / fps)
    sample_times = sample_times[sample_times < duration]  # 防止越界
    if len(sample_times) == 0:
        return []
    frame_indices = np.round(sample_times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    frames = vr.get_batch(frame_indices).asnumpy()  # shape: (N, H, W, C)

    frames_list = [
        Image.fromarray(frame).convert("RGB") 
        for frame in frames
    ]
    print(f"Sampled {len(frames_list)} frames from video {os.path.basename(path)} at {fps} fps.")

    return frames_list
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
def get_CLIP_model(device='cuda'):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    return model, preprocess
def uniform_sample_video_frames(frames_list, num_frames=16) -> list:
    """Uniformly sample a fixed number of frames from the provided frames list."""
    total_frames = len(frames_list)
    if total_frames == 0:
        return []
    if total_frames <= num_frames:
        return frames_list

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    print(f"Uniformly sampled frame indices: {indices}")
    sampled_frames = [frames_list[i] for i in indices]
    resoluted_frames = [i.resize((336,336)) for i in sampled_frames]
    return resoluted_frames
def key_frames_retrieval_clip(question, frames,topk,clip_model,clip_preprocess,device='cuda'):
    
    scores = []
    print(f"Processing {len(frames)} views for question: {question[:50]}...")

    text_inputs = clip.tokenize([question],truncate=True).to(device)

    for i, view in enumerate(tqdm(frames)):
        try:
            image = clip_preprocess(view).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                text_features = clip_model.encode_text(text_inputs)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

                # Compute similarity
                similarity = (image_features @ text_features.T).squeeze().item()
                scores.append(similarity)
                
        except Exception as e:
            print(f"Error processing view {i}: {e}")
            import random
            scores.append(random.random())
            continue
    
    print(f"Computed scores for {len(scores)} views")
    
    scores = torch.tensor(scores)
    topk_indices = torch.topk(scores, k=min(topk,len(scores))).indices.tolist()
    # topk_indices = sorted(topk_indices)
    print(f"Top-{topk} frame indices: {topk_indices}")
    selected_frames = [frames[i] for i in topk_indices]
    resoluted_frames = [i.resize((336,336)) for i in selected_frames]
    return resoluted_frames,scores.tolist(),topk_indices
def key_frames_retrieval_B(question, frames,topk,blip_model,blip_processor,**kwargs):
    is_sqa = kwargs.get('is_sqa', False)
    scores = []
    print(f"Processing {len(frames)} views for question: {question[:50]}...")

    for i, view in enumerate(tqdm(frames)):
        try:
            print(f'Image type: {type(view)}, size: {view.size}')
            inputs = blip_processor(images=view, text=question, return_tensors="pt")
            inputs = inputs.to(blip_model.device)
            with torch.no_grad():
                # text_embeddings = blip_model.text_embeds(**inputs)
                # image_embeddings = blip_model.image_embeds(**inputs)
                # Use the correct method for image-text matching
                outputs = blip_model(**inputs, use_image_text_matching_head=True)
                # Get matching score
                if hasattr(outputs, 'logits_per_image'):
                    logits = outputs.logits_per_image
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    # Take probability of positive match (index 1)
                    match_score = probs[0][1].item()
                else:
                    # If output format is different, use a default approach
                    match_score = 0.5
                
                scores.append(match_score)
                
        except Exception as e:
            print(f"Error processing view {i}: {e}")
            raise e
        
    print(f"Computed scores for {len(scores)} views")
    
    scores = torch.tensor(scores)
    topk_indices = torch.topk(scores, k=min(topk,len(scores))).indices.tolist()
    if is_sqa:
        topk_indices = sorted(topk_indices)
    # topk_indices = sorted(topk_indices)
    print(f"Top-{topk} frame indices: {topk_indices}")
    selected_frames = [frames[i] for i in topk_indices]

    # resoluted_frames = [i.resize((336,336)) for i in sorted(selected_frames)]
    return selected_frames, scores.tolist(),topk_indices
def frames_interpolation(_key_frames, target_num_frames,frames_indices,all_views):
    pass
def load_vsi_evalset():
    file_path = os.path.abspath(__file__)
    vsi_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", "eval_vsibench.json")
    with open(vsi_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def get_YOLO(device='cuda:6'):
    model = YOLOWorld("/root/dws/3D_QA/TStar/cdViews/model/yolov8x-worldv2.pt")
    model.to(device)
    return model
def count_and_visualize_detected_classes(image_source, save_path=None, show=True,yolo_model=None,yolo_device='cuda:7'):
    with torch.no_grad():
        results = yolo_model.predict(source=image_source, device=yolo_device)
        # 可视化检测框
        if show or save_path:
            # show()会弹窗显示，save()可保存图片
            if save_path:
                results[0].save(filename=save_path+'result.jpg')
            if show:
                results[0].show()
        score = results[0].boxes.conf.cpu().numpy()
        detected_classes = []
        if hasattr(results[0], 'names') and hasattr(results[0], 'boxes'):
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            detected_classes = [class_names[idx] for idx in class_indices]
        # print("Detected classes:", detected_classes)
        return len(detected_classes), detected_classes,score
def detector_2_keyframes(all_views,selected_objs,save_path,yolo_model,device,num_frames=16):
    key_frames = []
    scores = []
    frames_indices = []
    
    for i,view in enumerate(tqdm(all_views)):
        _, detected_classes, score = count_and_visualize_detected_classes(view, show=True, save_path=save_path, yolo_model=yolo_model,yolo_device=device)
        if len(detected_classes) > 0:
            if set(selected_objs)& set(detected_classes):
                    if max(score)>0.5-0.05:
                        key_frames.append(view)
                        scores.append(max(score))
                        frames_indices.append(i)
    print(f'Indices of selected key frames: {frames_indices}')
    print(f'Score of selected key frames: {scores}')
    mean_conf = np.mean(scores)
    print(f'Mean confidence of selected key frames: {mean_conf}')
    if len(key_frames) == 0:
        print("No key frames detected with the specified objects. Falling back to uniform sampling.")
        key_frames = uniform_sample_video_frames(all_views, num_frames=num_frames)
    else:
        _key_frames =[]
        selected_idx = []
        
        if len(key_frames) >num_frames:
            for i,(score,f_idx) in enumerate(zip(scores,frames_indices)):
                if score>=mean_conf:
                    _key_frames.append(key_frames[i])
                    selected_idx.append(f_idx)
            if len(_key_frames)>num_frames:
                key_frames = uniform_sample_video_frames(_key_frames, num_frames=num_frames)
            else:
                topk_scores_indices = torch.topk(torch.tensor(scores).cpu(), k=min(num_frames, len(scores))).indices.tolist()
                key_frames = [key_frames[i] for i in sorted(topk_scores_indices)]
                
        else:
            topk_scores_indices = torch.topk(torch.tensor(scores).cpu(), k=min(num_frames, len(scores))).indices.tolist()
            key_frames = [key_frames[i] for i in sorted(topk_scores_indices)]
        # elif len(_key_frames) <num_frames:
        #     # key_frames = frames_interpolation(_key_frames, target_num_frames=num_frames,frames_indices=selected_idx,all_views=all_views)
        #     key_frames = _key_frames
    del scores
    del frames_indices
    del all_views
    gc.collect()
    torch.cuda.empty_cache()
    return key_frames
def maximize_relative_and_minimize_uncertainty(item,frames, num_frames,vgg_model,yolo_model,device='cuda'   ):
    # processed_frames = load_and_preprocess_images(frames).to('cuda')
    key_objects = extract_labels_and_filter_objs_direct(item['problem'], item['original_question_type'])
    print(f"Extracted key objects: {key_objects}")
    yolo_model.set_classes(key_objects)
    key_frames = detector_2_keyframes(frames, key_objects, save_path="/root/dws/3D_QA/Spatial-MLLM-master/evaluate/temp_frames/", yolo_model=yolo_model,device=device,num_frames=num_frames)
    resoluted_frames = [i.resize((336,336)) for i in key_frames]
    return resoluted_frames
    # with torch.no_grad():
    #     processed_frames = load_and_preprocess_images(key_frames).to('cuda')
    #     processed_frames = processed_frames.unsqueeze(0).to('cuda')
    #     aggregated_tokens_list, ps_idx = vgg_model.aggregator(processed_frames)
    # depth_map, depth_conf = vgg_model.depth_head(aggregated_tokens_list, processed_frames, ps_idx)
    # # Predict Point Maps
    # point_map, point_conf = vgg_model.point_head(aggregated_tokens_list, processed_frames, ps_idx)
    
    

    
    
# vsi_data = load_vsi_evalset()

# data = vsi_data[5123]

# video_root = "/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench"
# visual = os.path.join(video_root, data['path'])
# if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
#     visual = os.path.join(video_root, visual.replace("./", ""))
#     vr = VideoReader(visual)
#     image_num = len(vr)
#     # sample max_num_frames frame indices from the video
#     if image_num < 32:
#         frame_indices = np.arange(image_num)
#     else:
#         frame_indices = np.linspace(0, image_num - 1, 32).astype(int)
#     # read the frames
#     frames = [vr[i].asnumpy() for i in frame_indices]
#     visual_content = []
#     dir = "/root/dws/3D_QA/Spatial-MLLM-master/evaluate/temp_frames/"
#     os.makedirs(dir, exist_ok=True)
#     for i,frame in enumerate(frames):
#         image = Image.fromarray(frame).convert("RGB").resize((384,384))
#         print(image.size)
#         visual_content.append({"type": "image", "image": image})
#         image.save(os.path.join(dir, f"selected_frame_{i}.png"))
# vgg_model=get_VGGT_model(device='cuda')
# yolo_model = get_YOLO(device='cuda:3')


# frames = sample_video_frames_w_fps(visual,  fps=1)
# os.makedirs(dir, exist_ok=True)
# for i, frame in enumerate(frames):
#     frame.save(os.path.join(dir, f"frame_{i}.png"))
# print(f'Promblem_id: {data["problem_id"]}')
# print(f'{data["problem"]}')
# selected_frames=maximize_relative_and_minimize_uncertainty(data,frames, num_frames=8,vgg_model=None,yolo_model=yolo_model)

# dir = "/root/dws/3D_QA/Spatial-MLLM-master/evaluate/temp_frames/selected_yolo_frames/"
# shutil.rmtree(dir)
# os.makedirs(dir, exist_ok=True)
# for i, frame in enumerate(selected_frames):
#     frame.save(os.path.join(dir, f"selected_frame_{i}.png"))

