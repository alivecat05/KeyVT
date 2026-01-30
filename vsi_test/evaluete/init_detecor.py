import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/evaluate')
sys.path.insert(0, '/root/dws/3D_QA/Spatial-MLLM-master/src/models')
from ultralytics import YOLOWorld
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
from transformers import AutoModel, AutoProcessor
from decord import VideoReader,cpu
from object_utils import extract_labels_and_filter_objs_direct
import json
import ray
video_root = '/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench'
import logging
import gc
logging.getLogger().handlers = []
logs_path = '/root/dws/3D_QA/Spatial-MLLM-master/logs/'
direction_dict = {
    'object_rel_direction': [],
    'object_rel_distance':[],
    'obj_appearance_order':[],
    'object_counting':None,
    'object_size_estimation':[],
    'object_abs_distance':[],
    'route_planning':[],
}
def sample_video_frames_w_fps(path, fps=2) -> list:
    """Sample frames from video at specified fps using decord (corrected)."""
    path = os.path.join(video_root, path.replace("./", ""))
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
    return frames_list

def load_vsi_evalset():
    file_path = os.path.abspath(__file__)
    vsi_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", "eval_vsibench.json")
    with open(vsi_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_YOLO(device='cuda:6'):
    print("Loading YOLO model...")
    def get_model_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            # 模型没有参数（极少见）
            return torch.device('cpu')
    model = YOLOWorld("/root/dws/3D_QA/Spatial-MLLM-master/src/models/yolov8l-worldv2.pt")
    model.to(device)
    device = get_model_device(model)
    print(f"Model loaded on device: {device}")
    return model,device

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
        print("Detected classes:", detected_classes)
        return len(detected_classes), detected_classes,score

def detector_2_keyframes(all_views,selected_objs,save_path,yolo_model,device):
    os.makedirs(save_path, exist_ok=True)
    key_frames = []
    scores = []
    frames_idx = []
    for i,view in enumerate(tqdm(all_views)):
        _, detected_classes, score = count_and_visualize_detected_classes(view, show=True, save_path=save_path, yolo_model=yolo_model,yolo_device=device)
        if len(detected_classes) > 0:
            if set(selected_objs)& set(detected_classes):
                    key_frames.append(view)
                    frames_idx.append(i)
        try:
            scores.append(score[0].item())
        except:
            scores.append(0.0)
    return key_frames,torch.tensor(scores),frames_idx

def get_directional_priors(question,question_type,direction_dict):
    for direction_type in direction_dict.keys():
        if direction_type in question_type:
            direction_dict[direction_type].append(question)
    return direction_dict


def maximize_relative_and_minimize_uncertainty(item,frames, num_frames,vgg_model,yolo_model,device='cuda'   ):
    # processed_frames = load_and_preprocess_images(frames).to('cuda')
    key_objects = extract_labels_and_filter_objs_direct(item['problem'], item['original_question_type'])
    yolo_model.set_classes(key_objects)
    key_frames,scores,frames_idx = detector_2_keyframes(frames, key_objects, save_path="/root/dws/3D_QA/Spatial-MLLM-master/evaluate/temp_frames/", yolo_model=yolo_model,device=device)
    resoluted_frames = [i.resize((336,336)) for i in key_frames]
    return resoluted_frames,scores,frames_idx

def Gaussian_diffusion(prob_dist, key_frames_idx,sigma=2.0,tau=1.0):
    def gaussian(x, mu=0, sigma=1, peak_height=None):
        base = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if peak_height is None:
            # 标准概率密度形式
            return base / (sigma * np.sqrt(2 * np.pi))
        else:
            # 直接缩放，使最大值 = peak_height
            return base*peak_height * 2
    diffuse_dist = prob_dist.clone()
    length = len(prob_dist)
    x = torch.arange(length)
    for i,idx in enumerate(key_frames_idx):
        mu = idx
        peak_height = prob_dist[idx]
        gauss = gaussian(x.numpy(), mu=mu, sigma=sigma, peak_height=peak_height.item())
        gauss_tensor = torch.tensor(gauss, device=prob_dist.device)
        visualize_probability_distribution(gauss_tensor, save_path=f'/root/dws/3D_QA/Spatial-MLLM-master/eval_results/gaussian_kernal_{i}_idx_{idx}_only.png', highlight_indices=[idx])
        diffuse_dist += gauss_tensor
        visualize_probability_distribution(gauss_tensor, save_path=f'/root/dws/3D_QA/Spatial-MLLM-master/eval_results/gaussian_frame_{i}_idx_{idx}.png', highlight_indices=None)
    lowest_value = torch.min(diffuse_dist)
    diffuse_dist -= lowest_value
    return torch.softmax(diffuse_dist/tau, dim=0)
    
def get_siglip_model(device='cuda:6'):
    print("Loading SigLIP model...")
    model_name = "/root/dws/3D_QA/Spatial-MLLM-master/src/models/siglip_so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"SigLIP Model loaded on device: {device}")
    return model, processor

def siglip_retrieval(question,frames, processor, model, device='cuda:6', topk=5):
    # if not isinstance(frames, List[Image.Image]):
    #     _frames = [Image.open(i).convert('RGB') for i in frames]
    #     frames = _frames
    inputs = processor(text=question, images=frames, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).cpu().T
    
    topk_indices = torch.topk(probs, k=topk, dim=1).indices.squeeze(0).tolist()
    selected_frames = [frames[i] for i in sorted(topk_indices)]
    return selected_frames, probs.squeeze(0).tolist(),sorted(topk_indices)
    
    
def visualize_probability_distribution(prob_dist, save_path, highlight_indices=None):
    """
    Plot a probability distribution and mark specific indices with orange dots.

    Args:
        prob_dist: torch.Tensor or numpy-like 1D array of probabilities.
        save_path: path to save the resulting plot image.
        highlight_indices: iterable of integer indices to highlight (e.g. [2,5,7]).
            If None, no special indices are highlighted.
    """
    import matplotlib.pyplot as plt
    import numpy as _np

    # Convert to numpy array if tensor
    if hasattr(prob_dist, "cpu"):
        probs = prob_dist.cpu().numpy()
    else:
        probs = _np.array(prob_dist)

    plt.figure(figsize=(10, 5))
    x = _np.arange(len(probs))
    plt.plot(x, probs, marker='o', label='probability')

    # If highlight indices provided, validate and mark them
    if highlight_indices is not None:
        # normalize input to numpy array of ints
        try:
            idxs = _np.array(list(highlight_indices), dtype=int)
        except Exception:
            # fallback: single int
            idxs = _np.array([int(highlight_indices)])

        # clip and filter valid indices
        idxs = _np.unique(idxs)
        idxs = idxs[(idxs >= 0) & (idxs < len(probs))]

        if idxs.size > 0:
            vals = probs[idxs]
            plt.scatter(idxs, vals, color='orange', zorder=5, s=80, label='highlight')
            for xi, yi in zip(idxs, vals):
                plt.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', color='orange', fontsize=8)

    plt.title('Frame Selection Probability Distribution')
    plt.xlabel('Frame Index')
    plt.ylabel('Probability')
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def Get_frames_prior(model,device,item,num_frames=8):
    siglip_model, processor = model
    video = os.path.join(video_root,item['path'])
    frame_list = sample_video_frames_w_fps(video, fps=1)
    # key_objects = extract_labels_and_filter_objs_direct(item['problem'], item['original_question_type'])
    # yolo_model.set_classes(key_objects)
    # key_frames,all_f_scores,key_frames_idx = detector_2_keyframes(frame_list, key_objects, save_path="/root/dws/3D_QA/Spatial-MLLM-master/evaluate/temp_frames/", yolo_model=yolo_model,device=device)
    anch_frames,all_f_scores,anch_frames_idx = siglip_retrieval(item['problem'], frame_list, processor, siglip_model, device=device, topk=num_frames)

    all_frames_idx = [i for i in range(len(frame_list))]
    # probability_distribution = torch.softmax(torch.tensor(all_f_scores), dim=0)
    scores_distrubution = torch.tensor(all_f_scores)*10.0
    visualize_probability_distribution(scores_distrubution, save_path='/root/dws/3D_QA/Spatial-MLLM-master/eval_results/frame_selection_prob_dist.png', highlight_indices=anch_frames_idx)
    diffuse_dist = Gaussian_diffusion(scores_distrubution, anch_frames_idx,sigma=3.0,tau=1)
    sampled_idx = np.random.choice(all_frames_idx, size=min(num_frames, len(frame_list)), replace=False, p=diffuse_dist.cpu().numpy())
    visualize_probability_distribution(diffuse_dist, save_path='/root/dws/3D_QA/Spatial-MLLM-master/eval_results/frame_selection_diffused_prob_dist.png',highlight_indices=sampled_idx)
    
    key_frames = [frame_list[i] for i in sorted(sampled_idx)]
    gc.collect()
    torch.cuda.empty_cache()
    return key_frames
    
    
    
# if __name__ == "__main__":
    
#     data_vsi = load_vsi_evalset()
#     item = data_vsi[0]
#     get_siglip_model_device = 'cuda:0'
#     siglip_model = get_siglip_model(device=get_siglip_model_device)
#     key_frames = Get_frames_prior(siglip_model, get_siglip_model_device, item, num_frames=8)

 