import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import ray
import json
from decord import VideoReader,cpu
from PIL import Image
import cv2
import numpy as np
sys.path.insert(0,'/root/dws/3D_QA/TStar/cdViews/cdviews')
from qa_utils import load_and_update, get_scanqa, get_sqa, custom_collate_fn
# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    get_vgg_input_imgs,
    get_sorted_image_paths,
    load_images_rgb,
    infer_vggt_and_reconstruct,
)
    
dtype = torch.bfloat16

def compute_sample_fps(video_path: str) -> float:
    
    """Return FPS that yields ≤ max_frames evenly-spaced frames over the video."""
    max_frames = 500
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / orig_fps if orig_fps else total / 1.0
    cap.release()

    if total <= max_frames or duration == 0:
        return orig_fps
    
    return max_frames / duration


def sample_video_frames_skip(path, skip=4) -> list:
    fps = compute_sample_fps(path)
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
    indices = np.arange(0, len(frames_list))
    frames_list = [frames_list[i] for i in indices if i % skip == 0]
    return frames_list

def sample_video_frames_w_fps(path, fps=2) -> list:
    """Sample frames from video at specified fps using decord (corrected)."""
    path = os.path.join(args.video_root, path.replace("./", ""))
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
def get_fastvggt_model(ckpt_path,merging=0,merge_ratio=0.9,vis_attn_map=False):

    print(f"🔄 Loading model: {ckpt_path}")
    model = VGGT(
        merging=merging,
        merge_ratio=merge_ratio,
        vis_attn_map=vis_attn_map,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print(f"✅ Model loaded")
    
    return model

def load_vsi_evalset():
    # file_path = '/root/dws/3D_QA/Spatial-MLLM-master/evaluate'
    # vsi_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", "eval_vsibench.json")
    vsi_annotation_path ='/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/eval_vsibench.json'
    with open(vsi_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_stibench():
    import pandas as pd
    
    PARQUET_FILE = "/root/dws/3D_QA/STI-Bench-main/STI-Bench/qa.parquet"
    print(f"[Data] Reading {PARQUET_FILE}")
    df_parquet = pd.read_parquet(PARQUET_FILE)    # remove .head(10) for full run
    sti_data = df_parquet.to_dict(orient="records")
    for it in sti_data:
        it["file"] = it.get("Video", "unknown.mp4")
        if isinstance(it.get("Candidates"), str):
            try: it["Candidates"] = json.loads(it["Candidates"])
            except json.JSONDecodeError: it["Candidates"] = {}
    print(f"[Data] {len(sti_data)} records loaded.\n")
    return sti_data


def fastvggt_pose_prediction(model,image_paths,input_frame_num=1000,depth_conf_thresh=3.0):
    if type(image_paths[0])==str:
        color_dir = image_paths / "color"
        image_paths = get_sorted_image_paths(color_dir)
    if len(image_paths) == 0:
        print(f"❌ Error: No images found in {color_dir}")
        return

    print(f"🖼️  Found {len(image_paths)} images")
    poses_gt = None
    first_gt_pose = None
    available_pose_frame_ids = None
    c2ws = None
    # Simply take the first N frames
    num_frames = min(len(image_paths), input_frame_num)
    selected_frame_ids = list(range(num_frames))
    image_paths = image_paths[:num_frames]
    print(f"📋 Selected {len(image_paths)} frames for processing")
    print(f"🔄 Loading images...")
    if type(image_paths[0])==str:
        images = load_images_rgb(image_paths)
    elif type(image_paths[0])==Image.Image:
        images = [cv2.cvtColor(np.asarray(i),cv2.COLOR_RGB2BGR) for i in image_paths]
        image_paths = None
    if not images or len(images) < 3:
        print(f"❌ Error: Not enough valid images (need at least 3)")
        return

    frame_ids = selected_frame_ids
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

    # Update attention layer patch dimensions in the model
    model.update_patch_dimensions(patch_width, patch_height)
    print(f"🚀 Start inference and reconstruction...")
    (
        extrinsic_np,
        intrinsic_np,
        all_world_points,
        all_point_colors,
        all_cam_to_world_mat,
        inference_time_ms,
    ) = infer_vggt_and_reconstruct(
        model, vgg_input, dtype, depth_conf_thresh, image_paths
    )
    print(f"⏱️  Inference time: {inference_time_ms:.2f}ms")
    # if not all_cam_to_world_mat or not all_world_points:
    #     print(f"❌ Error: Failed to obtain valid camera poses or point clouds")
    #     return
    if not all_cam_to_world_mat:
        print(f"❌ Error: Failed to obtain valid camera poses")
        return
    return all_cam_to_world_mat

# @ray.remote(num_gpus=1)
def go(args,start,end,**kwargs):
    vsi_data = kwargs.get('vsi_data', None)
    sti_data = kwargs.get('sti_data', None)
    qa_data = None
    sys.path.insert(0,'/root/dws/3D_QA/TStar/cdViews/cdviews')
    from qa_utils import get_scanqa, get_sqa
    print("Test the view selector... ")
    test_mode = ['val'] if args.dataset == 'ScanQA' else ['test', ]

    model = get_fastvggt_model(args.ckpt_path)
    print("Model loaded.")
    output_data = []
    for mode in test_mode:

        print('evaluating with QA for {}'.format(mode))
        if args.dataset == 'ScanQA':
            qa_data = get_scanqa(args, mode=mode)
        elif args.dataset == 'SQA':
            qa_data = get_sqa(args, mode=mode)
        elif args.dataset == 'vsibench':
            vsi_data = vsi_data
        elif args.dataset == 'stibench':
            sti_data = sti_data

        if qa_data:
            for i,line in tqdm(enumerate(qa_data[start:end],start=start), total=len(qa_data[start:end])):
                scene_id = line['scene_id']
                scene_path = args.image_folder + '/' + scene_id
                scene_path = Path(scene_path)
                all_cam_to_world_mat = fastvggt_pose_prediction(model,scene_path)
    
                output_data.append({i:all_cam_to_world_mat})
        elif vsi_data:
            for i,line in tqdm(enumerate(vsi_data[start:end],start=start), total=len(vsi_data[start:end])):
                scene_id = line['path']
                video_root = args.video_root
                video_path = os.path.normpath(os.path.join(video_root, line["path"]))
                # frames_list = sample_video_frames_w_fps(video_path,fps=1/4)
                video_path = os.path.join(args.video_root, video_path.replace("./", ""))
                frames_list = sample_video_frames_skip(video_path,skip=4) 
                all_cam_to_world_mat = fastvggt_pose_prediction(model,frames_list)
    
                output_data.append({i:all_cam_to_world_mat})
        elif sti_data:

            for i,entry in tqdm(enumerate(sti_data[start:end],start=start), total=len(sti_data[start:end])):
                vid_name, sample_id = entry["file"], entry.get("ID")
                VIDEO_DIR    = "/root/dws/3D_QA/STI-Bench-main/STI-Bench/video"
                print(f"[Run ] {i}/{len(sti_data)}  ({vid_name}, ID={sample_id})")
                video_path = os.path.join(VIDEO_DIR, vid_name)
                if not os.path.exists(video_path):
                    print(f"       ! video not found → {video_path}")
                    continue

                frames_list = sample_video_frames_skip(video_path,skip=4) 
                all_cam_to_world_mat = fastvggt_pose_prediction(model,frames_list)
    
                output_data.append({i:all_cam_to_world_mat})
    return output_data



def go_single(args,start,end,qa_data):
    

    for i,line in enumerate(qa_data[start:end],start=start):
            scene_id = line['scene_id']
            scene_path = args.image_folder + '/' + scene_id
            scene_path = Path(scene_path)
            all_cam_to_world_mat = fastvggt_pose_prediction(model,scene_path)
            return all_cam_to_world_mat
    
def parser_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='vsibench',
        choices=['ScanQA','SQA','vsibench','stibench'],
    )        
    parser.add_argument(
        "--cfg_file", 
        type=str, 
        default="/root/dws/3D_QA/TStar/cdViews/cfgs/QA.yaml")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/dws/3D_QA/FastVGGT-main/model_tracker_fixed_e20.pt",
    )
    parser.add_argument(
        "--video_root", 
        type=str,  
        default='/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench',
        help="Root directory for video files.")
    args = parser.parse_args()
    args = load_and_update(args)
    return args
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj  


if __name__=='__main__':

    args = parser_func()
    model = get_fastvggt_model(args.ckpt_path)
    args = load_and_update(args)
    ray.shutdown()
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
    if args.dataset == 'ScanQA':
        per_gpu_data_length = 4675 // n_gpu
        edge_case = 4675
    elif args.dataset == 'SQA':
        per_gpu_data_length = 3519 // n_gpu
        edge_case = 3519
    elif args.dataset == 'vsibench':
        vsi_data = load_vsi_evalset()
        per_gpu_data_length = len(vsi_data) // n_gpu
        edge_case = len(vsi_data)
    elif args.dataset == 'stibench':
        sti_data = load_stibench()
        per_gpu_data_length = len(sti_data) // n_gpu
        edge_case = len(sti_data)
    for i in range(n_gpu): 
        start = i * per_gpu_data_length
        end = (i + 1) * per_gpu_data_length if i != n_gpu - 1 else edge_case
        # features.append(go.remote(args,start,end, 
        #                           vsi_data=vsi_data if args.dataset == 'vsibench' else None,
        #                           sti_data=sti_data if args.dataset == 'stibench' else None
        #                           )
        #                 )
        features.append(go(args,start,end, 
                                  vsi_data=vsi_data if args.dataset == 'vsibench' else None,
                                  sti_data=sti_data if args.dataset == 'stibench' else None
                                  )
                        )
    
    results = ray.get(features)
    all_pred_data = []
    for pred_data in results:
        all_pred_data.extend(pred_data)
    all_pred_data=convert_numpy_types(all_pred_data)
    with open(f'/root/dws/3D_QA/FastVGGT-main/outputs/{args.dataset}_pose_f1p4.json','w') as f:
        json.dump(all_pred_data, f, indent=4)

