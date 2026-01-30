import json
import os
import numpy as np
import torch
from cal_bias import get_camera_centers,pose_distance
with open('./3D_QA/Spatial-MLLM-master/scores_frames_storage/vital/Scanqa_scores.json','r') as f:
    scores_dict = json.load(f)
    
with open('./3D_QA/FastVGGT-main/outputs/segs/ScanQA_segs_1.json','r') as f:
    segs_dict = json.load(f)
output_path=  './3D_QA/FastVGGT-main/outputs/selected_frames/ScanQA_fuckthemall_16.json'

frames_budget = 16


print(f"Total scenes to process: {len(segs_dict)}")
print(f"Scores dict length: {len(scores_dict)}")

output_dict = []
allocations = []
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

for idx, item in enumerate(segs_dict):
    segs = item[str(idx)]
    score = scores_dict[idx][str(idx)]
    
    
    
    frames_budget_current = min(frames_budget, len(score))
    if len(score) == 0:
        output_dict.append({str(idx): []})
        continue

    
    normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)  # 防除零
    
    len_per_seg = [len(s) for s in segs]
    scores_per_segs = []
    for s in segs:
        seg_scores = [normalized_data[i] for i in s]
        scores_per_segs.append(seg_scores)
    
    weights = []
    for s_p_seg, l_p_seg in zip(scores_per_segs, len_per_seg):
        reward = np.max(s_p_seg) + np.mean(s_p_seg)
        weight = reward * np.sqrt(l_p_seg)
        weights.append(weight)
    
    weights = np.array(weights)
    
    # ===== 核心修正：余数再分配 =====
    if len(weights) == 0:
        allocated_frames = []
    else:
        # 计算理想分配
        ideal_allocation = frames_budget_current * (weights / np.sum(weights))
        
        # 向下取整
        allocated_frames = np.floor(ideal_allocation).astype(int)
        remaining = frames_budget_current - np.sum(allocated_frames)
        
        # 如果还有剩余，分配给小数部分最大的段
        if remaining > 0:
            remainders = ideal_allocation - allocated_frames
            # 获取余数最大的前 `remaining` 个索引
            top_indices = np.argsort(remainders)[-remaining:]
            allocated_frames[top_indices] += 1
        
        # 安全检查：确保非负
        allocated_frames = np.maximum(allocated_frames, 0)
        
        # 如果某段分配0帧但实际有帧，可考虑最小保障（可选）
        # allocated_frames = np.maximum(allocated_frames, 1)  # 但可能超 budget
    
    print(f"Scene {idx}: allocated frames per segment: {allocated_frames}, total: {np.sum(allocated_frames)}, Scene len: {len(score)},Parts: {len(segs)}")
    
    selected_indices = []
    for alc, seg in zip(allocated_frames, segs):
        if alc > 0:
            seg_scores = [normalized_data[i] for i in seg]
            # 防止 alc > len(seg)
            alc = min(alc, len(seg))
            sorted_indices = np.argsort(seg_scores)[-alc:]
            selected_indices.extend([seg[i] for i in sorted_indices])
    
    selected_indices = sorted(set(selected_indices))
    
    # ===== 最终保障：如果还是不足，补选最高分帧 =====
    if len(selected_indices) < frames_budget_current:
        remaining_needed = frames_budget_current - len(selected_indices)
        # 从未选帧中选最高分的
        all_indices = set(range(len(normalized_data)))
        unselected = list(all_indices - set(selected_indices))
        if unselected:
            unselected_scores = [normalized_data[i] for i in unselected]
            top_unselected = np.argsort(unselected_scores)[-remaining_needed:]
            selected_indices.extend([unselected[i] for i in top_unselected])
            selected_indices = sorted(set(selected_indices))
    
    output_dict.append({str(idx): selected_indices[:frames_budget_current]})  # 确保不超过
    allocations.append({str(idx): allocated_frames.tolist()})

end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"Total calculation time consumption: {elapsed_time:.2f} milliseconds")

#Total calculation time consumption: 1414.20 milliseconds

# with open(output_path,'w') as f:
#     json.dump(output_dict, f, indent=4)
# with open('./3D_QA/FastVGGT-main/outputs/selected_frames/ScanQA_allocations_bias_16.json','w') as f:
#     json.dump(allocations, f, indent=4)