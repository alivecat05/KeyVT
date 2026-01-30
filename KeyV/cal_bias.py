import numpy as np
import json
import torch
def compute_directional_bias(Rs, ts):
    N = len(Rs)
    sigma = np.zeros(N)
    
    for i in range(N):
        d_prev = d_next = 0.0
        
        if i > 0:
            d_prev = pose_distance(Rs[i], ts[i], Rs[i-1], ts[i-1])
        else:
            d_prev = np.nan  # or 0, but we'll handle specially
        
        if i < N - 1:
            d_next = pose_distance(Rs[i], ts[i], Rs[i+1], ts[i+1])
        else:
            d_next = np.nan

        # Handle boundaries
        if i == 0:
            sigma[i] = 0  # only next exists → negative means "pull toward next"
        elif i == N - 1:
            sigma[i] = d_prev   # only prev exists
        else:
            sigma[i] = d_prev - d_next  # your definition: d1 - d2
    
    return np.clip(sigma,-0.3,0.3)
def rotation_angle(R1, R2):
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    cos_theta = (trace - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta
def pose_distance(R1, t1, R2, t2, alpha=1.0, beta=1.0):

    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    trans_dist = np.linalg.norm(C1 - C2)
    rot_angle = rotation_angle(R1, R2)
    return alpha * trans_dist + beta * rot_angle
def load_param(file_path):
    with open(file_path, 'r') as f:
        extrinsics_dict = json.load(f)
    extrinsics_list=  []
    for idx,item in enumerate(extrinsics_dict):
        frames = []
        extrinsics = item[str(idx)]
        for i, line in enumerate(extrinsics):
            frames.append(np.array(line, dtype=np.float64))

        extrinsics_list.append(frames)
    #     print(f"Loaded {len(frames)} frames.")
    # print(f"Loaded {len(extrinsics_list)} scenes.")
    return extrinsics_list
def get_camera_centers(frames):
    camera_centers = []
    Rs = []
    ts = []

    for i, T in enumerate(frames):
        R = T[:3, :3]
        t = T[:3, 3]
        C = -R.T @ t
        camera_centers.append(C)
        Rs.append(R)
        ts.append(t)

    # print(f"Frame {i:2d} camera center: [{C[0]: .6f}, {C[1]: .6f}, {C[2]: .6f}]")

    # C0 = camera_centers[0]
    # cum_displacements = [np.linalg.norm(C - C0) for C in camera_centers]# L2范数    
    
    return Rs, ts
def merge_isolated_segments(seg_result):
    """
    Merge isolated (length == 1) segments into neighboring segments.
    Strategy: prefer merging into the previous segment if exists; otherwise, next.
    """
    if not seg_result:
        return []

    # Convert to list of lists to allow mutation
    result = []
    i = 0
    n = len(seg_result)

    while i < n:
        current = seg_result[i]
        # If current segment is not isolated, just keep it
        if len(current) > 1:
            result.append(current)
            i += 1
        else:
            # Isolated segment: [x]
            merged = False
            # Try to merge with previous segment
            if result:  # there is a previous segment
                result[-1].extend(current)
                merged = True
            # If no previous, try next segment
            elif i + 1 < n:
                # Merge into next segment
                next_seg = seg_result[i + 1]
                result.append(current + next_seg)
                i += 1  # skip next segment as it's consumed
                merged = True

            if not merged:
                # No neighbors at all (shouldn't happen in normal cases)
                result.append(current)
            i += 1

    return result
def group_frames_by_integer_part(frame_values):
    if not frame_values:
        return []

    groups = []
    current_group = [0]
    current_int = int(frame_values[0])

    for i in range(1, len(frame_values)):
        val_int = int(frame_values[i])
        if val_int == current_int:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            current_int = val_int

    # Append the last group
    groups.append(current_group)
    groups=merge_isolated_segments(groups)
    return groups
def get_cum_pose_distances(frames,Rs, ts, alpha=1.0, beta=1.0):

    alpha = 1  # weight for translation (meters)
    beta  = 1  # weight for rotation (radians)

    cum_pose_distances = []
    for i in range(len(frames)):
        if i == 0:
            cum_pose_distances.append(0.0)
        else:
            dist = pose_distance(Rs[0], ts[0], Rs[i], ts[i], alpha=alpha, beta=beta)
            dist = round(dist, 1)
            cum_pose_distances.append(dist)

    print("\nCumulative pose distance from Frame 0 (translation + rotation):")
    for i, d in enumerate(cum_pose_distances):
        print(f"Frame {i:2d}: {d:.1f}")
    return cum_pose_distances
def compute_sigma_at(Rs, ts, i):
    N = len(Rs)
    if i == 0:
        return 0.0
    elif i == N - 1:
        return pose_distance(Rs[i], ts[i], Rs[i-1], ts[i-1])
    else:
        d_prev = pose_distance(Rs[i], ts[i], Rs[i-1], ts[i-1])
        d_next = pose_distance(Rs[i], ts[i], Rs[i+1], ts[i+1])
        return d_prev - d_next
def get_views_segments(frames):
    Rs, ts = get_camera_centers(frames) 
    cum_pose_distances = get_cum_pose_distances(frames,Rs, ts, alpha=1.0, beta=1.0)
    segs = group_frames_by_integer_part(cum_pose_distances)

    return segs
    

def get_views_segments_w_bias(frames,lam):
    Rs, ts = get_camera_centers(frames) 
    cum_pose_distances = get_cum_pose_distances(frames,Rs, ts, alpha=1.0, beta=1.0)
    sigma = compute_directional_bias(Rs, ts)
    cum_pose_distances = np.array(cum_pose_distances)
    sigma = np.array(sigma)
    
    pose_distances = cum_pose_distances + lam * sigma
    pose_distances = pose_distances.tolist()
    for i, d in enumerate(pose_distances):
        print(f"Frame {i:2d}: {d:.1f}")

    segs = group_frames_by_integer_part(pose_distances)
    return segs

def get_views_segments_w_bias_v2(frames, lam=0.1):
    # Step 1: 粗分段（不带 bias）
    Rs, ts = get_camera_centers(frames)
    cum_pose_distances = get_cum_pose_distances(frames, Rs, ts)  # 保留原始累计距离
    coarse_segs = group_frames_by_integer_part(cum_pose_distances)

    # Step 2: 仅对每个段的边界帧计算 sigma（段首 & 段尾）
    adjusted_cum = np.array(cum_pose_distances, dtype=float)

    for seg in coarse_segs:
        if len(seg) == 1:
            # 孤立帧，可直接跳过或特殊处理
            continue

        start_idx = seg[0]
        end_idx   = seg[-1]

        # 仅计算段首和段尾的 sigma（其他帧保持 cum_pose_distances 不变）
        sigma_start = compute_sigma_at(Rs, ts, start_idx)
        sigma_end   = compute_sigma_at(Rs, ts, end_idx)

        # 应用偏置：段首倾向于“向后拉”，段尾倾向于“向前推”
        adjusted_cum[start_idx] += lam * sigma_start
        adjusted_cum[end_idx]   += lam * sigma_end

    # Step 3: 基于调整后的距离重新分段
    adjusted_cum = np.round(adjusted_cum, 1).tolist()
    fine_segs = group_frames_by_integer_part(adjusted_cum)
    fine_segs = merge_isolated_segments(fine_segs)  # 保持鲁棒性

    return fine_segs

if __name__ == "__main__":
    file_path = './3D_QA/FastVGGT-main/outputs/pose/ScanQA_pose_refine.json'
    extrinsics_list = load_param(file_path)
    segments = []
    novelty = []
    from tqdm import tqdm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i, frames in tqdm(enumerate(extrinsics_list),total=len(extrinsics_list)):
        print(f"\nProcessing Scene {i}:")
        # segs = get_views_segments(frames)
        segs = get_views_segments(frames)
        # print(f"Identified {len(segs)} view segments:")
        # for seg_idx, seg in enumerate(segs):
            # print(f"Segment {seg_idx}: Frames {seg}")
        segments.append({i:segs})
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Total calculation time consumption: {elapsed_time:.2f} milliseconds")
    
    ###  Total calculation time consumption in 10015.77 milliseconds
    
    # with open('./3D_QA/FastVGGT-main/outputs/segs/ScanQA_segs_bias01.json','w') as f:
    #     json.dump(segments,f,indent=2)
    # with open('./3D_QA/FastVGGT-main/outputs/segs/scanqa_val_segs_real.json','w') as f:
    #     json.dump(segments,f,indent=2)
    

