from concurrent.futures import ThreadPoolExecutor
import torch
import math
from datasets import load_from_disk
from process_patch_coordinates import *
import heapq
import os

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
torch.manual_seed(42)



class OT_compressor():
    def __init__(self, original_features, intermediate_feature, budget, eps=0.1, max_iter=100, update_steps=10, lr=1e-2):
        
        self.device = original_features.device
        self.dtype = original_features.dtype

        self.original_features = F.normalize(original_features, dim=1).to(torch.float32)
        self.original_features.requires_grad = True
        self.intermediate_feature = F.normalize(intermediate_feature, dim=1).to(torch.float32)
        self.intermediate_feature.requires_grad = True
        
        self.Q = torch.nn.Parameter(self.intermediate_feature.clone().to(self.device))
        self.Q.requires_grad = True

        self.budget = budget
        print(f"OT Compressor initialized with budget {self.budget}, original size {original_features.size(0)}, compressed size {self.Q.size(0)}")
        self.N = self.Q.size(0)         # Compressed size (Source in OT)
        self.M = self.original_features.size(0) # Original size (Target in OT)
        
        self.eps = eps
        self.max_iter = max_iter
        self.update_steps = update_steps
        self.lr = lr

        self.optimizer = torch.optim.Adam([self.Q], lr=self.lr)
        
        self.scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=self.update_steps, 
                    eta_min=1e-5
                )
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    def compute_diversity_loss(self,Q):
        

        sim_matrix = torch.matmul(Q, Q.T) 

        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        sim_upper = sim_matrix[mask]
        
        # 4. 计算正则 Loss
        # 方案 A (强力): 惩罚所有相似度的平方和 (让大家尽量正交)
        loss_div = torch.mean(sim_upper ** 2)
        
        return loss_div
    def forward(self):

        for i in range(self.update_steps):
            self.optimizer.zero_grad()
            Q_norm = F.normalize(self.Q, dim=1,eps=1e-8)

            sim = torch.matmul(Q_norm, self.original_features.t()).to(torch.float32) 
            
            xx=torch.zeros(1, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
            yy=torch.zeros(1, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
            
            C = (1-sim).to(torch.float32) 
            # T = ot.sinkhorn(a, b, C, reg=self.eps, numItermax=self.max_iter, method='sinkhorn')
            K = torch.exp(-C / self.eps).unsqueeze(0).to(torch.float32) 
            

            T = self.Sinkhorn(K,xx,yy)


            loss = torch.sum(T * C)

            loss.backward()
            self.optimizer.step()
            
            current_lr = self.scheduler.get_last_lr()[0] 
            print(f"Step {i}, OT Loss: {loss.item():.6f}    LR: {current_lr:.6f}")

    def uniform_sampling(self,indices, k):
        total_frames = len(indices)
        if total_frames <= k:
            return torch.arange(total_frames).tolist()
        interval = total_frames / k
        selected_indices = [int(i * interval) for i in range(k)]

        return selected_indices
    def select_final_tokens(self,k=2):
        with torch.no_grad():
            Q_norm = F.normalize(self.Q, dim=1)
            sim = torch.matmul(Q_norm, self.original_features.t())

            C = (1.0 - sim).to(torch.float32)
            
            xx=torch.zeros(1, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
            yy=torch.zeros(1, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
            K = torch.exp(-C / self.eps).unsqueeze(0)
            T = self.Sinkhorn(K,xx,yy).squeeze(0)
            # print('T shape',T.shape)
            
        
            
            #######col topk#########
            topk_vals, topk_indices = torch.topk(T, k=k, dim=1) 
            #######col topk#########
            
            flatten_indices = topk_indices.flatten()
            kk = torch.unique(flatten_indices)
            print(f"Total unique tokens selected before budget check: {kk.shape}")
            unique_indices = torch.unique(flatten_indices)
            
            uniform_unique_indices = self.uniform_sampling(unique_indices.tolist(), k=self.budget)
            
            print(f"Selected {len(uniform_unique_indices)} tokens after OT compression.")
            self.original_features.detach()
            self.intermediate_feature.detach()
            self.Q.detach()
            return uniform_unique_indices


class data_param:
    def __init__(self):
        self.scene_id = None
        self.data_id = None
        self.dataset = None
        self.geometry_data = None
        self.key_frames_idx = None
        

        
    def get_geo_data(self):
        self.scene_id_set = {val: i for i, val in enumerate(self.geometry_data['scene_id'])}
        idx = self.scene_id_set[self.scene_id]
        extrinsics = self.geometry_data['extrinsics'][idx]
        intrinsics = self.geometry_data['intrinsics'][idx]
        depth_map = self.geometry_data['depth_maps'][idx]

        
        key_frames_idx = self.key_frames_idx
        
        
        return extrinsics, intrinsics,depth_map,key_frames_idx
           
        
geo_param = data_param()   



def rbf_kernel_distance(x, y, gamma):

    dist_sq = torch.cdist(x, y, p=2) ** 2

    rbf_similarity = torch.exp(-gamma * dist_sq)
    
    # rbf_similarity=torch.exp(-dist_sq / (2 * 1))  #sigma=1

    rbf_dist = 1.0 - rbf_similarity
    return rbf_dist

def kmeans_spatial_rbf(X, num_clusters, points=None, max_iter=20, tolerance=1e-4, 
                       spatial_weight=1.0,spatial_gamma=10):

    print(f'Using RBF-Kmeans initializer with {num_clusters} clusters')
    
    X = X.float()
    M, D = X.shape
    device = X.device
  
    indices = torch.randperm(M, device=device)[:num_clusters]
    centroids = X[indices].clone() # Embedding 质心
    
    geo_centroids = None
    if points is not None:
        points = points.float().to(device)
        geo_centroids = points[indices].clone() # 空间 质心初始化
        print(f"Joint clustering enabled. Spatial weight: {spatial_weight}")

    for i in range(max_iter):
        old_centroids = centroids.clone()
        if points is not None:
            old_geo_centroids = geo_centroids.clone()
            
    
        dists = torch.cdist(X, centroids) # [N, K]
        
        if points is not None:
            point_dists = rbf_kernel_distance(points, geo_centroids, spatial_gamma) # [N, K]
            
            dists = dists + (point_dists * spatial_weight)

        labels = torch.argmin(dists, dim=1) # [N]
        
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, 1, device=device)
        
        new_centroids.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(M, 1, device=device))

        counts = torch.clamp(counts, min=1.0)
        centroids = new_centroids / counts

        if points is not None:
            new_geo_centroids = torch.zeros_like(geo_centroids)
            new_geo_centroids.index_add_(0, labels, points)
            geo_centroids = new_geo_centroids / counts

        # --- 收敛判断 ---
        center_shift = torch.norm(centroids - old_centroids)
        
        if points is not None:
            geo_shift = torch.norm(geo_centroids - old_geo_centroids)
            total_shift = center_shift + geo_shift
        else:
            total_shift = center_shift

        if total_shift < tolerance:
            print(f"Converged at iteration {i}")
            break
            
    print(f'Centroids: {centroids.shape} Initialized')
    
    return centroids

def _compress_block(args):
    block_tokens, block_budget, start_idx, block_points = args
    if block_budget >= block_tokens.size(0):
        selected_local = list(range(block_tokens.size(0)))
    else:
        selected_local = floccompress_fully_no_print_func(block_tokens, block_budget,block_points)
    return [start_idx + idx for idx in selected_local]

def floccompress_fully_no_print_func(tokens, budget,points):
    N, D = tokens.shape

    assert budget <= N
    device = tokens.device
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=1)
    sim_matrix = torch.mm(tokens, tokens.t())  # [N, N]
    if points is not None:
        rbf_matrix = compute_rbf_matrix(points,1).to(device)

        sim_matrix =sim_matrix + rbf_matrix
        
    
    best_sim = torch.zeros(N, device=device)
    
    marginal_gain = sim_matrix.sum(dim=1)
    

    Q = [(-marginal_gain[i].item(), i) for i in range(N)]
    heapq.heapify(Q)

    selected = []
    for _ in range(budget):
        while True:
            neg_delta, candidate = heapq.heappop(Q)
            delta_cached = -neg_delta
            real_delta = torch.sum(torch.clamp(sim_matrix[:, candidate] - best_sim, min=0.0)).item()
            if Q:
                current_max = -Q[0][0]
            else:
                current_max = -float('inf')
            if real_delta >= current_max:
                selected.append(candidate)
                best_sim = torch.max(best_sim, sim_matrix[:, candidate])
                break
            else:
                heapq.heappush(Q, (-real_delta, candidate))
    return selected


def random_init(sample):
    N,D = sample.shape
    feature = torch.nn.Parameter(
        torch.nn.init.kaiming_uniform_(
            torch.empty(N, D)
        )
    )
    feature.data[:] = feature / feature.norm(dim=-1, keepdim=True)
    
    return feature.to(sample.device)
        
def kmeans(X, num_clusters,max_iter=20, tolerance=1e-4):
    print(f'Using kmeans initializer with {num_clusters} clusters')
    X = X.float()

    M, D = X.shape
    device = X.device
    indices = torch.randperm(M, device=device)[:num_clusters]
    centroids = X[indices].clone()

    
    for i in range(max_iter):
        old_centroids = centroids.clone()
        # print('X ',X.dtype)
        # print('center ',centroids.dtype)

        dists = torch.cdist(X, centroids)

        labels = torch.argmin(dists, dim=1)
        
        new_centroids = torch.zeros_like(centroids)

        counts = torch.zeros(num_clusters, 1, device=device)

        new_centroids.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(M, 1, device=device))

        counts = torch.clamp(counts, min=1.0)
        centroids = new_centroids / counts

        center_shift = torch.norm(centroids - old_centroids)
        if center_shift < tolerance:
            break
    print(f'Centroids: {centroids.shape} Initialized')
    return centroids

def kmeans_spatial(X, num_clusters, points=None, max_iter=20, tolerance=1e-4, spatial_weight=1.0):

    print(f'Using kmeans initializer with {num_clusters} clusters')
    
    X = X.float()
    M, D = X.shape
    device = X.device
  
    indices = torch.randperm(M, device=device)[:num_clusters]
    centroids = X[indices].clone() # Embedding 质心
    
    geo_centroids = None
    if points is not None:
        points = points.float().to(device)
        geo_centroids = points[indices].clone() # 空间 质心初始化
        print(f"Joint clustering enabled. Spatial weight: {spatial_weight}")

    for i in range(max_iter):
        old_centroids = centroids.clone()
        if points is not None:
            old_geo_centroids = geo_centroids.clone()
        dists = torch.cdist(X, centroids) # [N, K]

        if points is not None:
            
            point_dists = torch.cdist(points, geo_centroids) # [N, K]
            # dist_var = torch.var(dists)
            # point_dists_var = torch.var(point_dists)
            # alpha = dist_var/(point_dists_var+dist_var+1e-6)
            dists = dists + (point_dists * spatial_weight)
            # dists = alpha*dists + (1-alpha)*point_dists
            
        # 3. 分配标签
        labels = torch.argmin(dists, dim=1) # [N]

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, 1, device=device)
        
        # 累加 Embedding
        new_centroids.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(M, 1, device=device))
        
        # 防止除以0
        counts = torch.clamp(counts, min=1.0)
        centroids = new_centroids / counts
        
        # 如果有点云数据，必须同时更新 空间质心
        if points is not None:
            new_geo_centroids = torch.zeros_like(geo_centroids)
            # 累加 空间坐标
            new_geo_centroids.index_add_(0, labels, points)
            # 计算新的空间平均值
            geo_centroids = new_geo_centroids / counts

        center_shift = torch.norm(centroids - old_centroids)
        
        # 如果是联合聚类，通常也要考虑空间质心的位移，或者只看embedding也可以
        if points is not None:
            geo_shift = torch.norm(geo_centroids - old_geo_centroids)
            # 这里的收敛条件可以根据需求调整，这里选择两者之和
            total_shift = center_shift + geo_shift
        else:
            total_shift = center_shift

        if total_shift < tolerance:
            print(f"Converged at iteration {i}")
            break
            
    print(f'Centroids: {centroids.shape} Initialized')
    
    # 返回 embedding 质心 (如果需要返回空间质心，可以修改返回值)
    return centroids

def floccompress_with_blocks_parallel(tokens, budget, T=4, max_workers=8,is_geometry=False,is_ot = False):
    # tokens=tokens.view(-1, tokens.size(-1))
    print(f"Compressing {tokens.size(0)} tokens with budget {budget} using T={T} and max_workers={max_workers}")
    N, D = tokens.shape
    # tokens = tokens.cpu()
    if is_ot:
        real_budget = budget
        # print("Real budget: ",real_budget)
        budget =budget//2
        # print("Budget:",budget)
    if is_ot:
        points_tensor = None
        if is_geometry:
            print("Using geometry data for compression.")
            params = geo_param.get_geo_data()
            extrinsics, intrinsics,depth_map,key_frames_idx = params
            points_tensor = from_patch2coordinates(
                                            depths=depth_map,
                                            extrinsics=extrinsics, 
                                            intrinsics=intrinsics,
                                            key_frame_idx=key_frames_idx).to(tokens.device)
            print(f"Generated {points_tensor.size(0)} 3D points for geometry-aware compression.")
        with torch.enable_grad():
            # intermediate_feature = random_init(sample = tokens[sorted(selected_global)])
            
            # intermediate_feature = kmeans_spatial(tokens,budget,points=points_tensor)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            intermediate_feature=kmeans(tokens,budget)
            # intermediate_feature = kmeans_spatial_rbf(tokens,budget,points=points_tensor)
            
            # intermediate_feature = tokens[selected_global]
            
            
            update_steps = int(os.getenv('OT_UPDATE_STEPS',15))
            
            k = int (os.getenv('OT_TOPK',6))
            print(f"OT Compressor settings - Update Steps: {update_steps}, Top-K per centroid: {k}")
            ot_compressor = OT_compressor(original_features=tokens,
                                        intermediate_feature=intermediate_feature,
                                        budget=real_budget,
                                        eps=0.1,
                                        max_iter=100,
                                        update_steps=update_steps,
                                        # update_steps=7,
                                        lr=1e-2)
            ot_compressor.forward()
            selected_global = ot_compressor.select_final_tokens(k=k)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"OT Compression completed in {elapsed_time:.2f} milliseconds")
            
            
            return selected_global
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
        
    num_blocks = (N + T - 1) // T
    budget_per_block = budget // num_blocks
    extra = budget % num_blocks
    points_tensor = None
    if is_geometry:
        print("Using geometry data for compression.")
        params = geo_param.get_geo_data()
        extrinsics, intrinsics,depth_map,key_frames_idx = params
        points_tensor = from_patch2coordinates(
                                        depths=depth_map,
                                        extrinsics=extrinsics, 
                                        intrinsics=intrinsics,
                                        key_frame_idx=key_frames_idx).to(tokens.device)
        print(f"Generated {points_tensor.size(0)} 3D points for geometry-aware compression.")
    print('\nStarting parallel FLoC block compression...\n')
    tasks = []
    for i in range(num_blocks):
        start_idx = i * T
        end_idx = min((i + 1) * T, N)
        block_tokens = tokens[start_idx:end_idx]
        block_budget = budget_per_block + (1 if i < extra else 0)
        
        if points_tensor is not None:
            block_points = points_tensor[start_idx:end_idx]
            tasks.append((block_tokens, block_budget, start_idx, block_points))
        else:
            tasks.append((block_tokens, block_budget, start_idx, None))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compress_block, tasks))

    selected_global = []
    for res in results:
        selected_global.extend(res)
        
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"FLoC completed in {elapsed_time:.2f} milliseconds")
            
            
    return sorted(selected_global)






# def main():
#     num_image = 16
#     ratio = 2
#     with open('/home/wds/sdw/3D_QA/image_features.pt', 'rb') as f:
#         image_features = torch.load(f,weights_only=True).to('cuda:2')
#     tokens = image_features.view(-1, image_features.size(-1))
#     budgets = tokens.size(0)//ratio
#     import time 
#     start = time.time()
#     selected_indices = floccompress_with_blocks_parallel(tokens, budgets, T=8, max_workers=8,is_ot=True)

#     end = time.time()
#     print(f"Time taken: {end - start} seconds")
#     compressed_tokens = tokens[selected_indices]
#     compressed_images_len  = math.ceil(num_image/ratio)
#     per_image_len = compressed_tokens.size(0)//(compressed_images_len)
    
#     print("Compressed tokens shape:", compressed_tokens.shape)
#     # final_tokens = []
#     # for i in range(compressed_images_len):
#     #     start_ = i*per_image_len
#     #     end_ = (i+1)*per_image_len if i< (compressed_images_len -1) else compressed_tokens.size(0)
#     #     img_tokens = compressed_tokens[start_:end_]
#     #     print(f"Image {i} tokens shape: {img_tokens.shape}")
#     #     final_tokens.append(img_tokens)
#     # final_token_size=  final_tokens[-1].size(0)
#     # is_final_padding = False
#     # is_former_padding = False
#     # if final_token_size < per_image_len:
#     #     final_padding_size = per_image_len - final_token_size
#     #     padding_tokens = torch.zeros(final_padding_size, compressed_tokens.size(1))
#     #     final_tokens[-1] = torch.cat([final_tokens[-1], padding_tokens], dim=0)
#     #     is_final_padding = True
#     # elif final_token_size > per_image_len:
#     #     for i in range(len(final_tokens)-1):
#     #         token = final_tokens[i]
#     #         if token.size(0) < final_token_size:
#     #             former_padding_size = final_token_size - token.size(0)
#     #             padding_tokens = torch.zeros(former_padding_size, compressed_tokens.size(1))
#     #             final_tokens[i] = torch.cat([token, padding_tokens], dim=0)
#     #     is_former_padding = True     
#     # final_tokens = torch.stack(final_tokens, dim=0)

            
            
#     # print("Final tokens shape after regrouping:", final_tokens.shape)

    
#     # print(f"Selected {len(selected_indices)} indices out of {tokens.size(0)} tokens.")
#     # print("Sample of selected indices:", selected_indices[:10])
    
    
    
#     # current_tokens = []
#     # if is_final_padding:
#     #     for i in range(final_tokens.size(0)-1):
#     #         current_tokens.append(final_tokens[i])
#     #     actual_len = final_tokens.size(1) - final_padding_size
#     #     current_tokens.append(final_tokens[-1][:actual_len])
#     # elif is_former_padding:
#     #     actual_len = final_tokens.size(1) - former_padding_size
#     #     for i in range(final_tokens.size(0)-1):
#     #         current_tokens.append(final_tokens[i][:actual_len])
#     #     current_tokens.append(final_tokens[-1])
#     # else:
#     #     for i in range(final_tokens.size(0)):
#     #         current_tokens.append(final_tokens[i])
    
    
    
#     # print("Final adjusted tokens per image:")
#     # for i, token in enumerate(current_tokens):
#     #     print(f"Image {i} tokens shape: {token.shape}")
    
    
    
# if __name__ == "__main__":
#     main()