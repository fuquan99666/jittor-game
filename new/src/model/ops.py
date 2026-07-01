"""点云几何运算的 torch 实现: FPS / KNN / patch-based denoise。

这些算子是模型推理时用的(训练时用 numpy 版的 augment.Patch)。
纯 torch 实现, 无外部依赖; 若需更高性能可替换成 pytorch3d 的对应函数。
"""
from math import ceil
from typing import Tuple

import torch

from .layers import pairwise_squared_distance


def farthest_point_sampling(
    point_clouds: torch.Tensor,
    num_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """最远点采样。返回 (sampled_points (B, M, C), indices (B, M))。

    每次选离已选集合最远的点, 保证空间覆盖均匀。O(N*M) 复杂度。
    """
    if point_clouds.ndim != 3:
        raise ValueError("point_clouds must have shape (B, N, 3)")
    b, n, _ = point_clouds.shape
    m = max(1, min(num_points, n))

    centroids = torch.zeros((b, m), dtype=torch.long, device=point_clouds.device)
    distance = torch.full((b, n), float("inf"), dtype=point_clouds.dtype, device=point_clouds.device)
    farthest = torch.zeros(b, dtype=torch.long, device=point_clouds.device)
    batch_idx = torch.arange(b, device=point_clouds.device)

    for i in range(m):
        centroids[:, i] = farthest
        centroid = point_clouds[batch_idx, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((point_clouds - centroid) ** 2).sum(dim=-1))
        farthest = distance.max(dim=-1).indices

    sampled = point_clouds.gather(1, centroids.unsqueeze(-1).expand(-1, -1, point_clouds.shape[-1]))
    return sampled, centroids


def knn_query(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对 x 中每个点在 y 里找 k 近邻。

    返回 (distances (B, N, k), indices (B, N, k), neighbors (B, N, k, C))。
    """
    neighbor_count = min(k, y.shape[1])
    dist = pairwise_squared_distance(x, y)
    dist_k, indices = torch.topk(dist, k=neighbor_count, dim=-1, largest=False, sorted=True)
    neighbors = y.unsqueeze(1).expand(-1, x.shape[1], -1, -1).gather(
        2, indices.unsqueeze(-1).expand(-1, -1, -1, y.shape[-1])
    )
    return dist_k, indices, neighbors


@torch.no_grad()
def patch_based_denoise(
    denoise_fn,
    pcl_noisy: torch.Tensor,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
) -> torch.Tensor:
    """基于 patch 的去噪: FPS 选种子 -> KNN 切 patch -> 每个 patch 跑 denoise_fn -> 加权拼回。

    denoise_fn(patches_centered) -> denoised_patches, 形状 (num_patches, patch_size, 3)。
    拼回时每个点只采用其最近种子所在 patch 的结果(用 softmax 权重 argmax 决定归属)。
    """
    if pcl_noisy.ndim != 2 or pcl_noisy.shape[1] != 3:
        raise ValueError(f"expected point cloud shape (N, 3), got {pcl_noisy.shape}")

    n = pcl_noisy.shape[0]
    if n <= 1:
        return pcl_noisy.clone()

    actual_patch_size = min(patch_size, n)
    num_patches = min(max(1, int(seed_k * n / actual_patch_size)), n)

    batched = pcl_noisy.unsqueeze(0)
    seed_points, _ = farthest_point_sampling(batched, num_patches)
    patch_dist, point_idx, patches = knn_query(seed_points, batched, actual_patch_size)

    patches = patches[0]
    patch_dist = patch_dist[0]
    point_idx = point_idx[0]
    seed_points = seed_points[0]

    # patch 中心化: 减去各自种子点
    centered = patches - seed_points.unsqueeze(1).expand_as(patches)

    # 每个点到其 patch 最远邻居的距离做归一化, 再 exp 得权重(越近权重越大)
    norm_dist = patch_dist / patch_dist[:, -1:].clamp_min(1e-8)
    all_dist = torch.full(
        (num_patches, n), float("inf"), dtype=pcl_noisy.dtype, device=pcl_noisy.device
    )
    all_dist.scatter_(1, point_idx, norm_dist)
    weights = torch.exp(-all_dist)
    best_patch = weights.argmax(dim=0)             # 每个点归属哪个 patch
    covered = torch.isfinite(all_dist).any(dim=0)  # 是否被任何 patch 覆盖

    # 分块跑 denoise, 避免 num_patches 太大时显存爆
    chunk_size = max(1, int(ceil(n / max(seed_k_alpha * actual_patch_size, 1))))
    denoised_chunks = []
    for start in range(0, num_patches, chunk_size):
        out = denoise_fn(centered[start : start + chunk_size])
        denoised_chunks.append(out)
    denoised_patches = torch.cat(denoised_chunks, dim=0) + seed_points.unsqueeze(1).expand_as(patches)

    # 拼回: 每个点取其 best_patch 的去噪结果
    output = pcl_noisy.clone()
    for pid in range(num_patches):
        gidx = point_idx[pid]
        use = best_patch[gidx] == pid
        if use.any():
            output[gidx[use]] = denoised_patches[pid, use]

    output[~covered] = pcl_noisy[~covered]
    return output
