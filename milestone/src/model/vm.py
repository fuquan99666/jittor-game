from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import torch

from .feature import Decoder, FeatureExtraction, pairwise_squared_distance
from .spec import ModelSpec
from ..data.asset import Asset


def get_random_indices(n: int, m: int, device: torch.device) -> torch.Tensor:
    count = min(n, m)
    return torch.randperm(n, device=device)[:count]


class VelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        config = self.model_config

        self.frame_knn = int(config["frame_knn"])
        self.num_train_points = int(config["num_train_points"])
        self.dsm_sigma = float(config["dsm_sigma"])

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=int(config["feat_embedding_dim"]),
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=int(config["decoder_hidden_dim"]),
        )

    def get_supervised_loss(
        self,
        pc_noisy: torch.Tensor,
        pc_mix: torch.Tensor,
        pc_clean: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, point_count, dimension = pc_mix.shape
        point_indices = get_random_indices(
            point_count,
            self.num_train_points,
            pc_mix.device,
        )

        features = self.encoder(pc_mix)
        feature_dim = features.shape[2]

        features = features[:, point_indices, :]
        pc_noisy = pc_noisy[:, point_indices, :]
        pc_clean = pc_clean[:, point_indices, :]
        target_direction = pc_clean - pc_noisy

        predicted_direction = self.decoder(
            features.reshape(-1, feature_dim)
        ).reshape(batch_size, len(point_indices), dimension)

        return (
            ((predicted_direction - target_direction) ** 2.0) / self.dsm_sigma
        ).sum(dim=-1).mean()

    def denoise_langevin_dynamics(
        self,
        pcl_noisy: torch.Tensor,
        num_steps: int = 4,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, point_count, dimension = pcl_noisy.shape
        pcl_next = pcl_noisy.clone()
        for _ in range(num_steps):
            features = self.encoder(pcl_next)
            feature_dim = features.shape[2]
            predicted_direction = self.decoder(
                features.reshape(-1, feature_dim)
            ).reshape(batch_size, point_count, dimension)
            pcl_next = pcl_next + predicted_direction / float(num_steps)
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_mix = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        return {"loss": loss}

    def forward(self, **kwargs) -> Dict:
        return self.training_step(kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        if pc_noisy_batch.ndim != 3:
            raise ValueError(
                f"expected pc_noisy shape (B, N, 3), got {pc_noisy_batch.shape}"
            )

        results = []
        for pc_noisy in pc_noisy_batch:
            pc_denoised = patch_based_denoise(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=1,
            )
            results.append(
                {"pc_denoised": pc_denoised.detach().cpu().numpy().astype(np.float32)}
            )
        return results

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        results: List[Dict] = []
        for asset in batch:
            if not self.is_predict():
                if asset.meta is None:
                    raise ValueError("training asset does not contain patch metadata")
                results.append(
                    {
                        "pc_noisy": asset.meta["pc_noisy"],
                        "pc_clean": asset.meta["pc_clean"],
                        "pc_mix": asset.meta["pc_mix"],
                    }
                )
            else:
                if asset.sampled_vertices_noisy is None:
                    raise ValueError("prediction asset has no noisy point cloud")
                item = {"pc_noisy": asset.sampled_vertices_noisy}
                if asset.sampled_vertices is not None:
                    item["pc_clean"] = asset.sampled_vertices
                results.append(item)
        return results


def farthest_point_sampling(
    point_clouds: torch.Tensor,
    num_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if point_clouds.ndim != 3:
        raise ValueError("point_clouds must have shape (B, N, 3)")
    batch_size, point_count, _ = point_clouds.shape
    sample_count = max(1, min(num_points, point_count))

    centroids = torch.zeros(
        (batch_size, sample_count),
        dtype=torch.long,
        device=point_clouds.device,
    )
    distance = torch.full(
        (batch_size, point_count),
        float("inf"),
        dtype=point_clouds.dtype,
        device=point_clouds.device,
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=point_clouds.device)
    batch_indices = torch.arange(batch_size, device=point_clouds.device)

    for index in range(sample_count):
        centroids[:, index] = farthest
        centroid = point_clouds[batch_indices, farthest].unsqueeze(1)
        current_distance = ((point_clouds - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, current_distance)
        farthest = distance.max(dim=-1).indices

    sampled = point_clouds.gather(
        1,
        centroids.unsqueeze(-1).expand(-1, -1, point_clouds.shape[-1]),
    )
    return sampled, centroids


def knn_points(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbor_count = min(k, y.shape[1])
    distances = pairwise_squared_distance(x, y)
    dist_k, indices = torch.topk(
        distances,
        k=neighbor_count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    expanded_y = y.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    neighbors = expanded_y.gather(
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, y.shape[-1]),
    )
    return dist_k, indices, neighbors


@torch.no_grad()
def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy: torch.Tensor,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
) -> torch.Tensor:
    if pcl_noisy.ndim != 2 or pcl_noisy.shape[1] != 3:
        raise ValueError(f"expected point cloud shape (N, 3), got {pcl_noisy.shape}")

    point_count = pcl_noisy.shape[0]
    if point_count == 0:
        return pcl_noisy.clone()
    if point_count == 1:
        return pcl_noisy.clone()

    actual_patch_size = min(patch_size, point_count)
    num_patches = max(1, int(seed_k * point_count / actual_patch_size))
    num_patches = min(num_patches, point_count)

    batched_cloud = pcl_noisy.unsqueeze(0)
    seed_points, _ = farthest_point_sampling(batched_cloud, num_patches)
    patch_distances, point_indices, patches = knn_points(
        seed_points,
        batched_cloud,
        actual_patch_size,
    )

    patches = patches[0]
    patch_distances = patch_distances[0]
    point_indices = point_indices[0]
    seed_points = seed_points[0]

    seed_expand = seed_points.unsqueeze(1).expand_as(patches)
    centered_patches = patches - seed_expand

    denominator = patch_distances[:, -1:].clamp_min(1e-8)
    normalized_distances = patch_distances / denominator

    all_distances = torch.full(
        (num_patches, point_count),
        float("inf"),
        dtype=pcl_noisy.dtype,
        device=pcl_noisy.device,
    )
    all_distances.scatter_(1, point_indices, normalized_distances)
    weights = torch.exp(-all_distances)
    best_patch = weights.argmax(dim=0)
    covered = torch.isfinite(all_distances).any(dim=0)

    chunk_size = max(
        1,
        int(ceil(point_count / max(seed_k_alpha * actual_patch_size, 1))),
    )
    denoised_chunks = []
    for start in range(0, num_patches, chunk_size):
        current = centered_patches[start : start + chunk_size]
        output, _ = model.denoise_langevin_dynamics(current, num_steps=4)
        denoised_chunks.append(output)

    denoised_patches = torch.cat(denoised_chunks, dim=0) + seed_expand

    output_cloud = pcl_noisy.clone()
    for patch_id in range(num_patches):
        global_indices = point_indices[patch_id]
        use_local = best_patch[global_indices] == patch_id
        if use_local.any():
            selected_global = global_indices[use_local]
            output_cloud[selected_global] = denoised_patches[patch_id, use_local]

    output_cloud[~covered] = pcl_noisy[~covered]
    return output_cloud
