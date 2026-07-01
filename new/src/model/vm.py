"""VelocityModule: 基于 velocity field 的点云去噪模型。

核心思想:
  把去噪建模成"速度场" —— 模型学习从噪声点指向干净点的位移向量。
  训练时在 noisy/clean 之间随机插值构造混合点云 pc_mix, 让模型预测
  pc_clean - pc_noisy 这个目标方向(MSE loss)。
  推理时用 Langevin 动力学迭代: pc = pc + v(pc) / num_steps, 跑 4 步。

  这本质上是 score-based denoising 的简化形式:
  - velocity ≈ score * (噪声尺度), 迭代更新就是 Langevin 采样的离散化。
  - 与完整 score matching 的区别: 这里直接监督"位移"而非"得分",
    训练更稳定, 但理论上不如学完整 score 分布通用。
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from .layers import Decoder, FeatureExtraction
from .ops import farthest_point_sampling, knn_query, patch_based_denoise
from .spec import ModelSpec
from ..data.asset import Asset


def _random_indices(n: int, m: int, device: torch.device) -> torch.Tensor:
    """从 [0, n) 随机取 m 个(不重复), 不足则全取。"""
    return torch.randperm(n, device=device)[: min(n, m)]


class VelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.frame_knn = int(cfg["frame_knn"])              # EdgeConv 的 k 近邻数
        self.num_train_points = int(cfg["num_train_points"])  # 训练时每个 step 采样的点数
        self.dsm_sigma = float(cfg["dsm_sigma"])            # loss 的归一化因子

        # 推理 patch 参数(原版硬编码在 predict_step 里, 这里提到 config 可调)
        self.predict_patch_size = int(cfg.get("predict_patch_size", 1000))
        self.predict_seed_k = int(cfg.get("predict_seed_k", 6))
        self.predict_seed_k_alpha = int(cfg.get("predict_seed_k_alpha", 1))
        self.langevin_steps = int(cfg.get("langevin_steps", 4))

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=int(cfg["feat_embedding_dim"]),
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_size=int(cfg["decoder_hidden_dim"]),
        )

    # ---------- 训练 ----------

    def _supervised_loss(
        self,
        pc_noisy: torch.Tensor,
        pc_mix: torch.Tensor,
        pc_clean: torch.Tensor,
    ) -> torch.Tensor:
        """在 pc_mix 上提特征, 预测位移, 与 pc_clean - pc_noisy 做 MSE。

        为省显存, 只在随机采样的 num_train_points 个点上算 loss。
        """
        b, n, d = pc_mix.shape
        idx = _random_indices(n, self.num_train_points, pc_mix.device)

        feat = self.encoder(pc_mix)                       # (B, N, C)
        feat = feat[:, idx, :]
        pc_noisy = pc_noisy[:, idx, :]
        pc_clean = pc_clean[:, idx, :]
        target = pc_clean - pc_noisy                       # 目标位移方向

        pred = self.decoder(feat.reshape(-1, feat.shape[2])).reshape(b, len(idx), d)
        return (((pred - target) ** 2) / self.dsm_sigma).sum(dim=-1).mean()

    def _denoise_langevin(self, pcl_noisy: torch.Tensor, num_steps: int = 4) -> torch.Tensor:
        """Langevin 动力学迭代去噪: pc_{t+1} = pc_t + v(pc_t) / steps。"""
        b, n, d = pcl_noisy.shape
        pc = pcl_noisy.clone()
        for _ in range(num_steps):
            feat = self.encoder(pc)
            pred = self.decoder(feat.reshape(-1, feat.shape[2])).reshape(b, n, d)
            pc = pc + pred / float(num_steps)
        return pc

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_mix = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        return {"loss": self._supervised_loss(pc_noisy, pc_mix, pc_clean)}

    def forward(self, **kwargs) -> Dict:
        return self.training_step(kwargs)

    # ---------- 推理 ----------

    @torch.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy = batch["pc_noisy"]  # (B, N, 3)
        if pc_noisy.ndim != 3:
            raise ValueError(f"expected pc_noisy shape (B, N, 3), got {pc_noisy.shape}")

        results = []
        for pc in pc_noisy:
            denoised = patch_based_denoise(
                denoise_fn=lambda p: self._denoise_langevin(p, self.langevin_steps),
                pcl_noisy=pc,
                patch_size=self.predict_patch_size,
                seed_k=self.predict_seed_k,
                seed_k_alpha=self.predict_seed_k_alpha,
            )
            results.append({"pc_denoised": denoised.detach().cpu().numpy().astype(np.float32)})
        return results

    # ---------- Asset -> tensor dict ----------

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        results: List[Dict] = []
        for asset in batch:
            if not self.is_predict():
                # 训练: 从 augment.Patch 产出的 meta 里取 patch 张量
                if not asset.meta:
                    raise ValueError("training asset does not contain patch metadata")
                results.append({
                    "pc_noisy": asset.meta["pc_noisy"],
                    "pc_clean": asset.meta["pc_clean"],
                    "pc_mix": asset.meta["pc_mix"],
                })
            else:
                # 推理: 直接用测试集的噪声点云
                if asset.sampled_vertices_noisy is None:
                    raise ValueError("prediction asset has no noisy point cloud")
                item = {"pc_noisy": asset.sampled_vertices_noisy}
                if asset.sampled_vertices is not None:
                    item["pc_clean"] = asset.sampled_vertices
                results.append(item)
        return results
