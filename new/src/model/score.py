"""ScoreModule: 基于 Denoising Score Matching 的点云去噪模型。

与 VelocityModule 的区别:
  VelocityModule 监督"总位移" pc_clean - pc_noisy, 是 score-based 的简化。
  ScoreModule 监督"得分函数" -ε/σ, 是完整的 score matching:
    - 训练: x_noisy = x_clean + σε, 模型预测 s(x_noisy, σ) ≈ -ε/σ
    - 推理: 退火 Langevin 动力学, σ 从大到小多轮迭代去噪

  优势:
    1. 学概率密度梯度, 理论上更通用, 能处理多模态分布
    2. 退火机制能处理不同噪声级别, 鲁棒性更强
    3. sigma 作为条件注入, 模型显式感知当前噪声尺度

  复用: FeatureExtraction, Decoder, patch_based_denoise, data 层全部复用。
  新增: sigma_encoder (把标量 σ 编码后加到点特征上)。
"""
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from .layers import Decoder, FeatureExtraction
from .ops import patch_based_denoise
from .spec import ModelSpec
from ..data.asset import Asset


class ScoreModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.frame_knn = int(cfg["frame_knn"])
        self.embedding_dim = int(cfg["feat_embedding_dim"])

        # 推理参数
        self.predict_patch_size = int(cfg.get("predict_patch_size", 1000))
        self.predict_seed_k = int(cfg.get("predict_seed_k", 6))
        self.predict_seed_k_alpha = int(cfg.get("predict_seed_k_alpha", 1))
        # 退火 Langevin 的 sigma 序列(从大到小)和每级步数
        self.predict_sigmas = [float(s) for s in cfg.get("predict_sigmas", [0.02, 0.01, 0.005, 0.0025])]
        self.predict_steps_per_sigma = int(cfg.get("predict_steps_per_sigma", 5))
        # Langevin 步长系数(越小越稳定, 越大收敛越快)
        self.langevin_step_scale = float(cfg.get("langevin_step_scale", 0.5))

        # 复用现有 encoder + decoder
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=self.embedding_dim,
        )
        self.decoder = Decoder(
            z_dim=self.embedding_dim,
            out_dim=3,
            hidden_size=int(cfg["decoder_hidden_dim"]),
        )

        # 新增: sigma 条件编码器, 把标量 σ 变成 embedding_dim 向量
        # 用 sinusoidal 编码 + MLP, 使模型对不同 σ 有平滑响应
        self.sigma_encoder = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    # ---------- 得分预测 ----------

    def _predict_score(self, pc: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """给定点云 (B, N, 3) 和噪声级别 sigma (B,), 预测得分 (B, N, 3)。

        sigma 经 sigma_encoder 编码后加到每个点的特征上(条件注入)。
        """
        b, n, _ = pc.shape
        feat = self.encoder(pc)                              # (B, N, C)
        sigma_feat = self.sigma_encoder(sigma.unsqueeze(1))  # (B, C)
        feat = feat + sigma_feat.unsqueeze(1)                # 广播到每个点
        score = self.decoder(feat.reshape(-1, feat.shape[2])).reshape(b, n, 3)
        return score

    # ---------- 训练 ----------

    def _score_matching_loss(
        self,
        pc_noisy: torch.Tensor,      # (B, N, 3) 加噪点
        score_target: torch.Tensor,  # (B, N, 3) -ε/σ
        sigma: torch.Tensor,         # (B,) 每个 patch 的噪声级别
    ) -> torch.Tensor:
        pred = self._predict_score(pc_noisy, sigma)
        # 按 σ² 归一化: 让不同噪声级别的 loss 贡献量级一致。
        # 数学上等价于 score matching 的加权形式 E[σ²·||s - (-ε/σ)||²],
        # 也就是 E[||σ·s + ε||²], 即 "denoising score matching" 的等价目标。
        # 这样小 σ 的大 target 不会主导 loss, 训练更稳, 数值也更小。
        weight = (sigma ** 2).reshape(-1, 1, 1)  # (B, 1, 1)
        return (((pred - score_target) ** 2) * weight).sum(dim=-1).mean()

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        score_target = batch["score_target"].reshape(-1, patch_size, 3)
        sigma = batch["sigma"].reshape(-1)
        return {"loss": self._score_matching_loss(pc_noisy, score_target, sigma)}

    def forward(self, **kwargs) -> Dict:
        return self.training_step(kwargs)

    # ---------- 推理: 退火 Langevin 动力学 ----------

    @torch.no_grad()
    def _annealed_langevin(self, pcl_noisy: torch.Tensor) -> torch.Tensor:
        """退火 Langevin 动力学: σ 从大到小多轮迭代去噪。

        每一级 σ 跑 predict_steps_per_sigma 步:
          x = x + (α/2) * s(x, σ) + √α * z,  z ~ N(0, I)
        其中 α = langevin_step_scale * σ² / σ_min² (相对步长, 大 σ 时步大)。

        加随机项 z 是真正的 Langevin 采样; 为稳定性可选关闭(确定性梯度上升)。
        """
        if pcl_noisy.ndim == 2:
            pcl_noisy = pcl_noisy.unsqueeze(0)
        b, n, _ = pcl_noisy.shape
        pc = pcl_noisy.clone()
        sigma_min = self.predict_sigmas[-1]

        for sigma in self.predict_sigmas:
            # 步长与 σ² 成正比, 归一化到最小 σ
            alpha = self.langevin_step_scale * (sigma / sigma_min) ** 2
            sigma_t = torch.full((b,), sigma, device=pc.device, dtype=pc.dtype)

            for _ in range(self.predict_steps_per_sigma):
                score = self._predict_score(pc, sigma_t)
                pc = pc + 0.5 * alpha * score
                # 加随机噪声项( Langevin 采样), 幅度与 sqrt(alpha) 成正比
                pc = pc + (alpha ** 0.5) * torch.randn_like(pc) * sigma

        return pc

    @torch.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy = batch["pc_noisy"]  # (B, N, 3)
        if pc_noisy.ndim != 3:
            raise ValueError(f"expected pc_noisy shape (B, N, 3), got {pc_noisy.shape}")

        results = []
        for pc in pc_noisy:
            denoised = patch_based_denoise(
                denoise_fn=self._annealed_langevin,
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
                if not asset.meta:
                    raise ValueError("training asset does not contain score metadata")
                results.append({
                    "pc_noisy": asset.meta["pc_noisy"],
                    "score_target": asset.meta["score_target"],
                    "sigma": asset.meta["sigma"],
                })
            else:
                if asset.sampled_vertices_noisy is None:
                    raise ValueError("prediction asset has no noisy point cloud")
                item = {"pc_noisy": asset.sampled_vertices_noisy}
                if asset.sampled_vertices is not None:
                    item["pc_clean"] = asset.sampled_vertices
                results.append(item)
        return results
