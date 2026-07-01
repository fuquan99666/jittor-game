"""数据增强(augment): 每个 Augment 就地修改一个 Asset。

augment 链由 transform YAML 里的 augments 列表决定, 顺序执行:
  sample -> normalize_pc -> add_noise -> linear -> patch   (训练)
  sample -> normalize_pc -> add_noise -> patch             (验证)
  (空)                                                     (预测, 测试集已是 noisy.npy)

注意: predict_transform 必须为空, 否则会破坏已带噪的测试输入。
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .asset import Asset
from .sampling import random_euler_rotation, sample_vertex_groups, normalize_to_unit_sphere
from .spec import ConfigSpec


@dataclass(frozen=True)
class Augment(ConfigSpec, ABC):
    """Augment 基类。frozen=True 保证配置不可变, 多 worker 下安全。

    子类只需声明 dataclass 字段并实现 apply(); parse() 由基类统一提供:
    check_keys 校验后直接 cls(**kwargs) 构造。
    """

    @classmethod
    def parse(cls, **kwargs) -> "Augment":
        cls.check_keys(kwargs)
        return cls(**kwargs)

    @abstractmethod
    def apply(self, asset: Asset) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class AugmentSample(Augment):
    """从网格表面采样点云, 写入 asset.sampled_vertices。"""
    num_samples: int
    num_vertex_samples: int = 0

    def apply(self, asset: Asset) -> None:
        if asset.vertices is None or asset.faces is None:
            raise ValueError("AugmentSample needs mesh vertices/faces")
        sampled, _ = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled


@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    """把 sampled_vertices 归一化到单位球。"""

    def apply(self, asset: Asset) -> None:
        if asset.sampled_vertices is None:
            raise ValueError("AugmentNormalizePC needs sampled_vertices")
        normalized, _, _ = normalize_to_unit_sphere(asset.sampled_vertices)
        asset.sampled_vertices = normalized


@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    """对 sampled_vertices 加 Laplace 噪声, 结果存入 sampled_vertices_noisy。"""
    noise_std_min: float
    noise_std_max: float

    def apply(self, asset: Asset) -> None:
        if asset.sampled_vertices is None:
            raise ValueError("AugmentAddNoise needs sampled_vertices")
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0.0, noise_std, size=asset.sampled_vertices.shape)
        asset.sampled_vertices_noisy = asset.sampled_vertices + noise


@dataclass(frozen=True)
class AugmentLinear(Augment):
    """随机各向异性缩放 + 欧拉旋转, 对全部坐标字段施加同一仿射变换。

    scale_p / rotate_p 独立判定, 因此四象限概率为:
      都做 = scale_p*rotate_p, 只缩放 = scale_p*(1-rotate_p),
      只旋转 = (1-scale_p)*rotate_p, 都不做 = (1-scale_p)*(1-rotate_p)。
    """
    scale: Tuple[float, float] = (1.0, 1.0)
    rotate_x_range: Tuple[float, float] = (0.0, 0.0)
    rotate_y_range: Tuple[float, float] = (0.0, 0.0)
    rotate_z_range: Tuple[float, float] = (0.0, 0.0)
    scale_p: float = 0.0
    rotate_p: float = 0.0

    def apply(self, asset: Asset) -> None:
        trans = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            trans = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0] @ trans
        if np.random.rand() < self.scale_p:
            scale = np.eye(4, dtype=np.float32)
            scale[0, 0] = np.random.uniform(*self.scale)
            scale[1, 1] = np.random.uniform(*self.scale)
            scale[2, 2] = np.random.uniform(*self.scale)
            trans = scale @ trans
        asset.transform(trans)


@dataclass(frozen=True)
class AugmentPatch(Augment):
    """以随机种子点为中心切 KNN patch, 并在 noisy/clean 之间做线性插值。

    产出 asset.meta = {pc_noisy, pc_clean, pc_mix}, 都是 (num_patches, patch_size, 3),
    且都已减去种子点坐标(patch 中心化), 便于模型学习位移。
    """
    patch_size: int
    num_patches: int
    train_cvm_network: bool = False  # 保留字段以兼容旧 config, 当前未使用

    def apply(self, asset: Asset) -> None:
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        if pc is None or pc_noisy is None:
            raise ValueError("AugmentPatch needs clean and noisy sampled points")

        n = pc_noisy.shape[0]
        if n == 0:
            raise ValueError("cannot build patches from an empty point cloud")

        patch_size = min(self.patch_size, n)
        num_patches = min(self.num_patches, n)
        seed_idx = np.random.permutation(n)[:num_patches]
        seed_noisy = pc_noisy[seed_idx]

        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_noisy, k=patch_size)
        if patch_size == 1:
            nn_idx = nn_idx[:, None]

        pat_a = pc_noisy[nn_idx]   # noisy patch
        pat_b = pc[nn_idx]         # clean patch

        # 在 noisy->clean 方向上随机插值, 构造训练用的混合点云
        t = np.random.uniform(1e-8, 1.0, size=(num_patches, patch_size, 1))
        pat_t = t * pat_b + (1.0 - t) * pat_a
        seed_t = t[:, 0:1, :] * pc[seed_idx][:, None, :] + (1.0 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]

        asset.meta = {
            "pc_noisy": pat_a - seed_t,
            "pc_clean": pat_b - seed_t,
            "pc_mix": pat_t - seed_t,
        }


@dataclass(frozen=True)
class AugmentScorePerturb(Augment):
    """Denoising Score Matching 专用: 在 clean 点上加高斯噪声并计算得分目标。

    与 AugmentPatch 的区别:
    - AugmentPatch 在 noisy/clean 间线性插值, 监督"总位移"
    - AugmentScorePerturb 直接加高斯扰动, 监督"得分" -ε/σ

    每个 patch 采样一个 σ, 产出:
      pc_noisy   = clean + σε (加噪后的点, patch 中心化)
      pc_clean   = 原始 clean 点 (patch 中心化)
      score_target = -ε/σ (得分匹配目标)
      sigma      = 该 patch 的噪声级别
    """
    patch_size: int
    num_patches: int
    sigma_min: float
    sigma_max: float

    def apply(self, asset: Asset) -> None:
        pc = asset.sampled_vertices
        if pc is None:
            raise ValueError("AugmentScorePerturb needs sampled_vertices (clean)")

        n = pc.shape[0]
        if n == 0:
            raise ValueError("cannot build patches from an empty point cloud")

        patch_size = min(self.patch_size, n)
        num_patches = min(self.num_patches, n)
        seed_idx = np.random.permutation(n)[:num_patches]
        seeds = pc[seed_idx]

        tree = cKDTree(pc)
        _, nn_idx = tree.query(seeds, k=patch_size)
        if patch_size == 1:
            nn_idx = nn_idx[:, None]

        pat_clean = pc[nn_idx]  # (num_patches, patch_size, 3)

        # 每个 patch 采样一个 sigma
        sigma = np.random.uniform(
            self.sigma_min, self.sigma_max, size=(num_patches, 1, 1)
        ).astype(np.float32)

        # 高斯扰动: x_noisy = x_clean + sigma * eps
        eps = np.random.randn(num_patches, patch_size, 3).astype(np.float32)
        pat_noisy = pat_clean + sigma * eps

        # 得分匹配目标: score = -eps / sigma (指向 clean 的方向)
        score_target = -eps / sigma

        # patch 中心化: 减去种子点
        seed_centers = pc[seed_idx][:, None, :]

        asset.meta = {
            "pc_noisy": pat_noisy - seed_centers,
            "pc_clean": pat_clean - seed_centers,
            "score_target": score_target,
            "sigma": sigma.squeeze(-1),  # (num_patches, 1)
        }


# ===================== 工厂 =====================

_AUGMENT_MAP: Dict[str, type] = {
    "sample": AugmentSample,
    "normalize_pc": AugmentNormalizePC,
    "add_noise": AugmentAddNoise,
    "linear": AugmentLinear,
    "patch": AugmentPatch,
    "score_perturb": AugmentScorePerturb,
}


def get_augments(*configs) -> List[Augment]:
    """把 YAML 里 augments 列表解析成 Augment 实例列表。"""
    augments: List[Augment] = []
    for i, cfg in enumerate(configs):
        target = cfg.get("__target__")
        if target is None:
            raise ValueError(f"missing __target__ in augment at position {i}")
        if target not in _AUGMENT_MAP:
            raise ValueError(f"unsupported augment target: {target}")
        copied = deepcopy(cfg)
        del copied["__target__"]
        augments.append(_AUGMENT_MAP[target].parse(**copied))
    return augments


def register_augment(name: str, cls: type) -> None:
    """注册新 augment, 供外部扩展。"""
    _AUGMENT_MAP[name] = cls
