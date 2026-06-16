from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups


@dataclass(frozen=True)
class Augment(ConfigSpec, ABC):
    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> "Augment":
        raise NotImplementedError()

    @abstractmethod
    def apply(self, asset: Asset, **kwargs) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class AugmentSample(Augment):
    num_samples: int
    num_vertex_samples: int = 0

    @classmethod
    def parse(cls, **kwargs) -> "AugmentSample":
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)

    def apply(self, asset: Asset, **kwargs) -> None:
        if asset.vertices is None or asset.faces is None:
            raise ValueError("mesh vertices/faces are required by AugmentSample")
        sampled_vertices, _, _, _ = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices


@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    @classmethod
    def parse(cls, **kwargs) -> "AugmentNormalizePC":
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)

    def apply(self, asset: Asset, **kwargs) -> None:
        pc = asset.sampled_vertices
        if pc is None:
            raise ValueError("sampled_vertices is required by AugmentNormalizePC")
        center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1)).max()
        asset.sampled_vertices = pc / max(float(scale), 1e-12)


@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    noise_std_min: float
    noise_std_max: float

    @classmethod
    def parse(cls, **kwargs) -> "AugmentAddNoise":
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)

    def apply(self, asset: Asset, **kwargs) -> None:
        pc = asset.sampled_vertices
        if pc is None:
            raise ValueError("sampled_vertices is required by AugmentAddNoise")
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0.0, noise_std, size=pc.shape)
        asset.sampled_vertices_noisy = pc + noise


@dataclass(frozen=True)
class AugmentLinear(Augment):
    scale: Tuple[float, float] = (1.0, 1.0)
    rotate_x_range: Tuple[float, float] = (0.0, 0.0)
    rotate_y_range: Tuple[float, float] = (0.0, 0.0)
    rotate_z_range: Tuple[float, float] = (0.0, 0.0)
    scale_p: float = 0.0
    rotate_p: float = 0.0

    @classmethod
    def parse(cls, **kwargs) -> "AugmentLinear":
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)

    def apply(self, asset: Asset, **kwargs) -> None:
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            rotation = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = rotation @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.eye(4, dtype=np.float32)
            scale[0, 0] = np.random.uniform(*self.scale)
            scale[1, 1] = np.random.uniform(*self.scale)
            scale[2, 2] = np.random.uniform(*self.scale)
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)


@dataclass(frozen=True)
class AugmentPatch(Augment):
    patch_size: int
    num_patches: int
    train_cvm_network: bool

    @classmethod
    def parse(cls, **kwargs) -> "AugmentPatch":
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)

    def apply(self, asset: Asset, **kwargs) -> None:
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        if pc is None or pc_noisy is None:
            raise ValueError("clean and noisy sampled points are required by AugmentPatch")

        point_count = pc_noisy.shape[0]
        if point_count == 0:
            raise ValueError("cannot build patches from an empty point cloud")

        patch_size = min(self.patch_size, point_count)
        num_patches = min(self.num_patches, point_count)
        seed_idx = np.random.permutation(point_count)[:num_patches]
        seed_points = pc_noisy[seed_idx]

        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=patch_size)
        if patch_size == 1:
            nn_idx = nn_idx[:, None]

        pat_a = pc_noisy[nn_idx]
        pat_b = pc[nn_idx]

        t = np.random.uniform(
            1e-8,
            1.0,
            size=(num_patches, patch_size, 1),
        )
        pat_t = t * pat_b + (1.0 - t) * pat_a
        seed_points_t = (
            t[:, 0:1, :] * pc[seed_idx][:, None, :]
            + (1.0 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
        )

        if asset.meta is None:
            asset.meta = {}
        asset.meta["pc_noisy"] = pat_a - seed_points_t
        asset.meta["pc_clean"] = pat_b - seed_points_t
        asset.meta["pc_mix"] = pat_t - seed_points_t


def get_augments(*args) -> List[Augment]:
    mapping: Dict[str, type[Augment]] = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    augments: List[Augment] = []
    for index, config in enumerate(args):
        target = config.get("__target__")
        if target is None:
            raise ValueError(f"missing __target__ in augment at position {index}")
        if target not in mapping:
            raise ValueError(f"unsupported augment target: {target}")
        copied = deepcopy(config)
        del copied["__target__"]
        augments.append(mapping[target].parse(**copied))
    return augments
