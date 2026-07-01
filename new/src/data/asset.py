from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy import ndarray
import os


@dataclass
class Asset:
    """贯穿 dataloader / augment / model 的统一数据载体。

    流程: loader 填充 vertices/faces(网格) 或 sampled_vertices_noisy(点云)
    -> augment 就地修改 sampled_vertices / sampled_vertices_noisy / meta
    -> model.process_fn 把 Asset 转成 tensor dict

    meta 字段用于存放 augment 之间传递的中间结果(如 patch 切分出的 pc_noisy/pc_clean/pc_mix)。
    """

    path: Optional[str] = None
    cls: Optional[str] = None
    vertices: Optional[ndarray] = None
    faces: Optional[ndarray] = None
    sampled_vertices: Optional[ndarray] = None
    sampled_vertices_noisy: Optional[ndarray] = None
    meta: Optional[Dict] = field(default_factory=dict)

    def transform(self, trans: ndarray) -> None:
        """对全部点坐标字段施加 4x4 仿射变换(就地)。"""
        if self.vertices is not None:
            self.vertices = _apply_affine(self.vertices, trans)
        if self.sampled_vertices is not None:
            self.sampled_vertices = _apply_affine(self.sampled_vertices, trans)
        if self.sampled_vertices_noisy is not None:
            self.sampled_vertices_noisy = _apply_affine(self.sampled_vertices_noisy, trans)


def _apply_affine(points: ndarray, trans: ndarray) -> ndarray:
    return np.matmul(points, trans[:3, :3].transpose()) + trans[:3, 3]


class Exporter:
    """把点云写成 .obj 文件。"""

    @staticmethod
    def export_obj(vertices: ndarray, path: str, precision: int = 6) -> None:
        # 这里做 Y/Z 轴交换 + Y 取反, 是因为原始 ShapeNet 网格的坐标系约定:
        # trimesh 读出的 (x,y,z) 在写入 obj 时需要转成 (x,z,-y) 才与多数 viewer 一致。
        lines = [
            f"v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n"
            for v in vertices
        ]
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
