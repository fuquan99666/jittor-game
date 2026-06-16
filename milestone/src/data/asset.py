from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy import ndarray
import os


@dataclass
class Asset:
    path: Optional[str] = None
    cls: Optional[str] = None
    vertices: Optional[ndarray] = None
    faces: Optional[ndarray] = None
    sampled_vertices: Optional[ndarray] = None
    sampled_vertices_noisy: Optional[ndarray] = None
    meta: Optional[Dict] = None

    def transform(self, trans: ndarray) -> None:
        """Apply a 4x4 affine transform to all point-coordinate fields."""

        def _apply(v: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

        if self.vertices is not None:
            self.vertices = _apply(self.vertices)
        if self.sampled_vertices is not None:
            self.sampled_vertices = _apply(self.sampled_vertices)
        if self.sampled_vertices_noisy is not None:
            self.sampled_vertices_noisy = _apply(self.sampled_vertices_noisy)


class Exporter:
    @classmethod
    def _safe_make_dir(cls, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    @classmethod
    def export_obj(cls, vertices, path: str, precision: int = 6) -> None:
        lines = [
            f"v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n"
            for v in vertices
        ]
        cls._safe_make_dir(path)
        with open(path, "w", encoding="utf-8") as file:
            file.writelines(lines)
