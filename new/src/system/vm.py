"""VM 专用 system 与 writer。

VMSystem 目前只是 BaseSystem 的别名(无额外逻辑), 保留类是为了日后扩展
(如自定义训练逻辑、EMA、lr schedule 等)。

VMWriter 负责把去噪结果写成 .npy/.obj, 路径从 asset.path 恢复,
镜像测试集的目录结构。
"""
from typing import Dict, List, Optional

import numpy as np
import os
import torch

from .spec import BaseSystem, BaseWriter
from ..data.asset import Exporter


class VMWriter(BaseWriter):
    def __init__(
        self,
        save_dir: str = "results",
        save_name: str = "denoised",
        output_format: str = "npy",
    ):
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format

    def write(self, batch: Dict, prediction: List[Dict], dataset_module=None) -> None:
        assets = batch["asset"]
        for i, asset in enumerate(assets):
            if asset.path is None:
                raise ValueError("asset path is missing")

            # 从 asset.path 恢复相对目录, 去掉 .. 前缀, 镜像测试集布局
            source_dir = os.path.dirname(os.path.normpath(asset.path))
            if os.path.isabs(source_dir):
                try:
                    source_dir = os.path.relpath(source_dir, os.getcwd())
                except ValueError:
                    source_dir = os.path.basename(source_dir)
            while source_dir.startswith(".." + os.sep):
                source_dir = source_dir[3:]
            source_dir = source_dir.lstrip("/\\")

            output_dir = os.path.join(self.save_dir, source_dir)
            os.makedirs(output_dir, exist_ok=True)

            denoised = prediction[i]["pc_denoised"]
            if isinstance(denoised, torch.Tensor):
                denoised = denoised.detach().cpu().numpy()
            denoised = np.asarray(denoised, dtype=np.float32)

            if self.output_format == "npy":
                np.save(os.path.join(output_dir, f"{self.save_name}.npy"), denoised)
            elif self.output_format == "obj":
                Exporter.export_obj(denoised, os.path.join(output_dir, f"{self.save_name}.obj"))
            else:
                raise ValueError(f"unsupported output format: {self.output_format}")


class VMSystem(BaseSystem):
    pass
