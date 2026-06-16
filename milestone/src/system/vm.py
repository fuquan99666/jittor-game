from typing import Dict, List, Optional

import numpy as np
import os
import torch

from .spec import DummySystem, DummyWriter
from ..data.asset import Exporter


class VMWriter(DummyWriter):
    def __init__(
        self,
        save_dir: str = "results",
        save_name: str = "denoised",
        output_format: str = "npy",
    ):
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format

    def write(self, batch, prediction: List[Dict], dataset_module=None) -> None:
        assets = batch["asset"]
        for index, asset in enumerate(assets):
            if asset.path is None:
                raise ValueError("asset path is missing")

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

            denoised = prediction[index]["pc_denoised"]
            if isinstance(denoised, torch.Tensor):
                denoised_np = denoised.detach().cpu().numpy()
            else:
                denoised_np = np.asarray(denoised)
            denoised_np = denoised_np.astype(np.float32, copy=False)

            if self.output_format == "npy":
                np.save(
                    os.path.join(output_dir, f"{self.save_name}.npy"),
                    denoised_np,
                )
            elif self.output_format == "obj":
                Exporter.export_obj(
                    denoised_np,
                    os.path.join(output_dir, f"{self.save_name}.obj"),
                )
            else:
                raise ValueError(f"unsupported output format: {self.output_format}")


class VMSystem(DummySystem):
    pass
