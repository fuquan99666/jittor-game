"""Model 基类: 管理 config、checkpoint、train/predict 切换、process_fn 协议。

子类需要实现:
  - process_fn(batch) -> List[dict]: 把 Asset 列表转成 tensor dict
  - training_step(batch) -> {"loss": tensor}
  - predict_step(batch) -> List[dict]
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, final

import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf
import torch
from torch import nn

from ..data.asset import Asset
from ..data.transform import Transform


@dataclass
class ModelInput:
    asset: Asset
    tokens: Optional[ndarray] = None


class ModelSpec(nn.Module, ABC):
    def __init__(self, model_config, transform_config):
        super().__init__()
        # 统一成 dict 存储, OmegaConf 也接受
        self.model_config = (
            OmegaConf.to_container(model_config, resolve=True)
            if not isinstance(model_config, dict)
            else deepcopy(model_config)
        )
        self.transform_config = (
            OmegaConf.to_container(transform_config, resolve=True)
            if not isinstance(transform_config, dict)
            else deepcopy(transform_config)
        )
        self._is_predict = False

    # ---------- train/predict 模式 ----------

    def is_predict(self) -> bool:
        return self._is_predict

    def set_predict(self, is_predict: bool) -> None:
        self._is_predict = is_predict

    # ---------- process_fn 协议 ----------

    @final
    def process_fn_wrapper(self, batch: List[Asset]) -> List[Dict]:
        """包装子类的 process_fn: 非训练模式下把原始 Asset 挂到 non["asset"],
        供 writer 恢复输出路径。"""
        processed = self.process_fn(batch)
        if not self.training:
            for i, asset in enumerate(batch):
                non = processed[i].get("non", {})
                non["asset"] = deepcopy(asset)
                processed[i]["non"] = non
        return processed

    @abstractmethod
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        raise NotImplementedError()

    # ---------- transform 由 model 派发 ----------
    # 这样 model 可以根据自身需要覆盖 transform_config 里的某些项

    def get_train_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("train_transform")
        return None if cfg is None else Transform.parse(**cfg)

    def get_validate_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("validate_transform")
        return None if cfg is None else Transform.parse(**cfg)

    def get_predict_transform(self) -> Optional[Transform]:
        cfg = self.transform_config.get("predict_transform")
        return None if cfg is None else Transform.parse(**cfg)

    # ---------- checkpoint ----------

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ) -> None:
        payload = {"model": self.state_dict(), "epoch": epoch}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: str,
        device: torch.device,
        strict: bool = True,
    ) -> Dict:
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model", ckpt)
        self.load_state_dict(state, strict=strict)
        return ckpt if isinstance(ckpt, dict) else {"model": ckpt}

    @abstractmethod
    def training_step(self, batch: Dict) -> Dict:
        raise NotImplementedError()

    def predict_step(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError()
