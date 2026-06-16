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
    model_config: Dict
    transform_config: Dict

    @abstractmethod
    def __init__(self, model_config, transform_config):
        super().__init__()
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

    def is_predict(self) -> bool:
        return self._is_predict

    def set_predict(self, is_predict: bool) -> None:
        self._is_predict = is_predict

    @final
    def _process_fn(self, batch: List[Asset]) -> List[Dict]:
        processed = self.process_fn(batch)
        if not self.training:
            for index, asset in enumerate(batch):
                non = processed[index].get("non", {})
                non["asset"] = deepcopy(asset)
                processed[index]["non"] = non
        return processed

    @abstractmethod
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        raise NotImplementedError()

    def compile_model(self) -> None:
        pass

    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ) -> None:
        payload = {
            "model": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device,
        strict: bool = True,
    ) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        self.load_state_dict(state_dict, strict=strict)
        return checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}

    def get_train_transform(self) -> Optional[Transform]:
        config = self.transform_config.get("train_transform")
        return None if config is None else Transform.parse(**config)

    def get_validate_transform(self) -> Optional[Transform]:
        config = self.transform_config.get("validate_transform")
        return None if config is None else Transform.parse(**config)

    def get_predict_transform(self) -> Optional[Transform]:
        config = self.transform_config.get("predict_transform")
        return None if config is None else Transform.parse(**config)

    def predict_step(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError()
