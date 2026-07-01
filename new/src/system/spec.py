"""System 基类: 编排 model + dataloader + optimizer + loss + writer 的训练/推理循环。

命名说明: 原版叫 DummySystem/DummyWriter, 容易误解成"假的"。这里改名 Base 更清晰。
"""
from collections import defaultdict
from typing import Dict, List, Optional

import os
import torch
from tqdm import tqdm

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec


def get_optimizer(optimizer_config, model: ModelSpec) -> torch.optim.Optimizer:
    """优化器工厂: optimizer.__target__ 取 adam 或 sgd。"""
    from copy import deepcopy
    cfg = deepcopy(optimizer_config)
    target = cfg.pop("__target__")
    mapping = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
    if target not in mapping:
        raise ValueError(f"unsupported optimizer: {target}")
    return mapping[target](model.parameters(), **cfg)


class BaseWriter:
    """预测结果写入器基类(默认什么都不做)。"""

    def write(
        self,
        batch: Dict,
        prediction: List[Dict],
        dataset_module: Optional[PCDatasetModule] = None,
    ) -> None:
        pass


class BaseSystem:
    """训练/推理循环的主体。

    一个 epoch 的流程:
      1. 遍历 train_dataloader, 每个 batch -> prepare_batch -> forward -> backward
      2. (可选) 遍历 validate_dataloader, 算各类 loss 并打印
      3. 保存 checkpoint_<epoch>.pt (0-indexed)
    """

    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        device: torch.device,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[BaseWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.device = device
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        trainer_config = trainer_config or {}
        self.epochs = int(trainer_config.get("epochs", 1))
        self.optimizer = (
            get_optimizer(optimizer_config, model)
            if optimizer_config is not None
            else None
        )
        self._val_loss: Dict[str, List[float]] = defaultdict(list)

    def _compute_loss(self, batch: Dict, validate: bool = False) -> torch.Tensor:
        """跑 model.training_step, 按 loss_config 加权汇总。"""
        loss_dict = self.model.training_step(batch)
        if not isinstance(loss_dict, dict):
            raise TypeError("training_step must return a loss dictionary")
        if self.loss_config is None:
            raise ValueError("loss config is missing")

        total = torch.zeros((), device=self.device)
        for name, loss in loss_dict.items():
            if name not in self.loss_config:
                raise ValueError(f"unspecified loss: {name}")
            weight = float(self.loss_config[name])
            if weight > 0:
                total = total + weight * loss
            if validate:
                assets: List[Asset] = batch["asset"]
                cls_name = assets[0].cls
                self._val_loss[f"val/{cls_name}_{name}"].append(float(loss.detach().cpu()))

        if validate:
            assets = batch["asset"]
            cls_name = assets[0].cls
            self._val_loss[f"val/{cls_name}_loss_sum"].append(float(total.detach().cpu()))
        return total

    def train(self) -> None:
        if self.optimizer is None:
            raise ValueError("optimizer is required for training")

        self.model.set_predict(False)
        train_loader = self.dataset_module.train_dataloader()
        if train_loader is None:
            raise ValueError("train dataloader is unavailable")
        val_loaders = self.dataset_module.validate_dataloader()

        for epoch in range(self.epochs):
            self.model.train()
            for raw_assets in tqdm(train_loader, desc=f"Epoch {epoch}"):
                batch = self.dataset_module.prepare_batch(raw_assets, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._compute_loss(batch, validate=False)
                loss.backward()
                self.optimizer.step()

            if val_loaders is not None:
                self._validate(val_loaders)

            self._save_checkpoint(epoch)

    def _validate(self, val_loaders) -> None:
        self.model.eval()
        self._val_loss = defaultdict(list)
        with torch.no_grad():
            for name, loader in val_loaders.items():
                for raw_assets in tqdm(loader, desc=f"Validate {name}"):
                    batch = self.dataset_module.prepare_batch(raw_assets, self.device)
                    self._compute_loss(batch, validate=True)
        for key, values in sorted(self._val_loss.items()):
            if values:
                print(f"{key}: {sum(values) / len(values):.6f}")

    def _save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        path = os.path.join(self.ckpt_save_dir, f"{self.ckpt_save_name}_{epoch}.pt")
        self.model.save_checkpoint(path, optimizer=self.optimizer, epoch=epoch)

    @torch.no_grad()
    def predict(self) -> None:
        self.model.set_predict(True)
        self.model.eval()
        loaders = self.dataset_module.predict_dataloader()
        if loaders is None:
            raise ValueError("predict dataloader is unavailable")
        if not isinstance(loaders, dict):
            loaders = {"predict": loaders}

        for name, loader in loaders.items():
            for raw_assets in tqdm(loader, desc=f"Predict {name}"):
                batch = self.dataset_module.prepare_batch(raw_assets, self.device)
                output = self.model.predict_step(batch)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
