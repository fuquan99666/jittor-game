from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

import os
import torch
from tqdm import tqdm

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec


def get_optimizer(optimizer_config, model: ModelSpec) -> torch.optim.Optimizer:
    config = deepcopy(optimizer_config)
    target = config.pop("__target__")
    mapping = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    if target not in mapping:
        raise ValueError(f"unsupported optimizer: {target}")
    return mapping[target](model.parameters(), **config)


class DummyWriter:
    def write(
        self,
        batch,
        prediction: List[Dict],
        dataset_module: Optional[PCDatasetModule] = None,
    ) -> None:
        pass


class DummySystem:
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        device: torch.device,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
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
        self._validation_loss = defaultdict(list)

    def forward(self, batch: Dict, validate: bool = False) -> torch.Tensor:
        loss_dict = self.model.training_step(batch)
        if not isinstance(loss_dict, dict):
            raise TypeError("training_step must return a loss dictionary")
        if self.loss_config is None:
            raise ValueError("loss config is missing")

        loss_sum = torch.zeros((), device=self.device)
        for name, loss in loss_dict.items():
            if name not in self.loss_config:
                raise ValueError(f"unspecified loss: {name}")
            weight = float(self.loss_config[name])
            if weight > 0:
                loss_sum = loss_sum + weight * loss

            if validate:
                assets: List[Asset] = batch["asset"]
                cls_name = assets[0].cls
                self._validation_loss[f"val/{cls_name}_{name}"].append(
                    float(loss.detach().cpu())
                )

        if validate:
            assets = batch["asset"]
            cls_name = assets[0].cls
            self._validation_loss[f"val/{cls_name}_loss_sum"].append(
                float(loss_sum.detach().cpu())
            )
        return loss_sum

    def train(self) -> None:
        if self.optimizer is None:
            raise ValueError("optimizer is required for training")

        self.model.set_predict(False)
        train_dataloader = self.dataset_module.train_dataloader()
        if train_dataloader is None:
            raise ValueError("train dataloader is unavailable")

        validate_dataloaders = self.dataset_module.validate_dataloader()

        for epoch in range(self.epochs):
            self.model.train()
            progress = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for raw_assets in progress:
                batch = self.dataset_module.prepare_batch(raw_assets, self.device)
                if not isinstance(batch, dict):
                    raise TypeError("training batch must be a dictionary")

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.forward(batch, validate=False)
                loss.backward()
                self.optimizer.step()
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.6f}")

            if validate_dataloaders is not None:
                self.model.eval()
                self._validation_loss = defaultdict(list)
                with torch.no_grad():
                    for name, dataloader in validate_dataloaders.items():
                        progress = tqdm(
                            dataloader,
                            desc=f"Validate {epoch} {name}",
                        )
                        for raw_assets in progress:
                            batch = self.dataset_module.prepare_batch(
                                raw_assets,
                                self.device,
                            )
                            if not isinstance(batch, dict):
                                raise TypeError("validation batch must be a dictionary")
                            loss = self.forward(batch, validate=True)
                            progress.set_postfix(
                                loss=f"{float(loss.detach().cpu()):.6f}"
                            )

                for key, values in sorted(self._validation_loss.items()):
                    if values:
                        print(f"{key}: {sum(values) / len(values):.6f}")

            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.ckpt_save_dir,
                f"{self.ckpt_save_name}_{epoch}.pt",
            )
            self.model.save_checkpoint(
                checkpoint_path,
                optimizer=self.optimizer,
                epoch=epoch,
            )

    @torch.no_grad()
    def predict(self) -> None:
        self.model.set_predict(True)
        self.model.eval()
        predict_dataloaders = self.dataset_module.predict_dataloader()
        if predict_dataloaders is None:
            raise ValueError("predict dataloader is unavailable")
        if not isinstance(predict_dataloaders, dict):
            predict_dataloaders = {"predict": predict_dataloaders}

        for name, dataloader in predict_dataloaders.items():
            progress = tqdm(dataloader, desc=f"Predict {name}")
            for raw_assets in progress:
                batch = self.dataset_module.prepare_batch(raw_assets, self.device)
                if not isinstance(batch, dict):
                    raise TypeError("prediction batch must be a dictionary")
                output = self.model.predict_step(batch)
                if self.writer is not None:
                    self.writer.write(
                        batch,
                        output,
                        dataset_module=self.dataset_module,
                    )
