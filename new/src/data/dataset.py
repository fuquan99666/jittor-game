"""Dataset 与 DatasetModule。

- PCDataset: torch Dataset, __getitem__ 加载文件 -> 跑 transform -> 返回 Asset
- PCDatasetModule: 管理三种 dataloader(train/validate/predict), 并负责把 Asset 列表
  collate 成 model 能吃的 tensor batch(stack/cat/non 三类字段)。
"""
from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .asset import Asset
from .datapath import Datapath, LazyAsset
from .spec import ConfigSpec
from .transform import Transform


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 的随机种子初始化, 保证多 worker 可复现。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DatasetConfig(ConfigSpec):
    shuffle: bool
    batch_size: int
    num_workers: int
    datapath: Datapath

    @classmethod
    def parse(cls, **kwargs) -> "DatasetConfig":
        cls.check_keys(kwargs)
        return DatasetConfig(
            shuffle=kwargs.get("shuffle", False),
            batch_size=kwargs.get("batch_size", 1),
            num_workers=kwargs.get("num_workers", 0),
            datapath=Datapath.parse(**kwargs["datapath"]),
        )

    def split_by_cls(self) -> Dict[Optional[str], "DatasetConfig"]:
        return {
            name: DatasetConfig(
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                datapath=dp,
            )
            for name, dp in self.datapath.split_by_cls().items()
        }


class PCDataset(Dataset):
    def __init__(
        self,
        data: List[LazyAsset],
        transform: Transform,
        name: Optional[str] = None,
    ):
        self.data = data
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Asset:
        asset = self.data[index].load()
        self.transform.apply(asset)
        return asset


class PCDatasetModule:
    """聚合 train/validate/predict 三套 dataloader 与 batch collate 逻辑。"""

    def __init__(
        self,
        process_fn: Optional[Callable[[List[Asset]], List[Dict]]] = None,
        train_dataset_config: Optional[DatasetConfig] = None,
        validate_dataset_config: Optional[Dict[Optional[str], DatasetConfig]] = None,
        predict_dataset_config: Optional[Dict[Optional[str], DatasetConfig]] = None,
        train_transform: Optional[Transform] = None,
        validate_transform: Optional[Transform] = None,
        predict_transform: Optional[Transform] = None,
        debug: bool = False,
    ):
        self.process_fn = process_fn
        self.train_dataset_config = train_dataset_config
        self.validate_dataset_config = validate_dataset_config
        self.predict_dataset_config = predict_dataset_config
        self.train_transform = train_transform
        self.validate_transform = validate_transform
        self.predict_transform = predict_transform
        self.debug = debug

    # ---------- dataloader 构造 ----------

    @staticmethod
    def _identity_collate(batch: List[Asset]) -> List[Asset]:
        # Asset 大小不固定, 不能用默认 collate, 直接返回列表
        return batch

    @staticmethod
    def _make_dataloader(dataset: Dataset, config: DatasetConfig) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            drop_last=False,
            collate_fn=PCDatasetModule._identity_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
            worker_init_fn=seed_worker if config.num_workers > 0 else None,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_transform is None or self.train_dataset_config is None:
            return None
        dp = self.train_dataset_config.datapath
        dataset = PCDataset(dp.get_data(), self.train_transform, name="train")
        return self._make_dataloader(dataset, self.train_dataset_config)

    def validate_dataloader(self):
        if self.validate_transform is None or self.validate_dataset_config is None:
            return None
        return {
            name: self._make_dataloader(
                PCDataset(cfg.datapath.get_data(), self.validate_transform, name=f"validate-{name}"),
                cfg,
            )
            for name, cfg in self.validate_dataset_config.items()
        }

    def predict_dataloader(self):
        if self.predict_transform is None or self.predict_dataset_config is None:
            return None
        return {
            name: self._make_dataloader(
                PCDataset(cfg.datapath.get_data(), self.predict_transform, name=f"predict-{name}"),
                cfg,
            )
            for name, cfg in self.predict_dataset_config.items()
        }

    # ---------- batch collate ----------

    @staticmethod
    def _as_tensor(value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=torch.float32, non_blocking=True)
        raise TypeError(f"cannot convert type {type(value)} to tensor")

    def prepare_batch(
        self,
        assets: List[Asset],
        device: torch.device,
    ) -> Union[List[Asset], Dict]:
        """把 Asset 列表转成 tensor batch。

        model.process_fn 把每个 Asset 变成 dict, 其中:
          - 顶层标量/张量字段 -> torch.stack 成 (B, ...)
          - "cat" 子 dict     -> torch.cat  在 dim=1 拼接(用于 patch 维度合并)
          - "non" 子 dict     -> 原样保留(非张量, 如 asset 引用)
        """
        if self.debug:
            return assets
        if self.process_fn is None:
            raise ValueError("missing data processing function")

        processed = self.process_fn(assets)
        if not processed:
            raise ValueError("process_fn returned an empty batch")

        stack_fields: Dict[str, List[torch.Tensor]] = {}
        cat_fields: Dict[str, List[torch.Tensor]] = {}
        non_fields: Dict[str, List] = {}
        seen = set()

        def register(name: str) -> None:
            if name in seen:
                raise ValueError(f"multiple keys found: {name}")
            seen.add(name)

        for key, value in processed[0].items():
            if key == "cat":
                for inner in value:
                    register(inner)
                    cat_fields[inner] = [self._as_tensor(item["cat"][inner], device) for item in processed]
            elif key == "non":
                for inner in value:
                    register(inner)
                    non_fields[inner] = [item["non"][inner] for item in processed]
            else:
                register(key)
                stack_fields[key] = [self._as_tensor(item[key], device) for item in processed]

        collated = {k: torch.stack(v, dim=0) for k, v in stack_fields.items()}
        collated.update({k: torch.cat(v, dim=1) for k, v in cat_fields.items()})
        collated.update(non_fields)
        return collated
