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
    worker_seed = torch.initial_seed() % (2**32)
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
                datapath=datapath,
            )
            for name, datapath in self.datapath.split_by_cls().items()
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

        self.train_datapath = (
            None if train_dataset_config is None else train_dataset_config.datapath
        )
        self.validate_datapath = (
            None
            if validate_dataset_config is None
            else {
                name: config.datapath
                for name, config in validate_dataset_config.items()
            }
        )
        self.predict_datapath = (
            None
            if predict_dataset_config is None
            else {
                name: config.datapath
                for name, config in predict_dataset_config.items()
            }
        )

    @staticmethod
    def _identity_collate(batch: List[Asset]) -> List[Asset]:
        return batch

    @staticmethod
    def _create_single_dataloader(
        dataset: Dataset,
        config: DatasetConfig,
    ) -> DataLoader:
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
        if (
            self.train_transform is None
            or self.train_dataset_config is None
            or self.train_datapath is None
        ):
            return None
        dataset = PCDataset(
            data=self.train_datapath.get_data(),
            transform=self.train_transform,
            name="train",
        )
        return self._create_single_dataloader(dataset, self.train_dataset_config)

    def validate_dataloader(self):
        if (
            self.validate_transform is None
            or self.validate_dataset_config is None
            or self.validate_datapath is None
        ):
            return None
        return {
            name: self._create_single_dataloader(
                PCDataset(
                    data=datapath.get_data(),
                    transform=self.validate_transform,
                    name=f"validate-{name}",
                ),
                self.validate_dataset_config[name],
            )
            for name, datapath in self.validate_datapath.items()
        }

    def predict_dataloader(self):
        if (
            self.predict_transform is None
            or self.predict_dataset_config is None
            or self.predict_datapath is None
        ):
            return None
        return {
            name: self._create_single_dataloader(
                PCDataset(
                    data=datapath.get_data(),
                    transform=self.predict_transform,
                    name=f"predict-{name}",
                ),
                self.predict_dataset_config[name],
            )
            for name, datapath in self.predict_datapath.items()
        }

    @staticmethod
    def _as_tensor(value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        raise TypeError(f"cannot convert type {type(value)} to tensor")

    def prepare_batch(
        self,
        assets: List[Asset],
        device: torch.device,
    ) -> Union[List[Asset], Dict]:
        if self.debug:
            return assets
        if self.process_fn is None:
            raise ValueError("missing data processing function")

        processed_batch = self.process_fn(assets)
        if not processed_batch:
            raise ValueError("process_fn returned an empty batch")

        tensors_stack: Dict[str, List[torch.Tensor]] = {}
        tensors_cat: Dict[str, List[torch.Tensor]] = {}
        non_tensors: Dict[str, List] = {}
        seen = set()

        def register(name: str) -> None:
            if name in seen:
                raise ValueError(f"multiple keys found: {name}")
            seen.add(name)

        for key, value in processed_batch[0].items():
            if key == "cat":
                if not isinstance(value, dict):
                    raise TypeError("cat must contain a dict")
                for inner_key in value:
                    register(inner_key)
                    tensors_cat[inner_key] = [
                        self._as_tensor(item["cat"][inner_key], device)
                        for item in processed_batch
                    ]
            elif key == "non":
                if not isinstance(value, dict):
                    raise TypeError("non must contain a dict")
                for inner_key in value:
                    register(inner_key)
                    non_tensors[inner_key] = [
                        item["non"][inner_key] for item in processed_batch
                    ]
            else:
                register(key)
                tensors_stack[key] = [
                    self._as_tensor(item[key], device) for item in processed_batch
                ]

        collated_stack = {
            key: torch.stack(values, dim=0) for key, values in tensors_stack.items()
        }
        collated_cat = {
            key: torch.cat(values, dim=1) for key, values in tensors_cat.items()
        }
        return {**collated_stack, **collated_cat, **non_tensors}
