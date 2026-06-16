from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from random import shuffle
from typing import Dict, List, Optional

import numpy as np
import os
import trimesh

from .asset import Asset
from .spec import ConfigSpec


@dataclass
class LazyAsset(ABC):
    path: str
    cls: Optional[str] = None

    @abstractmethod
    def load(self) -> Asset:
        raise NotImplementedError()


@dataclass
class ObjLazyAsset(LazyAsset):
    def load(self) -> Asset:
        mesh = trimesh.load(self.path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return Asset(
            path=self.path,
            cls=self.cls,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int64),
        )


@dataclass
class NpyLazyAsset(LazyAsset):
    def load(self) -> Asset:
        pc = np.load(self.path).astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"expected point cloud shape (N, 3), got {pc.shape}: {self.path}")
        return Asset(
            path=self.path,
            cls=self.cls,
            sampled_vertices_noisy=pc,
        )


@dataclass
class Datapath(ConfigSpec):
    filepaths: List[str]
    input_dataset_dir: str = ""
    cls_name: Optional[List[str]] = None
    cls_bias: Optional[List[int]] = None
    cls_length: Optional[List[int]] = None
    num_files: Optional[int] = None
    use_prob: bool = False
    cls_weight: Optional[List[float]] = None
    loader: type[LazyAsset] = ObjLazyAsset
    data_name: Optional[str] = None
    ignore_check: bool = False

    @classmethod
    def parse(cls, **kwargs) -> "Datapath":
        mapping = {
            None: ObjLazyAsset,
            "obj": ObjLazyAsset,
            "npy": NpyLazyAsset,
        }
        input_dataset_dir = kwargs.get("input_dataset_dir", "")
        num_files = kwargs.get("num_files")
        use_prob = kwargs.get("use_prob", False)
        data_name = kwargs.get("data_name", "raw_data.npz")
        data_path = kwargs.get("data_path")
        loader_name = kwargs.get("loader")
        if loader_name not in mapping:
            raise ValueError(f"unsupported loader: {loader_name}")
        loader_cls = mapping[loader_name]
        ignore_check = kwargs.get("ignore_check", False)

        if data_path is not None:
            if not isinstance(data_path, dict):
                raise NotImplementedError("data_path must be a dict")
            filepaths: List[str] = []
            cls_name: List[str] = []
            cls_bias: List[int] = []
            cls_length: List[int] = []
            cls_weight: List[float] = []

            for name, values in data_path.items():
                if not isinstance(values, list):
                    raise ValueError("each data_path item must be a list")
                for item in values:
                    if isinstance(item, str):
                        datalist_path = item
                        weight = 1.0
                    else:
                        datalist_path = item[0]
                        weight = float(item[1])
                    if not os.path.exists(datalist_path):
                        raise FileNotFoundError(
                            f"data list not found: {datalist_path}. "
                            "Create it under datalist/ or update the YAML path."
                        )
                    with open(datalist_path, "r", encoding="utf-8") as file:
                        lines = [line.strip() for line in file if line.strip()]

                    ok_lines: List[str] = []
                    missing = 0
                    for line in lines:
                        candidate = os.path.join(input_dataset_dir, line)
                        if data_name is not None:
                            candidate = os.path.join(candidate, data_name)
                        if ignore_check or os.path.exists(candidate):
                            ok_lines.append(line)
                        else:
                            missing += 1
                    if missing:
                        print(f"{datalist_path}: {missing} missing files")

                    cls_name.append(name)
                    cls_bias.append(len(filepaths))
                    cls_length.append(len(ok_lines))
                    cls_weight.append(weight)
                    filepaths.extend(ok_lines)
        else:
            raw_filepaths = kwargs["filepaths"]
            if isinstance(raw_filepaths, list):
                filepaths = raw_filepaths
                cls_name = None
                cls_bias = None
                cls_length = None
                cls_weight = None
            elif isinstance(raw_filepaths, dict):
                filepaths = []
                cls_name = []
                cls_bias = []
                cls_length = []
                cls_weight = []
                for name, values in raw_filepaths.items():
                    if not isinstance(values, list):
                        raise ValueError("each filepaths item must be a list")
                    cls_name.append(name)
                    cls_bias.append(len(filepaths))
                    cls_length.append(len(values))
                    cls_weight.append(1.0)
                    filepaths.extend(values)
            else:
                raise NotImplementedError("filepaths must be a list or dict")

        if cls_weight is not None:
            total = sum(cls_weight)
            if total <= 0:
                raise ValueError("class weights must have a positive sum")
            cls_weight = [weight / total for weight in cls_weight]

        return Datapath(
            filepaths=filepaths,
            input_dataset_dir=input_dataset_dir,
            cls_name=cls_name,
            cls_bias=cls_bias,
            cls_length=cls_length,
            num_files=num_files,
            use_prob=use_prob,
            cls_weight=cls_weight,
            loader=loader_cls,
            data_name=data_name,
            ignore_check=ignore_check,
        )

    def make(self, path: str, cls: Optional[str]) -> LazyAsset:
        return self.loader(path=path, cls=cls)

    def __getitem__(self, index: int) -> LazyAsset:
        if self.use_prob and self.cls_weight is not None:
            if self.cls_bias is None or self.cls_length is None:
                raise ValueError("probability sampling requires class metadata")
            if not hasattr(self, "perms"):
                self.perms = [list(range(length)) for length in self.cls_length]
                self.current_bias = [0 for _ in self.cls_length]
                for perm in self.perms:
                    shuffle(perm)

            cls_index = int(np.random.choice(len(self.cls_weight), p=self.cls_weight))
            local_index = self.perms[cls_index][self.current_bias[cls_index]]
            self.current_bias[cls_index] += 1
            if self.current_bias[cls_index] >= self.cls_length[cls_index]:
                shuffle(self.perms[cls_index])
                self.current_bias[cls_index] = 0

            name = None if self.cls_name is None else self.cls_name[cls_index]
            raw_path = self.filepaths[local_index + self.cls_bias[cls_index]]
        else:
            name = None
            if (
                self.cls_name is not None
                and self.cls_bias is not None
                and self.cls_length is not None
            ):
                for cls_index, start in enumerate(self.cls_bias):
                    if start <= index < start + self.cls_length[cls_index]:
                        name = self.cls_name[cls_index]
                        break
            raw_path = self.filepaths[index]

        path = os.path.join(self.input_dataset_dir, raw_path)
        if self.data_name is not None:
            path = os.path.join(path, self.data_name)
        return self.make(path=path, cls=name)

    def get_data(self) -> List[LazyAsset]:
        return [self[index] for index in range(len(self))]

    def split_by_cls(self) -> Dict[Optional[str], "Datapath"]:
        if self.cls_name is None:
            return {None: self}
        if self.cls_bias is None or self.cls_length is None:
            raise ValueError("class metadata is incomplete")

        grouped_paths = defaultdict(list)
        grouped_weights = defaultdict(list)
        for index, name in enumerate(self.cls_name):
            start = self.cls_bias[index]
            end = start + self.cls_length[index]
            grouped_paths[name].extend(self.filepaths[start:end])
            if self.cls_weight is not None:
                grouped_weights[name].append(self.cls_weight[index])

        result: Dict[Optional[str], Datapath] = {}
        for name, paths in grouped_paths.items():
            weights = None
            if self.cls_weight is not None:
                weights = grouped_weights[name]
                total = sum(weights)
                weights = [weight / total for weight in weights]
            result[name] = Datapath(
                filepaths=paths,
                input_dataset_dir=self.input_dataset_dir,
                cls_name=[name],
                cls_bias=[0],
                cls_length=[len(paths)],
                num_files=self.num_files,
                use_prob=self.use_prob,
                cls_weight=weights,
                loader=self.loader,
                data_name=self.data_name,
                ignore_check=self.ignore_check,
            )
        return result

    def __len__(self) -> int:
        if self.use_prob:
            if self.num_files is None:
                raise ValueError("num_files is required when use_prob=True")
            return self.num_files
        return len(self.filepaths)
