"""数据路径管理: 从 datalist/*.txt 读取模型清单, 按 loader(obj/npy) 惰性加载。

支持两类采样:
- use_prob=False: 按文件顺序遍历(验证/预测用)
- use_prob=True: 按类别权重概率采样, 每个类别内部维护一个洗牌指针循环遍历(训练用)
"""
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


class LazyAsset(ABC):
    """惰性加载器: 只存路径, __getitem__ 被调用时才真正读文件。"""

    def __init__(self, path: str, cls: Optional[str] = None):
        self.path = path
        self.cls = cls

    @abstractmethod
    def load(self) -> Asset:
        raise NotImplementedError()


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


class NpyLazyAsset(LazyAsset):
    def load(self) -> Asset:
        pc = np.load(self.path).astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"expected point cloud shape (N, 3), got {pc.shape}: {self.path}")
        return Asset(path=self.path, cls=self.cls, sampled_vertices_noisy=pc)


_LOADER_MAP = {None: ObjLazyAsset, "obj": ObjLazyAsset, "npy": NpyLazyAsset}


@dataclass
class Datapath(ConfigSpec):
    """描述一组数据文件及其采样方式。"""

    filepaths: List[str]
    input_dataset_dir: str = ""
    cls_name: Optional[List[str]] = None
    cls_bias: Optional[List[int]] = None       # 每个类别在 filepaths 中的起始偏移
    cls_length: Optional[List[int]] = None     # 每个类别的文件数
    num_files: Optional[int] = None            # use_prob=True 时一个 epoch 产出多少样本
    use_prob: bool = False
    cls_weight: Optional[List[float]] = None   # 类别采样权重(归一化到和为1)
    loader: type = ObjLazyAsset
    data_name: Optional[str] = None            # 文件相对路径后缀, 如 models/model_normalized.obj
    ignore_check: bool = False                 # True 时不校验文件是否存在

    @classmethod
    def parse(cls, **kwargs) -> "Datapath":
        input_dataset_dir = kwargs.get("input_dataset_dir", "")
        num_files = kwargs.get("num_files")
        use_prob = kwargs.get("use_prob", False)
        data_name = kwargs.get("data_name")
        loader_name = kwargs.get("loader")
        if loader_name not in _LOADER_MAP:
            raise ValueError(f"unsupported loader: {loader_name}")
        ignore_check = kwargs.get("ignore_check", False)
        data_path = kwargs.get("data_path")

        if data_path is None:
            raise ValueError("data_path is required")

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
                    datalist_path, weight = item, 1.0
                else:
                    datalist_path, weight = item[0], float(item[1])
                if not os.path.exists(datalist_path):
                    raise FileNotFoundError(
                        f"data list not found: {datalist_path}. "
                        "Create it under datalist/ or update the YAML path."
                    )
                with open(datalist_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]

                ok_lines, missing = [], 0
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

        total = sum(cls_weight)
        if total <= 0:
            raise ValueError("class weights must have a positive sum")
        cls_weight = [w / total for w in cls_weight]

        return Datapath(
            filepaths=filepaths,
            input_dataset_dir=input_dataset_dir,
            cls_name=cls_name,
            cls_bias=cls_bias,
            cls_length=cls_length,
            num_files=num_files,
            use_prob=use_prob,
            cls_weight=cls_weight,
            loader=_LOADER_MAP[loader_name],
            data_name=data_name,
            ignore_check=ignore_check,
        )

    def make(self, path: str, cls: Optional[str]) -> LazyAsset:
        return self.loader(path=path, cls=cls)

    def __getitem__(self, index: int) -> LazyAsset:
        if self.use_prob:
            return self._sample_by_prob()
        return self._get_by_index(index)

    def _sample_by_prob(self) -> LazyAsset:
        if self.cls_weight is None or self.cls_bias is None or self.cls_length is None:
            raise ValueError("probability sampling requires class metadata")
        # 惰性初始化每个类别的洗牌指针
        if not hasattr(self, "_perms"):
            self._perms = [list(range(n)) for n in self.cls_length]
            self._cursor = [0] * len(self.cls_length)
            for p in self._perms:
                shuffle(p)

        cls_idx = int(np.random.choice(len(self.cls_weight), p=self.cls_weight))
        local_idx = self._perms[cls_idx][self._cursor[cls_idx]]
        self._cursor[cls_idx] += 1
        if self._cursor[cls_idx] >= self.cls_length[cls_idx]:
            shuffle(self._perms[cls_idx])
            self._cursor[cls_idx] = 0

        name = None if self.cls_name is None else self.cls_name[cls_idx]
        raw_path = self.filepaths[local_idx + self.cls_bias[cls_idx]]
        return self._build_asset(raw_path, name)

    def _get_by_index(self, index: int) -> LazyAsset:
        name = None
        if self.cls_name is not None and self.cls_bias is not None and self.cls_length is not None:
            for cls_idx, start in enumerate(self.cls_bias):
                if start <= index < start + self.cls_length[cls_idx]:
                    name = self.cls_name[cls_idx]
                    break
        return self._build_asset(self.filepaths[index], name)

    def _build_asset(self, raw_path: str, cls: Optional[str]) -> LazyAsset:
        path = os.path.join(self.input_dataset_dir, raw_path)
        if self.data_name is not None:
            path = os.path.join(path, self.data_name)
        return self.make(path=path, cls=cls)

    def get_data(self) -> List[LazyAsset]:
        return [self[i] for i in range(len(self))]

    def split_by_cls(self) -> Dict[Optional[str], "Datapath"]:
        """按类别拆成多个 Datapath, 用于验证/预测时按类分别建 dataloader。"""
        if self.cls_name is None:
            return {None: self}
        if self.cls_bias is None or self.cls_length is None:
            raise ValueError("class metadata is incomplete")

        grouped_paths: Dict = defaultdict(list)
        grouped_weights: Dict = defaultdict(list)
        for idx, name in enumerate(self.cls_name):
            start = self.cls_bias[idx]
            end = start + self.cls_length[idx]
            grouped_paths[name].extend(self.filepaths[start:end])
            if self.cls_weight is not None:
                grouped_weights[name].append(self.cls_weight[idx])

        result: Dict[Optional[str], "Datapath"] = {}
        for name, paths in grouped_paths.items():
            weights = None
            if self.cls_weight is not None:
                weights = grouped_weights[name]
                total = sum(weights)
                weights = [w / total for w in weights]
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
