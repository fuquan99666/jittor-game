"""统一入口: 根据 task YAML 的 mode 执行 train / predict / debug。

配置组装流程:
  task.yaml
    ├── mode, components, optimizer, loss, trainer, writer, load_ckpt
    └── components: {data, transform, model, system}
            └── 每个 name -> configs/<kind>/<name>.yaml
                  └── 内部 __target__ -> 工厂查表实例化 Python 类

注意: OMP/MKL 线程数必须在 import torch 之前设置, 否则 numpy/torch 会竞争线程。
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from src.data.asset import Asset
from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system, get_writer


def load_config(kind: str, path: str) -> Dict:
    """加载 YAML 并 resolve 引用。path 可带或不带 .yaml 后缀。"""
    if path.endswith(".yaml"):
        path = path[:-5]
    yaml_path = path + ".yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{kind} config not found: {yaml_path}")
    print(f"load {kind} config: {yaml_path}")
    return OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def debug_iterate(data: PCDatasetModule) -> None:
    """跑一轮 train dataloader, 打印第一个 batch 的 Asset 形状(不训练)。"""
    loader = data.train_dataloader()
    if loader is None:
        raise ValueError("train dataloader is unavailable")
    for assets in tqdm(loader, desc="Debug dataloader"):
        if not all(isinstance(a, Asset) for a in assets):
            raise TypeError("debug dataloader should return Asset objects")
        first = assets[0]
        print({
            "batch_size": len(assets),
            "path": first.path,
            "sampled_vertices": None if first.sampled_vertices is None else first.sampled_vertices.shape,
            "sampled_vertices_noisy": None if first.sampled_vertices_noisy is None else first.sampled_vertices_noisy.shape,
            "meta_keys": [] if not first.meta else list(first.meta.keys()),
        })
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"device: {device}")

    task = load_config("task", args.task)
    mode = task["mode"]
    if mode not in {"train", "predict", "debug"}:
        raise ValueError(f"unsupported mode: {mode}")
    components = task["components"]

    # ---------- data ----------
    data_config = load_config("data", os.path.join("configs/data", components["data"]))
    train_cfg = data_config.get("train_dataset")
    train_config = None if train_cfg is None else DatasetConfig.parse(**train_cfg)
    val_cfg = data_config.get("validate_dataset")
    validate_config = None if val_cfg is None else DatasetConfig.parse(**val_cfg).split_by_cls()
    pred_cfg = data_config.get("predict_dataset")
    predict_config = None if pred_cfg is None else DatasetConfig.parse(**pred_cfg).split_by_cls()

    # ---------- transform ----------
    transform_config = load_config("transform", os.path.join("configs/transform", components["transform"]))

    # ---------- model ----------
    model = None
    model_component = components.get("model")
    if model_component is not None:
        model_config = load_config("model", os.path.join("configs/model", model_component))
        model = get_model(model_config=model_config, transform_config=transform_config).to(device)

    # model 存在时由它派发 transform(允许 model 覆盖), 否则用 transform_config 原值
    if model is None:
        train_transform = Transform.parse(**transform_config.get("train_transform", {}))
        validate_transform = Transform.parse(**transform_config.get("validate_transform", {}))
        predict_transform = Transform.parse(**transform_config.get("predict_transform", {}))
    else:
        train_transform = model.get_train_transform()
        validate_transform = model.get_validate_transform()
        predict_transform = model.get_predict_transform()

    # 安全检查: 预测 transform 必须为空, 否则会破坏已带噪的测试输入
    if mode == "predict" and predict_transform.augments:
        raise ValueError(
            "predict_transform must be empty (augments: []). "
            "Test inputs are already noisy.npy; applying sample/normalize/noise is wrong."
        )

    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model.process_fn_wrapper,
        train_dataset_config=train_config,
        validate_dataset_config=validate_config,
        predict_dataset_config=predict_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=bool(task.get("debug", False)),
    )

    # ---------- checkpoint ----------
    load_ckpt = task.get("load_ckpt")
    if load_ckpt is not None:
        if model is None:
            raise ValueError("cannot load a checkpoint without a model")
        model.load_checkpoint(load_ckpt, device=device)
        print(f"loaded checkpoint: {load_ckpt}")

    # ---------- system ----------
    system = None
    system_component = components.get("system")
    if system_component is not None:
        if model is None:
            raise ValueError("system requires a model")
        system_config = load_config("system", os.path.join("configs/system", system_component))
        writer_config = task.get("writer")
        writer = None if writer_config is None else get_writer(**writer_config)
        system = get_system(
            dataset_module=dataset_module,
            model=model,
            device=device,
            optimizer_config=task.get("optimizer"),
            loss_config=task.get("loss"),
            trainer_config=task.get("trainer"),
            writer=writer,
            **system_config,
        )

    # ---------- run ----------
    if mode == "debug":
        debug_iterate(dataset_module)
    elif mode == "train":
        if system is None:
            raise ValueError("training system is unavailable")
        system.train()
    elif mode == "predict":
        if system is None:
            raise ValueError("prediction system is unavailable")
        system.predict()


if __name__ == "__main__":
    main()
