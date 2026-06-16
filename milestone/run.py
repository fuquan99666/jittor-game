import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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
    if path.endswith(".yaml"):
        path = path[:-5]
    yaml_path = path + ".yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{kind} config not found: {yaml_path}")
    print(f"load {kind} config: {yaml_path}")
    return OmegaConf.to_container(
        OmegaConf.load(yaml_path),
        resolve=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def debug_fn(data: PCDatasetModule) -> None:
    dataloader = data.train_dataloader()
    if dataloader is None:
        raise ValueError("train dataloader is unavailable")
    for assets in tqdm(dataloader, desc="Debug dataloader"):
        if not all(isinstance(asset, Asset) for asset in assets):
            raise TypeError("debug dataloader should return Asset objects")
        first = assets[0]
        print(
            {
                "batch_size": len(assets),
                "path": first.path,
                "sampled_vertices": None
                if first.sampled_vertices is None
                else first.sampled_vertices.shape,
                "sampled_vertices_noisy": None
                if first.sampled_vertices_noisy is None
                else first.sampled_vertices_noisy.shape,
                "meta_keys": [] if first.meta is None else list(first.meta.keys()),
            }
        )
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"device: {device}")

    task = load_config("task", args.task)
    mode = task["mode"]
    if mode not in {"train", "predict", "debug"}:
        raise ValueError(f"unsupported mode: {mode}")
    components = task["components"]

    data_config = load_config(
        "data",
        os.path.join("configs/data", components["data"]),
    )

    raw_train_config = data_config.get("train_dataset")
    train_config = (
        None if raw_train_config is None else DatasetConfig.parse(**raw_train_config)
    )

    raw_validate_config = data_config.get("validate_dataset")
    validate_config = (
        None
        if raw_validate_config is None
        else DatasetConfig.parse(**raw_validate_config).split_by_cls()
    )

    raw_predict_config = data_config.get("predict_dataset")
    predict_config = (
        None
        if raw_predict_config is None
        else DatasetConfig.parse(**raw_predict_config).split_by_cls()
    )

    transform_config = load_config(
        "transform",
        os.path.join("configs/transform", components["transform"]),
    )

    model = None
    model_component = components.get("model")
    if model_component is not None:
        model_config = load_config(
            "model",
            os.path.join("configs/model", model_component),
        )
        model = get_model(
            model_config=model_config,
            transform_config=transform_config,
        ).to(device)

    train_transform = (
        Transform.parse(**transform_config.get("train_transform", {}))
        if model is None
        else model.get_train_transform()
    )
    validate_transform = (
        Transform.parse(**transform_config.get("validate_transform", {}))
        if model is None
        else model.get_validate_transform()
    )
    predict_transform = (
        Transform.parse(**transform_config.get("predict_transform", {}))
        if model is None
        else model.get_predict_transform()
    )

    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_config,
        validate_dataset_config=validate_config,
        predict_dataset_config=predict_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=bool(task.get("debug", False)),
    )

    load_checkpoint = task.get("load_ckpt")
    if load_checkpoint is not None:
        if model is None:
            raise ValueError("cannot load a checkpoint without a model")
        model.load_checkpoint(load_checkpoint, device=device)
        print(f"loaded checkpoint: {load_checkpoint}")

    system = None
    system_component = components.get("system")
    if system_component is not None:
        if model is None:
            raise ValueError("system requires a model")
        system_config = load_config(
            "system",
            os.path.join("configs/system", system_component),
        )
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

    if mode == "debug":
        debug_fn(dataset_module)
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
