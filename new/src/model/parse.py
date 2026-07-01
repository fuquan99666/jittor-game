"""模型工厂: 根据 __target__ 查表实例化模型类。

新增模型步骤:
  1. 实现类(继承 ModelSpec)
  2. 在 _MODEL_MAP 注册名字
  3. 加 configs/model/<name>.yaml
"""
from copy import deepcopy
from typing import Dict

from .spec import ModelSpec
from .vm import VelocityModule
from .score import ScoreModule

_MODEL_MAP: Dict[str, type] = {
    "VelocityModule": VelocityModule,
    "ScoreModule": ScoreModule,
}


def get_model(model_config, **kwargs) -> ModelSpec:
    config = deepcopy(model_config)
    target = config.pop("__target__")
    if target not in _MODEL_MAP:
        raise ValueError(f"expected one of {list(_MODEL_MAP)}, found {target}")
    return _MODEL_MAP[target](model_config=config, **kwargs)


def register_model(name: str, cls: type) -> None:
    _MODEL_MAP[name] = cls
