"""System / Writer 工厂: 根据 __target__ 查表实例化。

新增 system 步骤:
  1. 实现类(继承 BaseSystem / BaseWriter)
  2. 在 _SYSTEM_MAP / _WRITER_MAP 注册名字
  3. 加 configs/system/<name>.yaml
"""
from copy import deepcopy
from typing import Dict

from .spec import BaseSystem, BaseWriter
from .vm import VMSystem, VMWriter

_SYSTEM_MAP: Dict[str, type] = {"vm": VMSystem}
_WRITER_MAP: Dict[str, type] = {"vm": VMWriter}


def get_system(**kwargs) -> BaseSystem:
    config = deepcopy(kwargs)
    target = config.pop("__target__")
    if target not in _SYSTEM_MAP:
        raise ValueError(f"expected one of {list(_SYSTEM_MAP)}, found {target}")
    return _SYSTEM_MAP[target](**config)


def get_writer(**kwargs) -> BaseWriter:
    config = deepcopy(kwargs)
    target = config.pop("__target__")
    if target not in _WRITER_MAP:
        raise ValueError(f"expected one of {list(_WRITER_MAP)}, found {target}")
    return _WRITER_MAP[target](**config)
