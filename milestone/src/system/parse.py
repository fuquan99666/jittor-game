from copy import deepcopy

from .spec import DummySystem, DummyWriter
from .vm import VMSystem, VMWriter


def get_system(**kwargs) -> DummySystem:
    config = deepcopy(kwargs)
    target = config.pop("__target__")
    mapping = {"dummy": DummySystem, "vm": VMSystem}
    if target not in mapping:
        raise ValueError(f"expected one of {list(mapping)}, found {target}")
    return mapping[target](**config)


def get_writer(**kwargs) -> DummyWriter:
    config = deepcopy(kwargs)
    target = config.pop("__target__")
    mapping = {"dummy": DummyWriter, "vm": VMWriter}
    if target not in mapping:
        raise ValueError(f"expected one of {list(mapping)}, found {target}")
    return mapping[target](**config)
