from copy import deepcopy

from .spec import ModelSpec
from .vm import VelocityModule


def get_model(model_config, **kwargs) -> ModelSpec:
    config = deepcopy(model_config)
    target = config.pop("__target__")
    mapping = {"VelocityModule": VelocityModule}
    if target not in mapping:
        raise ValueError(f"expected one of {list(mapping)}, found {target}")
    return mapping[target](model_config=config, **kwargs)
