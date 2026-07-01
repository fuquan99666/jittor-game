from dataclasses import dataclass
from typing import List, Optional

from .asset import Asset
from .augment import Augment, get_augments
from .spec import ConfigSpec


@dataclass
class Transform(ConfigSpec):
    """有序 augment 容器。apply() 依次对 asset 执行。"""
    augments: Optional[List[Augment]] = None

    @classmethod
    def parse(cls, **kwargs) -> "Transform":
        cls.check_keys(kwargs)
        cfg = kwargs.get("augments")
        if cfg is None:
            return Transform()
        return Transform(augments=get_augments(*cfg))

    def apply(self, asset: Asset) -> None:
        if self.augments is None:
            return
        for aug in self.augments:
            aug.apply(asset)
