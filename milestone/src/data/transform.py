from dataclasses import dataclass
from typing import List, Optional

from .asset import Asset
from .augment import Augment, get_augments
from .spec import ConfigSpec


@dataclass
class Transform(ConfigSpec):
    augments: Optional[List[Augment]] = None

    @classmethod
    def parse(cls, **kwargs) -> "Transform":
        cls.check_keys(kwargs)
        augment_config = kwargs.get("augments")
        if augment_config is None:
            return Transform()
        return Transform(augments=get_augments(*augment_config))

    def apply(self, asset: Asset) -> None:
        if self.augments is not None:
            for augment in self.augments:
                augment.apply(asset)
