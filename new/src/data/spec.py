from dataclasses import fields
from abc import ABC, abstractmethod


class ConfigSpec(ABC):
    """所有可从 YAML 解析的配置数据类基类。

    子类用 @dataclass 声明字段，实现 parse() 从 dict 构造自身。
    check_keys() 在解析时拒绝未知字段，避免配置写错却静默生效。
    """

    @classmethod
    def check_keys(cls, config: dict, expect=None) -> None:
        if expect is None:
            expect = [f.name for f in fields(cls)]
        for key in config.keys():
            if key not in expect:
                raise ValueError(f"expect names {expect} in {cls.__name__}, found {key}")

    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> "ConfigSpec":
        raise NotImplementedError()
