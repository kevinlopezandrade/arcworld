from abc import ABCMeta, abstractmethod
from typing import Any, Protocol

from arcworld.dsl.arc_types import Objects


class ObjectsTransform(metaclass=ABCMeta):
    """
    Interface for every object transform
    """

    @abstractmethod
    def transform(self, objects: Objects) -> Objects:
        pass


class TransformProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    @abstractmethod
    def transform(self, grid: Any) -> Any:
        ...
