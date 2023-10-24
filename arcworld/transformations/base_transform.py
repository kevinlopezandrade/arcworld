from abc import ABCMeta, abstractmethod
from typing import Any, Protocol

from arcworld.dsl.arc_types import Shapes


class ShapesTransform(metaclass=ABCMeta):
    """
    Interface for every object transform
    """

    @abstractmethod
    def transform(self, objects: Shapes) -> Shapes:
        pass


class TransformProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    @abstractmethod
    def transform(self, grid: Any) -> Any:
        ...
