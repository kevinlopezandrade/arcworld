from abc import ABCMeta, abstractmethod
from typing import Optional, Set

from arcworld.dsl.arc_types import Shapes
from arcworld.grid.grid_protocol import GridProtocol


class ObjectsTransform(metaclass=ABCMeta):
    """
    Interface for every object transform
    """

    @abstractmethod
    def transform(self, objects: Shapes) -> Shapes:
        pass


class GridTransform(metaclass=ABCMeta):
    """
    Interface for evey grid transform
    """

    @abstractmethod
    def transform(
        self, grids: Set[GridProtocol], seed: Optional[int] = None
    ) -> Set[GridProtocol]:
        pass
