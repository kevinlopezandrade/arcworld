from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Shapes


class ShapesTransform(metaclass=ABCMeta):
    """
    Interface for every object transform
    """

    @abstractmethod
    def transform(self, objects: Shapes) -> Shapes:
        pass


class GridsTransform(metaclass=ABCMeta):
    """
    Interface for evey grid transform
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        pass
