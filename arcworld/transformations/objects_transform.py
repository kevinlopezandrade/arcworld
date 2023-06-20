from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Shapes


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
        self, grids: List[NDArray[np.float64]], seed: int
    ) -> List[NDArray[np.float64]]:
        pass
