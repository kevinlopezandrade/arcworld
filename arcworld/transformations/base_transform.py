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
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform(
        self, grids: List[NDArray[np.float64]], seed: int
    ) -> List[NDArray[np.float64]]:
        pass
