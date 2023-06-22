from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from arcworld.transformations.base_transform import GridsTransform

PickupAlterator = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


class FunctionalGridTransform(GridsTransform):
    """
    Class for defining a transformation of grids from a Callable

    Use this class for fast prototyping of Grid Transforms, ideally
    you should implement a new class, and avoid this one.
    """

    def __init__(self, name: str, func: PickupAlterator) -> None:
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def transform(
        self, grids: List[NDArray[np.float64]], seed: int
    ) -> List[NDArray[np.float64]]:
        res: List[NDArray[np.float64]] = []
        for grid in grids:
            transformed_grid = self._func(grid, seed)
            res.append(transformed_grid)

        return res
