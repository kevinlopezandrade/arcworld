from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from arcworld.transformations.objects_transform import GridTransform

PickupAlterator = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


class FunctionalGridTransform(GridTransform):
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
