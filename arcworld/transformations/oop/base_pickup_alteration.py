from typing import Callable, Dict, List, Type, cast

import numpy as np
from numpy.typing import NDArray

from arcworld.transformations.base_transform import GridsTransform
from arcworld.transformations.oop.deprecated.pickup_alteration import PICKUP_ALTERATIONS

PickupAlterator = Callable[[NDArray[np.int8], int], NDArray[np.int8]]


def fix_seed(
    func: PickupAlterator, seed: int
) -> Callable[[NDArray[np.int8]], NDArray[np.int8]]:
    return lambda x: func(x, seed)


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
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        res: List[NDArray[np.int8]] = []
        for grid in grids:
            transformed_grid = self._func(grid, seed)
            res.append(transformed_grid)

        return res


class Identity(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["show_as_is"], seed)
        return list(map(func, grids))


class DuplicateGridVerticallyTwo(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["duplicate_grid_2_times_vertically"], seed)
        return list(map(func, grids))


class DuplicateHorizontallyTwo(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["duplicate_grid_2_times_horizontally"], seed)
        return list(map(func, grids))


class DuplicateVerticallyFour(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["duplicate_grid_4_times_vertically"], seed)
        return list(map(func, grids))


class DuplicateHorizontallyFour(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["duplicate_grid_4_times_horizontally"], seed)
        return list(map(func, grids))


class Recolor(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["rot90_grid"], seed)
        return list(map(func, grids))


class Rot90(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["recolor_grid"], seed)
        return list(map(func, grids))


class MirrorVertical(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["mirror_vertical_grid"], seed)
        return list(map(func, grids))


class MirrorHorizontal(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["mirror_horizontal_grid"], seed)
        return list(map(func, grids))


class Duplicate4Times2By2(GridsTransform):
    def transform(
        self, grids: List[NDArray[np.int8]], seed: int
    ) -> List[NDArray[np.int8]]:
        func = fix_seed(PICKUP_ALTERATIONS["duplicate_grid_4_times_2by2"], seed)
        return list(map(func, grids))


DEFAULT_ALTERATIONS_DICT = cast(
    Dict[str, GridsTransform],
    {
        "show_as_is": Identity,
        "duplicate_grid_4_times_2by2": Duplicate4Times2By2,
        "duplicate_grid_2_times_vertically": DuplicateGridVerticallyTwo,
        "duplicate_grid_2_times_horizontally": DuplicateHorizontallyTwo,
        "duplicate_grid_4_times_vertically": DuplicateVerticallyFour,
        "duplicate_grid_4_times_horizontally": DuplicateHorizontallyFour,
        "rot90_grid": Rot90,
        "mirror_vertical_grid": MirrorVertical,
        "mirror_horizontal_grid": MirrorHorizontal,
        "recolor_grid": Recolor,
    },
)

DEFAULT_ALTERATIONS: List[Type[GridsTransform]] = list(
    DEFAULT_ALTERATIONS_DICT.values()
)
