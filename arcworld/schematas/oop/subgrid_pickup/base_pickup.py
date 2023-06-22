from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from arcworld.filters.dsl.dsl_filter import DSLFilter
from arcworld.filters.functional.shape_filter import get_filter
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.grid.oop.grid_oop import GridObject
from arcworld.internal.program import FilterProgram
from arcworld.schematas.oop.subgrid_pickup.resamplers import (
    OnlyShapesRepeated,
    RepeatedShapesResampler,
    UniqueShapeParemeter,
)
from arcworld.schematas.oop.subgrid_pickup.subgrid_pickup_sampler import (
    SubgridPickupGridSampler,
)

SymmetricDef = (
    "Symmetric",
    "fork(either, fork(either, fork(equality, compose(toindices, dmirror), toindices), fork(equality, compose(toindices, cmirror), toindices)), fork(either, fork(equality, compose(toindices, vmirror), toindices), fork(equality, compose(toindices, hmirror), toindices)))",  # noqa
)


# TODO: Maybe avoid this pattern and just pass a functional.
class SubgridPickup(metaclass=ABCMeta):
    """
    Base Class for all SubgridPickup selectors.
    """

    CONSTRAINTS = {
        "same_grid_size_accross_tasks": False,
        "same_number_of_shapes_accross_tasks": False,
        "num_shapes_range": (1, 7),
        "some_shapes_repeated_in_grid": False,
        "only_shapes_repeated_in_grid": False,
        "num_shapes_even": None,
        "grid_rows_range": (1, 30),
        "grid_cols_range": (1, 30),
        "grid_size_even": None,
        "shape_parameter_to_distinguish": None,
        "median": False,
    }

    DEFAULT_CONDITIONS: List[ShapesFilter] = [
        DSLFilter(FilterProgram(*SymmetricDef)),
        get_filter("is_shape_evenly_colored"),
        get_filter("is_shape_fully_connected"),
    ]

    def __init__(self) -> None:
        SubgridPickup._process_constraints(self.CONSTRAINTS)

    @property
    def conditions(self) -> List[ShapesFilter]:
        """
        Every SubgridPickup can define here their shape conditions or use
        the default ones.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _pre_create_grid_sampler(self) -> SubgridPickupGridSampler:
        """
        Creates a grid sampler that already fullfill its
        constraints. The user can overwrite this method to adjust it to its needs,
        or leave this default one.
        """
        num_shapes_range = cast(Tuple[int, int], self.CONSTRAINTS["num_shapes_range"])
        grid_rows_range = cast(Tuple[int, int], self.CONSTRAINTS["grid_rows_range"])
        grid_cols_range = cast(Tuple[int, int], self.CONSTRAINTS["grid_cols_range"])

        same_number_of_shapes_accross_tasks = cast(
            bool, self.CONSTRAINTS["same_number_of_shapes_accross_tasks"]
        )
        same_grid_size_accross_tasks = cast(
            bool, self.CONSTRAINTS["same_grid_size_accross_tasks"]
        )

        grid_size_even = cast(Optional[bool], self.CONSTRAINTS["grid_size_even"])
        num_shapes_even = cast(Optional[bool], self.CONSTRAINTS["num_shapes_even"])

        sampler = SubgridPickupGridSampler(
            num_shapes_range=num_shapes_range,
            same_number_of_shapes_accross_tasks=same_number_of_shapes_accross_tasks,
            grid_rows_range=grid_rows_range,
            grid_cols_range=grid_cols_range,
            same_grid_size_accross_tasks=same_grid_size_accross_tasks,
            num_shapes_even=num_shapes_even,
            grid_size_even=grid_size_even,
        )

        return sampler

    def _add_resampler(self, sampler: SubgridPickupGridSampler):
        """
        User can overwrite this method to add a resampler to the
        grid sampler already created.
        """
        if self.CONSTRAINTS["some_shapes_repeated_in_grid"]:
            median = False
            if self.CONSTRAINTS["median"]:
                median = True

            resampler = RepeatedShapesResampler(median)
        elif self.CONSTRAINTS["only_shapes_repeated_in_grid"]:
            resampler = OnlyShapesRepeated()
        elif cast(
            Optional[str], self.CONSTRAINTS["shape_parameter_to_distinguish"]
        ) in [
            "n_rows",
            "n_cols",
            "n_cells",
        ]:
            param = cast(str, self.CONSTRAINTS["shape_parameter_to_distinguish"])
            resampler = UniqueShapeParemeter(param)
        else:
            resampler = None

        sampler.set_resampler(resampler)

    def create_grid_sampler(self) -> SubgridPickupGridSampler:
        """
        Creates a SubgridPickupGridSampler object that will
        allow to sample input grids already satisying the
        constraints of the SubgridPickup.

        User is not assumed to overload this method, for that
        purpose use _pre_create_grid_sampler or _add_resampler

        Returns:
            SubgridPickupGridSampler
        """
        grid_sampler = self._pre_create_grid_sampler()
        self._add_resampler(grid_sampler)

        return grid_sampler

    @staticmethod
    def _process_constraints(transformation_constraints: Dict[str, Any]):
        """
        This method ensures that all constraint values are either the ones provided
        by the user or the ones in the default CONSTRAINTS.
        It permutes the original dict.
        """
        for key in SubgridPickup.CONSTRAINTS.keys():
            if key not in transformation_constraints.keys():
                transformation_constraints[key] = SubgridPickup.CONSTRAINTS[key]

    @abstractmethod
    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        ...
        """
        Every derived class should implement this method defining
        how to pick a subgrid.

        Args:
            grid: Grid in which to act.

        Returns:
            Grid or shape selection as a numpy array.
        """
        ...
