import random
import time
from typing import List, Optional, Tuple

import numpy as np

from arcworld.dsl.arc_types import Shapes
from arcworld.filters.functional.shape_filter import FunctionalFilter, get_filter
from arcworld.grid.oop.grid_oop import GridObject, to_shape_object
from arcworld.schematas.oop.subgrid_pickup.resamplers import Resampler
from arcworld.shape.oop.base import ShapeObject


class SubgridPickupGridSampler:
    def __init__(
        self,
        num_shapes_range: Tuple[int, int] = (1, 7),
        same_number_of_shapes_accross_tasks: bool = False,
        grid_rows_range: Tuple[int, int] = (1, 30),
        grid_cols_range: Tuple[int, int] = (1, 30),
        same_grid_size_accross_tasks: bool = False,
        num_shapes_even: Optional[bool] = None,
        grid_size_even: Optional[bool] = None,
    ):
        self._num_shapes_range = num_shapes_range
        self._same_number_of_shapes_accross_tasks = same_number_of_shapes_accross_tasks
        self._grid_rows_range = grid_rows_range
        self._grid_cols_range = grid_cols_range
        self._same_grid_size_accross_tasks = same_grid_size_accross_tasks
        self._num_shapes_even = num_shapes_even
        self._grid_size_even = grid_size_even
        self._fixed_cross_example_seed = int(time.time() * 1000)
        self._resampler: Optional[Resampler] = None

    @property
    def num_shapes_range(self) -> Tuple[int, int]:
        return self._num_shapes_range

    @property
    def num_shapes_even(self) -> Optional[bool]:
        return self._num_shapes_even

    @property
    def same_number_of_shapes_accross_tasks(self) -> bool:
        return self._same_number_of_shapes_accross_tasks

    @property
    def grid_rows_range(self) -> Tuple[int, int]:
        return self._grid_rows_range

    @property
    def grid_cols_range(self) -> Tuple[int, int]:
        return self._grid_cols_range

    @property
    def same_grid_size_accross_tasks(self) -> bool:
        return self._same_grid_size_accross_tasks

    @property
    def grid_size_even(self) -> Optional[bool]:
        return self._grid_size_even

    def set_resampler(self, resampler: Optional[Resampler]):
        self._resampler = resampler

    def _sample_num_shapes(self) -> int:
        min_n_shapes, max_n_shapes = self.num_shapes_range

        if self.same_number_of_shapes_accross_tasks:
            example_seed = self._fixed_cross_example_seed
        else:
            example_seed = int(time.time() * 1000)

        random.seed(example_seed)
        n_shapes = random.randint(min_n_shapes, max_n_shapes)

        if self.num_shapes_even is not None:
            if self.num_shapes_even:
                n_shapes = int(2 * round(float(n_shapes) / 2))
            else:
                n_shapes = int(2 * round(float(n_shapes) / 2)) + 1

        return n_shapes

    def _sample_grid_size(self) -> Tuple[int, int]:
        min_grid_rows, max_grid_rows = self.grid_rows_range
        min_grid_cols, max_grid_cols = self.grid_cols_range

        if self.same_grid_size_accross_tasks:
            example_seed = int(time.time() * 1000)
        else:
            example_seed = self._fixed_cross_example_seed

        if self.grid_size_even is not None:
            if self.grid_size_even:
                offset = 0
            else:
                offset = 1

            min_grid_rows = int(2 * round(float(min_grid_rows) / 2)) + offset
            max_grid_rows = int(2 * round(float(max_grid_rows) / 2)) + offset
            min_grid_cols = int(2 * round(float(min_grid_cols) / 2)) + offset
            max_grid_cols = int(2 * round(float(max_grid_cols) / 2)) + offset

            possible_row_values = np.arange(min_grid_rows, max_grid_rows + 1, 2)
            possible_col_values = np.arange(min_grid_cols, max_grid_cols + 1, 2)

            random.seed(example_seed)
            grid_size = (
                np.random.choice(possible_row_values),
                np.random.choice(possible_col_values),
            )

        else:
            possible_row_values = np.arange(min_grid_rows, max_grid_rows + 1, 1)
            possible_col_values = np.arange(min_grid_cols, max_grid_cols + 1, 1)
            random.seed(example_seed)
            grid_size = (
                np.random.choice(possible_row_values),
                np.random.choice(possible_col_values),
            )

        return grid_size

    def _get_max_shape_size_filter(
        self, n_shapes_per_grid: int, grid_size: Tuple[int, int]
    ) -> List[FunctionalFilter]:
        rows = grid_size[0]
        cols = grid_size[1]

        filters: List[FunctionalFilter] = []

        if rows <= 1:
            row_cond = get_filter("is_shape_less_than_2_rows")
            filters.append(row_cond)
        elif rows < 15:
            row_cond = get_filter("is_shape_less_than_" + str(rows) + "_rows")
            filters.append(row_cond)

        if cols <= 1:
            col_cond = get_filter("is_shape_less_than_2_cols")
            filters.append(col_cond)
        elif cols < 15:
            col_cond = get_filter("is_shape_less_than_" + str(cols) + "_cols")
            filters.append(col_cond)

        n_cells_available = rows * cols
        ratio_grid_cell_to_shape_cell = int(n_cells_available / n_shapes_per_grid)

        if ratio_grid_cell_to_shape_cell < 2:
            ratio_grid_cell_to_shape_cell = 2

        if ratio_grid_cell_to_shape_cell <= 1:
            # TODO: Ask Yassine this conditions is never reached.
            cell_cond = get_filter("is_shape_less_than_2_cell")
            filters.append(cell_cond)
        elif ratio_grid_cell_to_shape_cell < 15:
            cell_cond = get_filter(
                "is_shape_less_than_" + str(ratio_grid_cell_to_shape_cell) + "_cell"
            )
            filters.append(cell_cond)

        return filters

    def _resample_shapes_and_place(
        self, shapes_objects: List[ShapeObject]
    ) -> GridObject:
        """
        Takes a set of shapes and resamples according to some resampler criteria.
        For the moment the resampler criteria is yet to be defined.
        """
        # Sample Initial Conditions
        n_shapes_per_grid = self._sample_num_shapes()
        grid_size = self._sample_grid_size()

        # Sample Extra Filters
        extra_filters = self._get_max_shape_size_filter(n_shapes_per_grid, grid_size)

        # Ignore the type issues for now.
        for filter in extra_filters:
            shapes_objects = filter.filter(shapes_objects)  # type: ignore

        if self._resampler is None:
            # Assume the simplest possible resampler.
            sampled_shapes = random.sample(shapes_objects, n_shapes_per_grid)
        else:
            sampled_shapes = self._resampler.resample(shapes_objects, n_shapes_per_grid)

        # Place the sampled shapes
        h, w = grid_size
        grid = GridObject(h, w)

        try:
            for i, s in enumerate(sampled_shapes):
                # s = make_uniform_color(s, i + 1)
                grid.place_object(s)
        # TODO: Change general Exception for a more specific one.
        except Exception:
            raise ValueError("Error while placing the sampled shapes")
        else:
            # Here by definition of the foor loop, you must
            # have placed 'n_shapes_per_grid' in the grid.
            # Otherwise it should have failed.
            return grid

    def sample_input_grid(self, shapes: Shapes, max_trials: int = 20) -> GridObject:
        """
        Here we perform the initial placement of objects.

        As Per Yassine here is where he defines the filter programs.
        It would be nice here to use Yassine Filters to proof the
        interoperability between Michael and Yassine filters.

        For that I need to pass from shapes to the actual object Shape.
        Before this method though I can use normal DSL filters as well.
        """
        # Transform the objects to ShapesObjects as used by Yassine.
        # This set in theory should not work since ShapeObject is not
        # hashable.
        shapes_objects = list(to_shape_object(shape) for shape in shapes)

        # So I stop either when I ran out of trials or when I placed n_objects.

        trial = 0
        while trial < max_trials:
            print(f"Attempting trial: {trial}")
            try:
                sampled_grid = self._resample_shapes_and_place(shapes_objects)
            # TODO: Change general Exception for a more specific one.
            except Exception:
                trial += 1
            else:
                return sampled_grid

        # TODO: Raise a more specifi exception.
        raise ValueError
