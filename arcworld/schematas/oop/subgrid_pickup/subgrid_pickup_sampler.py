import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from arcworld.dsl.arc_constants import TWO
from arcworld.dsl.arc_types import Shapes
from arcworld.dsl.functional import compose, greater, height, lbind, size, width
from arcworld.filters.functional.shape_filter import FunctionalFilter, get_filter
from arcworld.grid.grid_protocol import GridProtocol
from arcworld.grid.oop.grid_oop import GridObject
from arcworld.internal.constants import DoesNotFitError, GridConstructionError
from arcworld.schematas.oop.subgrid_pickup.resamplers import Resampler

logger = logging.getLogger(__name__)


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

    def set_grid_class(self, grid_cls: Type[GridProtocol], **kwargs: Dict[str, Any]):
        """
        Sets the grid class to use and the kwargs passsed when creating an instance
        of that class.
        """
        self._grid_cls = grid_cls
        self._grid_kwargs = kwargs

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
            example_seed = self._fixed_cross_example_seed
        else:
            example_seed = int(time.time() * 1000)

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
            boundary = compose(lbind(greater, TWO), height)
            row_cond = FunctionalFilter("is_shape_less_than_2_rows", boundary)
            filters.append(row_cond)
        elif rows < 15:
            boundary = compose(lbind(greater, rows), height)
            row_cond = FunctionalFilter(
                f"is_shape_less_than_{str(rows)}_rows", boundary
            )
            filters.append(row_cond)

        if cols <= 1:
            boundary = compose(lbind(greater, TWO), width)
            col_cond = FunctionalFilter("is_shape_less_than_2_cols", boundary)
            filters.append(col_cond)
        elif cols < 15:
            boundary = compose(lbind(greater, cols), width)
            col_cond = FunctionalFilter(
                f"is_shape_less_than_{str(cols)}_cols", boundary
            )
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
            boundary = compose(lbind(greater, ratio_grid_cell_to_shape_cell), size)
            cell_cond = FunctionalFilter(
                f"is_shape_less_than_{str(ratio_grid_cell_to_shape_cell)}_cell",
                boundary,
            )
            filters.append(cell_cond)

        return filters

    def _resample_shapes_and_place(self, shapes: Shapes) -> GridObject:
        """
        Takes a set of shapes and resamples according to some resampler criteria.
        For the moment the resampler criteria is yet to be defined.
        """
        # Sample Initial Conditions
        n_shapes_per_grid = self._sample_num_shapes()
        grid_size = self._sample_grid_size()

        # Sample Extra Filters
        extra_filters = self._get_max_shape_size_filter(n_shapes_per_grid, grid_size)

        for filter in extra_filters:
            shapes = filter.filter(shapes)

        if len(shapes) == 0:
            raise GridConstructionError("No shapes to be placed after filtering")

        logger.debug(f"Len after extra filters: {len(shapes)}")

        if self._resampler is None:
            # Assume the simplest possible resampler.
            sampled_shapes = random.sample(shapes, n_shapes_per_grid)
        else:
            sampled_shapes = self._resampler.resample(shapes, n_shapes_per_grid)

        logger.debug(f"Resampled shapes {len(sampled_shapes)}")

        if len(sampled_shapes) < n_shapes_per_grid:
            raise GridConstructionError(
                f"Number of resampled shapes is different from "
                f"the desired {n_shapes_per_grid}"
            )

        # Place the sampled shapes
        h, w = grid_size
        grid = self._grid_cls(h, w, **self._grid_kwargs)

        try:
            for s in sampled_shapes:
                grid.place_object(s)
        except DoesNotFitError:
            raise GridConstructionError(
                f"Could not place {n_shapes_per_grid} sampled shapes in the grid"
            ) from DoesNotFitError
        else:
            # Here by definition of the foor loop, you must
            # have placed 'n_shapes_per_grid' in the grid.
            # Otherwise it should have failed.
            # assert len(grid.shapes) == n_shapes_per_grid

            return grid

    def sample_input_grid(self, shapes: Shapes, max_trials: int = 20) -> GridObject:
        trial = 0
        while trial < max_trials:
            logger.debug(f"Attempting trial {trial}")
            try:
                sampled_grid = self._resample_shapes_and_place(shapes)
            except GridConstructionError as e:
                logger.debug(f"Trial {trial} failed: {e}")
                trial += 1
            except Exception as e:
                logger.debug(f"Trial {trial} failed: Unknow exception {e}")
                trial += 1
            else:
                return sampled_grid

        raise GridConstructionError(f"{max_trials} trials without a a valid grid.")
