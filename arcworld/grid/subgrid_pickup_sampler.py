import random
import time
from typing import List, Optional, Tuple, cast

import numpy as np
import scipy
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Shape, Shapes
from arcworld.dsl.functional import normalize
from arcworld.filters.functional.shape_filter import FunctionalFilter, get_filter
from arcworld.internal.constants import DoesNotFitException
from arcworld.shape.oop.base import ShapeObject
from arcworld.shape.oop.utils import grid_to_cropped_grid, grid_to_pc, shift_indexes


def to_shape_object(shape: Shape) -> ShapeObject:
    shape = cast(Shape, normalize(shape))  # Cast because of union type.
    point_cloud = {}

    for color, (x, y) in shape:
        point_cloud[x, y] = color

    return ShapeObject(point_cloud)


def make_uniform_color(shape: ShapeObject, color: int) -> ShapeObject:
    new_pc = {}
    for x, y in shape.pc:
        new_pc[x, y] = color

    return ShapeObject(new_pc)


class GridObject:
    def __init__(self, h: int, w: int) -> None:
        self._h = h
        self._w = w
        self._grid_shape = (h, w)
        self._objects: List[Shape] = []
        # TODO: Check if I can deifine this as a np.uint8
        self._grid = np.zeros(shape=self._grid_shape)

        self._objects: List[ShapeObject] = []

    @property
    def height(self) -> int:
        return self._h

    @property
    def width(self) -> int:
        return self._w

    @property
    def shape(self) -> Tuple[int, int]:
        return self._grid_shape

    @property
    def grid(self) -> NDArray[np.float64]:
        return self._grid

    @property
    def objects(self) -> List[ShapeObject]:
        return self._objects

    def place_object(
        self,
        shape: ShapeObject,
        background: int = 0,
        allow_touching_objects: bool = False,
    ) -> ShapeObject:
        """Randomly chooses position for the shape in the grid"""
        shape = ShapeObject(shape)
        zeroedworld = self.grid.copy()
        zeroedworld[self.grid == background] = 0

        if not allow_touching_objects:
            dilated_shape = scipy.ndimage.morphology.binary_dilation(
                shape.grid, structure=scipy.ndimage.generate_binary_structure(2, 2)
            ).astype(int)
            positions = self._find_possible_positions(zeroedworld, dilated_shape)
        else:
            positions = self._find_possible_positions(zeroedworld, shape.grid)

        if len(positions) == 0:
            raise DoesNotFitException("Shape does not fit")

        position = random.choice(positions)
        shape.move_to_position(position)
        shape_grid_at_world_size = shape.grid[
            : self.grid.shape[0], : self.grid.shape[1]
        ]

        # Update the grid
        self.grid[shape_grid_at_world_size > 0] = shape_grid_at_world_size[
            shape_grid_at_world_size > 0
        ]

        # Update the objects list.
        self._objects.append(shape)

        return shape

    @staticmethod
    def _find_possible_positions(
        world: NDArray[np.float64], grid: NDArray[np.float64], allow_holes: bool = True
    ) -> List[Tuple[int, int]]:
        world = world.copy()
        grid = grid_to_cropped_grid(grid)
        world[world != 0] = 1
        grid[grid != 0] = 1

        if not allow_holes:
            grid = scipy.ndimage.binary_fill_holes(grid).astype(int)
            world = scipy.ndimage.binary_fill_holes(world).astype(int)
        if world.shape[0] < grid.shape[0] or world.shape[1] < world.shape[1]:
            return []

        res = (
            scipy.signal.correlate2d(world, grid, mode="same", fillvalue=1) == 0
        ).astype(int)
        # values that are 0 are possible positions, but they use the middle as position
        # and not the top left corner, so shift to get top left corners
        dx = (grid.shape[0] - 1) // 2 * -1
        dy = (grid.shape[1] - 1) // 2 * -1
        indexes = shift_indexes(grid_to_pc(res).indexes, dx, dy)

        return indexes


class SubgridPickupGridSampler:
    def __init__(
        self,
        num_objects_range: Tuple[int, int] = (1, 7),
        same_number_of_shapes_accross_tasks: bool = False,
        grid_rows_range: Tuple[int, int] = (1, 30),
        grid_cols_range: Tuple[int, int] = (1, 30),
        same_grid_size_accross_tasks: bool = False,
        objects_even: Optional[bool] = None,
        grid_size_even: Optional[bool] = None,
    ):
        self._num_objects_range = num_objects_range
        self._same_number_of_shapes_accross_tasks = same_number_of_shapes_accross_tasks
        self._grid_rows_range = grid_rows_range
        self._grid_cols_range = grid_cols_range
        self._same_grid_size_accross_tasks = same_grid_size_accross_tasks
        self._objects_even = objects_even
        self._grid_size_even = grid_size_even
        self._fixed_cross_example_seed = int(time.time() * 1000)

    @property
    def num_objects_range(self) -> Tuple[int, int]:
        return self._num_objects_range

    @property
    def objects_even(self) -> Optional[bool]:
        return self._objects_even

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

    def _sample_num_shapes(self) -> int:
        min_n_shapes, max_n_shapes = self.num_objects_range

        if self.same_number_of_shapes_accross_tasks:
            example_seed = self._fixed_cross_example_seed
        else:
            example_seed = int(time.time() * 1000)

        random.seed(example_seed)
        n_shapes = random.randint(min_n_shapes, max_n_shapes)

        if self.objects_even is not None:
            if self.objects_even:
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
        self, shapes_objects: List[ShapeObject], resampler: object = None
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

        # TODO: Now we should use the Resamplers here, but let's ignore
        # that until I define the resamplers interface.

        # Assume the simplest possible resampler.
        sampled_shapes = random.sample(shapes_objects, n_shapes_per_grid)

        # Place the sampled shapes
        h, w = grid_size
        grid = GridObject(h, w)

        try:
            for i, s in enumerate(sampled_shapes):
                s = make_uniform_color(s, i + 1)
                grid.place_object(s)
        # TODO: Change general Exception for a more specific one.
        except Exception:
            raise ValueError("Error while placing the sampled shapes")
        else:
            # Here by definition of the foor loop, you must
            # have placed 'n_shapes_per_grid' in the grid.
            # Otherwise it should have failed.
            return grid

    def sample_input_grid(
        self, shapes: Shapes, max_trials: int = 20
    ) -> Optional[GridObject]:
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

        # TODO: Decide if its better to return None or raise an Exception.
        return None
