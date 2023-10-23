import math
import random
from typing import Callable, List, Optional, Tuple, cast

from arcworld.dsl.arc_types import Coordinate, Grid, Shape
from arcworld.dsl.functional import (
    cover,
    fill,
    height,
    llcorner,
    lrcorner,
    shift,
    sign,
    toindices,
    ulcorner,
    width,
)
from arcworld.filters.functional.shape_filter import FunctionalFilter
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.grid.oop.grid_bruteforce import BinaryRelation, BSTGridBruteForce
from arcworld.grid.oop.util import Node
from arcworld.internal.constants import ALLOWED_COLORS
from arcworld.schematas.oop.drop.grid import BarOrientation, GravityGridBuilder
from arcworld.shape.resamplers import OnlyShapesRepeated


def in_order_append(node: Optional[Node[Shape]], shapes: List[Shape]):
    if node:
        in_order_append(node.left, shapes)
        shapes.append(node.key)
        in_order_append(node.right, shapes)


def displace(grid: Grid, shape: Shape, disp: Coordinate, background: int = 0):
    displaced = shift(shape, disp)
    to_be_freed = toindices(shape) - toindices(displaced)
    grid = fill(grid, background, to_be_freed)
    return grid


def shift_until_touches_y(
    grid: BSTGridBruteForce,
    shape: Shape,
    y_dest: int,
    remove_if_limit: bool = False,
    from_below: bool = True,
):
    if from_below:
        y_orig = ulcorner(shape)[0]
    else:
        y_orig = llcorner(shape)[0]
    steps = y_dest - y_orig
    factor = cast(int, sign(steps))

    disp: Coordinate = (0, 0)
    trials = 1
    while len(
        toindices(shift(shape, (disp[0] + (factor * 1), 0)))
        & (grid._occupied - toindices(shape))
    ) == 0 and trials <= abs(steps):
        disp = (disp[0] + factor * 1, 0)
        trials += 1

    if remove_if_limit and trials == abs(steps) + 1:
        grid._grid = cover(grid._grid, shape)
        grid._occupied = grid._occupied - toindices(shape)
    else:
        grid._grid = displace(grid._grid, shape, disp, background=grid._bg_color)

        # Update the occupied ones to remove the previous points.
        grid._occupied = grid._occupied - toindices(shape)

        # Update the grid.
        grid.add_shape(shift(shape, disp), padding=0, no_bbox=True)


def shift_until_touches_x(
    grid: BSTGridBruteForce,
    shape: Shape,
    x_dest: int,
    remove_if_limit: bool = False,
    from_left: bool = True,
):
    if from_left:
        x_orig = lrcorner(shape)[1]
    else:
        x_orig = llcorner(shape)[1]
    steps = x_dest - x_orig
    factor = cast(int, sign(steps))

    disp: Coordinate = (0, 0)
    trials = 1
    while len(
        toindices(shift(shape, (0, disp[1] + (factor * 1))))
        & (grid._occupied - toindices(shape))
    ) == 0 and trials <= abs(steps):
        disp = (0, disp[1] + factor * 1)
        trials += 1

    if remove_if_limit and trials == abs(steps) + 1:
        grid._grid = cover(grid._grid, shape)
        grid._occupied = grid._occupied - toindices(shape)
    else:
        grid._grid = displace(grid._grid, shape, disp, background=grid._bg_color)

        # Update the occupied ones to remove the previous points.
        grid._occupied = grid._occupied - toindices(shape)

        # Update the grid.
        grid.add_shape(shift(shape, disp), padding=0, no_bbox=True)


def _get_orientation(bar: Shape) -> BarOrientation:
    if height(bar) == 1:
        return BarOrientation.H
    elif width(bar) == 1:
        return BarOrientation.V
    else:
        raise ValueError("Not a bar")


def get_max_dimension_filter(dim: float) -> Callable[[Shape], bool]:
    def max_dim(shape: Shape) -> bool:
        if width(shape) <= dim and height(shape) <= dim:
            return True
        else:
            return False

    return max_dim


class DropBidirectional:
    def __init__(
        self,
        max_shape_dimesion: float = 3,
        grid_dimensions_range: Tuple[int, int] = (20, 25),
        max_shapes_range: Tuple[int, int] = (5, 15),
        bar_orientations: Tuple[BarOrientation, ...] = (
            BarOrientation.H,
            BarOrientation.V,
        ),
        holes_fraction_range: Tuple[float, float] = (0, 2 / 4),
    ) -> None:
        self._max_shape_dimension = max_shape_dimesion
        self.program = self.__class__.__name__

        self.grid_dimensions_range = grid_dimensions_range
        self.max_shapes_range = max_shapes_range
        self.bar_orientations = bar_orientations
        self.holes_fraction_range = holes_fraction_range

    @property
    def filters(self) -> List[ShapesFilter]:
        if self._max_shape_dimension < math.inf:
            filter = FunctionalFilter(
                f"MAX_DIM_{self._max_shape_dimension}",
                get_max_dimension_filter(self._max_shape_dimension),
            )
            return [filter]
        else:
            return []

    def grid_sampler(self) -> Callable[[], GravityGridBuilder]:
        """
        Creates a grid builder with random parameters.
        """
        bg_color = random.choice(list(ALLOWED_COLORS))
        bar_color = random.choice(list(ALLOWED_COLORS - {bg_color}))

        def sampler():
            h = random.randint(*self.grid_dimensions_range)
            w = random.randint(*self.grid_dimensions_range)

            if math.inf in self.max_shapes_range:
                max_shapes = math.inf
            else:
                max_shapes = random.randint(*self.max_shapes_range)

            bar_orientation = random.choice(self.bar_orientations)
            holes_fraction = random.uniform(*self.holes_fraction_range)

            return GravityGridBuilder(
                height=h,
                width=w,
                max_shapes=max_shapes,
                bar_orientation=bar_orientation,
                holes_fraction=holes_fraction,
                bg_color=bg_color,
                bar_color=bar_color,
            )

        return sampler

    @staticmethod
    def _shift_shapes_vertically(
        root_input: Node[Shape],
        output_grid: BSTGridBruteForce,
        remove_if_limit: bool = False,
    ) -> BSTGridBruteForce:
        # Grab all the shapes except by the root.
        shapes_below: List[Shape] = []
        shapes_above: List[Shape] = []

        in_order_append(root_input.left, shapes_below)
        in_order_append(root_input.right, shapes_above)

        # Add the shapes.
        for shape in shapes_below + shapes_above:
            output_grid.add_shape(shape, no_bbox=True)

        # Fill the shapes below the bar.
        y_dest = ulcorner(root_input.key)[0]
        shapes_below = shapes_below[::-1]
        for shape in shapes_below:
            shift_until_touches_y(
                output_grid,
                shape,
                y_dest,
                remove_if_limit=remove_if_limit,
                from_below=True,
            )

        # Fill the shapes above the bar.
        for shape in shapes_above:
            shift_until_touches_y(
                output_grid,
                shape,
                y_dest,
                remove_if_limit=remove_if_limit,
                from_below=False,
            )

        return output_grid

    @staticmethod
    def _shift_shapes_horizontally(
        root_input: Node[Shape],
        output_grid: BSTGridBruteForce,
        remove_if_limit: bool = False,
    ):
        shapes_left: List[Shape] = []
        shapes_right: List[Shape] = []

        in_order_append(root_input.left, shapes_left)
        in_order_append(root_input.right, shapes_right)

        # Add the shapes.
        for shape in shapes_left + shapes_right:
            output_grid.add_shape(shape, no_bbox=True)

        # Fill the shapes to the left of the bar.
        x_dest = lrcorner(root_input.key)[1]
        shapes_left = shapes_left[::-1]
        for shape in shapes_left:
            shift_until_touches_x(
                output_grid,
                shape,
                x_dest,
                remove_if_limit=remove_if_limit,
                from_left=True,
            )

        # Fill the shapes to the right of the bar.
        for shape in shapes_right:
            shift_until_touches_x(
                output_grid,
                shape,
                x_dest,
                remove_if_limit=remove_if_limit,
                from_left=False,
            )

        return output_grid

    def transform(self, grid: BSTGridBruteForce) -> BSTGridBruteForce:
        root = grid.tree.root

        assert root is not None

        orientation = _get_orientation(root.key)

        # Create new output grid.
        if orientation == BarOrientation.H:
            output_grid = BSTGridBruteForce(
                grid.height,
                grid.width,
                margin=grid.margin,
                mode=BinaryRelation.BelowOf,
                bg_color=grid._bg_color,
            )
        else:
            output_grid = BSTGridBruteForce(
                grid.height,
                grid.width,
                margin=grid.margin,
                mode=BinaryRelation.LeftOf,
                bg_color=grid._bg_color,
            )

        output_grid.add_shape(root.key, padding=0, no_bbox=True)

        if orientation == BarOrientation.H:
            self._shift_shapes_vertically(root_input=root, output_grid=output_grid)
        else:
            self._shift_shapes_horizontally(root_input=root, output_grid=output_grid)

        return output_grid


class DropBidirectionalDots(DropBidirectional):
    def __init__(
        self,
        max_shape_dimesion: float = 1,
        grid_dimensions_range: Tuple[int, int] = (20, 25),
        max_shapes_range: Tuple[int, int] = (5, 15),
        bar_orientations: Tuple[BarOrientation, ...] = (
            BarOrientation.H,
            BarOrientation.V,
        ),
        holes_fraction_range: Tuple[float, float] = (0, 0.5),
    ) -> None:
        super().__init__(
            max_shape_dimesion,
            grid_dimensions_range,
            max_shapes_range,
            bar_orientations,
            holes_fraction_range,
        )

    @property
    def filters(self) -> List[ShapesFilter]:
        max_dim_filter = FunctionalFilter("MAX_DIM_1", get_max_dimension_filter(1))

        return [max_dim_filter]

    def grid_sampler(self) -> Callable[[], GravityGridBuilder]:
        """
        A grid sampler samples GridBuilders that satisfy their conditions.
        It should return a callback that will generate random parameters
        passed to the constructor, that satisfy its requirements.
        """
        bg_color = random.choice(list(ALLOWED_COLORS))
        bar_color = random.choice(list(ALLOWED_COLORS - {bg_color}))

        def sampler():
            h = random.randint(*self.grid_dimensions_range)
            w = random.randint(*self.grid_dimensions_range)
            max_shapes = random.randint(*self.max_shapes_range)
            bar_orientation = random.choice(self.bar_orientations)
            holes_fraction = random.uniform(*self.holes_fraction_range)

            builder = GravityGridBuilder(
                height=h,
                width=w,
                max_shapes=max_shapes,
                bar_orientation=bar_orientation,
                holes_fraction=holes_fraction,
                bg_color=bg_color,
                bar_color=bar_color,
            )
            builder.resampler = OnlyShapesRepeated()

            return builder

        return sampler

    def transform(self, grid: BSTGridBruteForce) -> BSTGridBruteForce:
        root = grid.tree.root

        assert root is not None

        orientation = _get_orientation(root.key)

        # Create new output grid.
        if orientation == BarOrientation.H:
            output_grid = BSTGridBruteForce(
                grid.height,
                grid.width,
                margin=grid.margin,
                mode=BinaryRelation.BelowOf,
                bg_color=grid._bg_color,
            )
        else:
            output_grid = BSTGridBruteForce(
                grid.height,
                grid.width,
                margin=grid.margin,
                mode=BinaryRelation.LeftOf,
                bg_color=grid._bg_color,
            )

        output_grid.add_shape(root.key, padding=0, no_bbox=True)

        if orientation == BarOrientation.H:
            self._shift_shapes_vertically(
                root_input=root, output_grid=output_grid, remove_if_limit=True
            )
        else:
            self._shift_shapes_horizontally(
                root_input=root, output_grid=output_grid, remove_if_limit=True
            )

        return output_grid


class Gravitate(DropBidirectional):
    def __init__(
        self,
        max_shape_dimesion: float = 3,
        grid_dimensions_range: Tuple[int, int] = (20, 25),
        max_shapes_range: Tuple[int, int] = (5, 15),
        bar_orientations: Tuple[BarOrientation, ...] = (
            BarOrientation.H,
            BarOrientation.V,
        ),
        holes_fraction_range: Tuple[float, float] = (0, 2 / 4),
    ) -> None:
        super().__init__(
            max_shape_dimesion,
            grid_dimensions_range,
            max_shapes_range,
            bar_orientations,
            holes_fraction_range,
        )

    def grid_sampler(self) -> Callable[[], GravityGridBuilder]:
        """
        Creates a grid builder with random parameters.
        """
        bg_color = random.choice(list(ALLOWED_COLORS))
        bar_color = random.choice(list(ALLOWED_COLORS - {bg_color}))
        bar_orientation = random.choice(self.bar_orientations)

        def sampler():
            h = random.randint(*self.grid_dimensions_range)
            w = random.randint(*self.grid_dimensions_range)

            if math.inf in self.max_shapes_range:
                max_shapes = math.inf
            else:
                max_shapes = random.randint(*self.max_shapes_range)

            holes_fraction = 0.0

            return GravityGridBuilder(
                height=h,
                width=w,
                max_shapes=max_shapes,
                bar_orientation=bar_orientation,
                holes_fraction=holes_fraction,
                bg_color=bg_color,
                bar_color=bar_color,
                no_bar=True,
            )

        return sampler
