from __future__ import annotations

import math
import random
from enum import Enum
from typing import Optional

from arcdsl.arc_types import Coordinates, Objects
from arcdsl.dsl import normalize, recolor, shift, toindices, width

from arcworld.grid.oop.grid_bruteforce import BinaryRelation, BSTGridBruteForce
from arcworld.internal.constants import ALLOWED_COLORS, DoesNotFitError
from arcworld.shape.resamplers import Resampler


def _draw_holes(proto_shape: Coordinates, n_holes: int) -> Coordinates:
    """
    Assuming horizontal proto bar, randomly removes n_holes indices to create
    holes.
    """
    # Compute width an drop n pixels.
    w = width(proto_shape)
    holes_index = random.sample(list(range(w)), k=n_holes)

    shape_with_holes = set(proto_shape)
    for index in holes_index:
        shape_with_holes = shape_with_holes - {(0, index)}

    return frozenset(shape_with_holes)


def _rot90clockwise(proto_shape: Coordinates):
    return normalize(frozenset((-c[1], c[0]) for c in proto_shape))


def _construct_bar(upper_bound_expand: int, holes_fraction: float) -> Coordinates:
    """
    Constructs an horizontal proto bar, with n_holes holes.
    """
    # Assume its horizontal.
    shape = frozenset((0, i) for i in range(upper_bound_expand))
    n_holes = math.floor(holes_fraction * width(shape))

    if n_holes > 0:
        shape = _draw_holes(shape, n_holes)

    return shape


class BarOrientation(Enum):
    V = 0
    H = 1


class GravityGridBuilder:
    """
    Inspired in : f83cb3f6.json , 5ffb2104.json, 6ad5bdfd.json
    """

    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        bg_color: int = 0,
        bar_color: int = 2,
        max_shapes: float = math.inf,
        bar_orientation: BarOrientation = BarOrientation.H,
        holes_fraction: float = 0,
        no_bar: bool = False,
    ) -> None:
        self.height = height
        self.width = width
        self.max_shapes = max_shapes
        self.bar_orientation = bar_orientation
        self.holes_fraction = holes_fraction
        self.bg_color = bg_color
        self.bar_color = bar_color
        self.resampler: Optional[Resampler] = None
        self.no_bar = no_bar

        self._available_colors = ALLOWED_COLORS - {bg_color} - {bar_color}

    def _construct_base_form(self) -> BSTGridBruteForce:
        """
        Base form of the grid has the bar in horizontal position.
        """
        if self.bar_orientation == BarOrientation.H:
            grid = BSTGridBruteForce(
                self.height,
                self.width,
                mode=BinaryRelation.BelowOf,
                bg_color=self.bg_color,
            )
            # Choose a random position to place the bar.
            if self.no_bar:
                random_pos = (grid.height, 0)
            else:
                random_pos = (random.randint(0, grid.height - 1), 0)

            # Mark the edge as occupied.
            edge = _construct_bar(grid.width, 0)
            grid._occupied = grid._occupied | toindices(shift(edge, random_pos))

            # Create the bar with holes.
            proto_bar = _construct_bar(grid.width, self.holes_fraction)
            bar = recolor(self.bar_color, proto_bar)
            grid.place_object(bar, random_pos)

        else:
            grid = BSTGridBruteForce(
                self.height,
                self.width,
                mode=BinaryRelation.LeftOf,
                bg_color=self.bg_color,
            )
            # Choose a random position to place the bar.
            if self.no_bar:
                random_pos = (0, grid.width)
            else:
                random_pos = (0, random.randint(0, grid.width - 1))

            # Mark the edge as occupied.
            edge = _construct_bar(grid.height, 0)
            edge = _rot90clockwise(edge)
            grid._occupied = grid._occupied | toindices(shift(edge, random_pos))

            # Create the bar with holes.
            proto_bar = _construct_bar(grid.height, self.holes_fraction)
            # Rotate the horizontal bar.
            proto_bar = _rot90clockwise(proto_bar)
            bar = recolor(self.bar_color, proto_bar)
            grid.place_object(bar, random_pos)

        return grid

    def build_input_grid(self, shapes: Objects) -> BSTGridBruteForce:
        grid = self._construct_base_form()

        N = self.max_shapes  # noqa

        if N < math.inf:
            if self.resampler is None:
                # TODO: Decide which strategy is better either sample
                # without replacement or with replacement.
                sampled_shapes = random.choices(list(shapes), k=int(N))
            else:
                sampled_shapes = self.resampler.resample(
                    shapes, n_shapes_per_grid=int(N)
                )
        else:
            # Just shuffle the shapes.
            sampled_shapes = list(shapes)
            random.shuffle(sampled_shapes)

        placed = 0
        for shape in sampled_shapes:
            try:
                grid.place_object_random(shape, color_palette=self._available_colors)
            except DoesNotFitError:
                pass
            else:
                placed += 1

        if placed == 0:
            raise RuntimeError("Could not place any shape in the grid")

        return grid
