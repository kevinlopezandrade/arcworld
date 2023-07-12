from __future__ import annotations

import math
import random
from enum import Enum
from typing import Callable, Optional, Tuple

from arcworld.dsl.arc_types import Coordinates, Shapes
from arcworld.dsl.functional import normalize, recolor, width
from arcworld.grid.oop.grid_bruteforce import BinaryRelation, BSTGridBruteForce
from arcworld.internal.constants import ALLOWED_COLORS, DoesNotFitError
from arcworld.schematas.oop.subgrid_pickup.resamplers import Resampler


def _draw_holes(proto_shape: Coordinates, n_holes: int) -> Coordinates:
    """
    Assuming horizontal proto bar, randomly removes n_holes indices to create
    holes.
    """
    # Compute width an drop n pixels.
    w = width(proto_shape)
    holes_index = random.choices(list(range(w)), k=n_holes)

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


class BarPos(Enum):
    V = 0
    H = 1


class DropGridBuilder:
    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        bg_color: int = 0,
        bar_color: int = 2,
        max_shapes: float = math.inf,
        bar_orientation: BarPos = BarPos.H,
        holes_fraction: float = 0,
    ) -> None:
        self.height = height
        self.width = width
        self.max_shapes = max_shapes
        self.bar_orientation = bar_orientation
        self.holes_fraction = holes_fraction
        self.bg_color = bg_color
        self.bar_color = bar_color
        self.resampler: Optional[Resampler] = None

        self._available_colors = ALLOWED_COLORS - {bg_color} - {bar_color}

    @classmethod
    def sampler(
        cls,
        grid_dimensions_range: Tuple[int, int] = (10, 30),
        max_shapes_range: Tuple[float, float] = (3, 10),
        bar_orientations: Tuple[BarPos, ...] = (BarPos.H, BarPos.V),
        holes_fraction_range: Tuple[float, float] = (0, 2 / 4),
    ) -> Callable[[], DropGridBuilder]:
        """
        Creates a grid builder with random parameters.
        """
        bg_color = random.choice(list(ALLOWED_COLORS))
        bar_color = random.choice(list(ALLOWED_COLORS - {bg_color}))

        def sampler():
            h = random.randint(*grid_dimensions_range)
            w = random.randint(*grid_dimensions_range)

            if math.inf in max_shapes_range:
                max_shapes = math.inf
            else:
                max_shapes = random.randint(*max_shapes_range)

            bar_orientation = random.choice(bar_orientations)
            holes_fraction = random.uniform(*holes_fraction_range)

            return cls(
                height=h,
                width=w,
                max_shapes=max_shapes,
                bar_orientation=bar_orientation,
                holes_fraction=holes_fraction,
                bg_color=bg_color,
                bar_color=bar_color,
            )

        return sampler

    def _construct_base_form(self) -> BSTGridBruteForce:
        """
        Base form of the grid has the bar in horizontal position.
        """
        if self.bar_orientation == BarPos.H:
            grid = BSTGridBruteForce(
                self.height,
                self.width,
                mode=BinaryRelation.BelowOf,
                bg_color=self.bg_color,
            )
            proto_bar = _construct_bar(grid.width, self.holes_fraction)

            # Choose a random position to place the bar.
            random_pos = (random.randint(0, grid.height), 0)
        else:
            grid = BSTGridBruteForce(
                self.height,
                self.width,
                mode=BinaryRelation.LeftOf,
                bg_color=self.bg_color,
            )
            proto_bar = _construct_bar(grid.height, self.holes_fraction)
            # Rotate the horizontal bar.
            proto_bar = _rot90clockwise(proto_bar)

            # Choose a random position to place the bar.
            random_pos = (0, random.randint(0, grid.width))

        bar = recolor(self.bar_color, proto_bar)
        grid.place_object_deterministic(bar, random_pos)

        return grid

    def build_input_grid(self, shapes: Shapes) -> BSTGridBruteForce:
        grid = self._construct_base_form()

        N = self.max_shapes  # noqa

        if N < math.inf:
            if self.resampler is None:
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
                grid.place_object(shape, color_palette=self._available_colors)
            except DoesNotFitError:
                pass
            else:
                placed += 1

        if placed == 0:
            raise RuntimeError("Could not place any shape in the grid")

        return grid
