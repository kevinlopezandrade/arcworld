import random
from collections import defaultdict
from typing import Dict, Set, cast

from arcworld.dsl.arc_types import Coordinate, Shape, Shapes
from arcworld.dsl.functional import add, centerofmass, color, multiply
from arcworld.grid.oop.grid_bruteforce import BinaryRelation, BSTGridBruteForce
from arcworld.internal.constants import ALLOWED_COLORS, DoesNotFitError

# Defintions:
# dot := Any shape with a square bounding box
# line := A set of adjacent dots.


class LinesGrid(BSTGridBruteForce):
    def __init__(
        self,
        h: int,
        w: int,
        bg_color: int = 0,
        margin: int = 2,
        mode: BinaryRelation = BinaryRelation.BelowOf,
    ) -> None:
        super().__init__(h, w, bg_color, margin, mode)
        self._lines: Dict[Shape, Shapes] = defaultdict(frozenset)

    @property
    def lines(self) -> Dict[Shape, Shapes]:
        return self._lines

    def place_shape_random(
        self, shape: Shape, color_palette: Set[int] | None = None
    ) -> Shape:
        shifted_shape = super().place_shape_random(shape, color_palette)
        orig = centerofmass(shifted_shape)

        # Occupy horizontal bar
        self.occupied = self.occupied | frozenset(
            (orig[0], i) for i in range(self.width)
        )

        # Occupy vertical bar
        self.occupied = self.occupied | frozenset(
            (i, orig[1]) for i in range(self.height)
        )

        # Occupy negative diagonal
        self.occupied = self.occupied | frozenset(
            cast(Coordinate, add(orig, multiply(i, (1, 1))))
            for i in range(
                1, min(self.height - 1 - orig[0], self.width - 1 - orig[1]) + 1
            )
        )

        self.occupied = self.occupied | frozenset(
            cast(Coordinate, add(orig, multiply(i, (-1, -1))))
            for i in range(1, min(orig[0], orig[1]) + 1)
        )

        # Occupy positive diagonal
        self.occupied = self.occupied | frozenset(
            cast(Coordinate, add(orig, multiply(i, (-1, 1))))
            for i in range(1, min(orig[0], self.width - 1 - orig[1]) + 1)
        )

        self.occupied = self.occupied | frozenset(
            cast(Coordinate, add(orig, multiply(i, (1, -1))))
            for i in range(1, min(self.height - 1 - orig[0], orig[1]) + 1)
        )

        return shifted_shape

    def clone_no_shapes(self) -> "LinesGrid":
        grid = LinesGrid(self.height, self.width, self.bg_color, self.margin)
        grid._binary_relation = self._binary_relation

        return grid


class ExpansionGridBuilder:
    def __init__(
        self, height: int = 20, width: int = 20, bg_color: int = 0, max_dots: int = 5
    ) -> None:
        self.height = height
        self.width = width
        self.max_dots = max_dots
        self.bg_color = bg_color

        self._avalaible_colors = ALLOWED_COLORS - {bg_color}

    def _base_form(self):
        """
        Base form of the grid.
        """
        grid = LinesGrid(self.height, self.width, bg_color=self.bg_color)

        return grid

    def build_input_grid(self, shapes: Shapes) -> LinesGrid:
        grid = self._base_form()

        N = self.max_dots  # noqa
        sampled_shapes = random.choices(list(shapes), k=N)

        placed = 0
        for shape in sampled_shapes:
            try:
                placed_shape = grid.place_shape_random(
                    shape, color_palette=self._avalaible_colors
                )
            except DoesNotFitError:
                pass
            else:
                self._avalaible_colors = self._avalaible_colors - {color(placed_shape)}
                placed += 1

        if placed == 0:
            raise RuntimeError("Could not place any shape in the grid")

        return grid
