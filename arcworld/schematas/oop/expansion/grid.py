import random
from collections import defaultdict
from typing import Dict, Set, cast

from arcworld.dsl.arc_types import Coordinate, Coordinates, Shape, Shapes
from arcworld.dsl.functional import (
    add,
    backdrop,
    centerofmass,
    color,
    height,
    multiply,
    recolor,
    shift,
    toindices,
    width,
)
from arcworld.grid.oop.grid_bruteforce import (
    BinaryRelation,
    BSTGridBruteForce,
    bounding_box,
    rectangle,
)
from arcworld.internal.constants import ALLOWED_COLORS, DoesNotFitError


class LinesGrid(BSTGridBruteForce):
    def __init__(
        self,
        h: int,
        w: int,
        bg_color: int = 0,
        margin: int = 2,
        mode: BinaryRelation = BinaryRelation.BelowOf,
        directions: Set[str] | None = None,
    ) -> None:
        super().__init__(h, w, bg_color, margin, mode)
        self._lines: Dict[Shape, Shapes] = defaultdict(frozenset)

        if directions is not None:
            self.directions: Set[str] = directions
        else:
            self.directions: Set[str] = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}

    @property
    def lines(self) -> Dict[Shape, Shapes]:
        return self._lines

    @staticmethod
    def _possible_lines(
        directions: Set[str], height: int, width: int, shape: Shape
    ) -> Coordinates:
        orig = centerofmass(backdrop(shape))
        lines: Coordinates = frozenset()

        # Occupy horizontal bar
        if directions & {"W"}:
            lines = lines | frozenset((orig[0], i) for i in range(orig[1]))

        if directions & {"E"}:
            lines = lines | frozenset((orig[0], i) for i in range(orig[1] + 1, width))

        # Occupy vertical bar
        if directions & {"N"}:
            lines = lines | frozenset((i, orig[1]) for i in range(orig[0]))

        if directions & {"S"}:
            lines = lines | frozenset((i, orig[1]) for i in range(orig[0] + 1, height))

        # Occupy negative diagonal
        if directions & {"SE"}:
            lines = lines | frozenset(
                cast(Coordinate, add(orig, multiply(i, (1, 1))))
                for i in range(1, min(height - 1 - orig[0], width - 1 - orig[1]) + 1)
            )

        if directions & {"NW"}:
            lines = lines | frozenset(
                cast(Coordinate, add(orig, multiply(i, (-1, -1))))
                for i in range(1, min(orig[0], orig[1]) + 1)
            )

        # Occupy positive diagonal
        if directions & {"NE"}:
            lines = lines | frozenset(
                cast(Coordinate, add(orig, multiply(i, (-1, 1))))
                for i in range(1, min(orig[0], width - 1 - orig[1]) + 1)
            )

        if directions & {"SW"}:
            lines = lines | frozenset(
                cast(Coordinate, add(orig, multiply(i, (1, -1))))
                for i in range(1, min(height - 1 - orig[0], orig[1]) + 1)
            )

        return lines

    def _prune_possible_lines(self, shape: Shape):
        self.occupied = self.occupied | LinesGrid._possible_lines(
            self.directions, self.height, self.width, shape
        )

    def place_shape_random(
        self, shape: Shape, color_palette: Set[int] | None = None
    ) -> Shape:
        shifted_shape = super().place_shape_random(shape, color_palette)
        self._prune_possible_lines(shifted_shape)

        return shifted_shape

    def place_shape_on_intersection(
        self, shape: Shape, color_palette: Set[int] | None = None
    ) -> Shape:
        # Define boundaries
        h = height(shape)
        w = width(shape)

        # The ideal rectangle
        rect = rectangle((0, 0), self.height - h + 1, self.width - w + 1)

        # The coordinates occupied by the objects
        objects = set(
            coord
            for shp in self.shapes
            for coord in toindices(bounding_box(shp, self.margin))
        )

        # Compute an early list of possible locations
        free = (rect - objects) & self.occupied

        # Iterate over all indices in grid and discard the ones
        overlapping: set[Coordinate] = set()
        bbox = bounding_box(shape)
        for i, j in free:
            # Move bbox to (i, j) and check if intersects
            # with the occupied coodinates of the objects
            disp_bbox = cast(Coordinates, shift(bbox, (i, j)))

            if len(disp_bbox & objects) > 0:
                overlapping.add((i, j))

            # Check if it overlaps with the center of mass of some
            # shape already placed in the grid.
            for shp in self.shapes:
                if LinesGrid._possible_lines(
                    self.directions, self.height, self.width, recolor(8, disp_bbox)
                ) & {centerofmass(backdrop(shp))}:
                    overlapping.add((i, j))

        # Update the possible free locations
        free = free - overlapping

        if len(free) == 0:
            raise DoesNotFitError("Shape does not fit")

        # Place the shape in a random cell
        random_cell = random.choice(list(free))

        shifted_shape = cast(Shape, shift(shape, random_cell))

        if color_palette is not None:
            c = random.choice(list(color_palette))
            shifted_shape = recolor(c, shifted_shape)
        else:
            c = random.choice(list(set(range(9)) - {self._bg_color}))
            shifted_shape = recolor(c, shifted_shape)

        self.add_shape(shifted_shape, padding=self.margin)
        self._prune_possible_lines(shifted_shape)

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


class IntersectionGridBuilder:
    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        bg_color: int = 0,
        max_dots: int = 5,
        directions: Set[str] | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.max_dots = max_dots
        self.bg_color = bg_color

        self._directions = directions
        self._avalaible_colors = ALLOWED_COLORS - {bg_color}

    def _base_form(self) -> LinesGrid:
        """
        Base form of the grid.
        """
        grid = LinesGrid(
            self.height, self.width, bg_color=self.bg_color, directions=self._directions
        )

        return grid

    def build_input_grid(self, shapes: Shapes) -> LinesGrid:
        grid = self._base_form()

        N = self.max_dots  # noqa
        sampled_shapes = random.choices(list(shapes), k=N)

        placed = 0
        state = 0
        for shape in sampled_shapes:
            try:
                if state == 0:
                    placed_shape = grid.place_shape_random(
                        shape, color_palette=self._avalaible_colors
                    )
                    state = 1
                else:
                    placed_shape = grid.place_shape_on_intersection(
                        shape, color_palette=self._avalaible_colors
                    )
                    state = 0

            except DoesNotFitError:
                pass
            else:
                self._avalaible_colors = self._avalaible_colors - {color(placed_shape)}
                placed += 1

        if placed == 0:
            raise RuntimeError("Could not place any shape in the grid")

        return grid
