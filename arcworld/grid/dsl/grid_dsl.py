import random
from typing import Any, List, Tuple

from arcworld.deprecated.generator_utils import get_locations
from arcworld.dsl.arc_types import Coordinate, Grid, Shape
from arcworld.dsl.functional import (
    add,
    apply,
    astuple,
    backdrop,
    canvas,
    double,
    height,
    paint,
    rbind,
    recolor,
    shift,
    width,
)


class GridDSL:
    def __init__(
        self,
        h: int,
        w: int,
        max_obj_dimension: int,
        margin: int,
        background_color: int,
        palette_schemes: Tuple[int],
    ) -> None:
        self._h = h
        self._w = w
        self._max_obj_dimension = max_obj_dimension
        self._margin = margin
        self._background_color = background_color
        self._palette_schemes = palette_schemes
        self._grid_shape = (h, w)

        self._grid: Grid = canvas(background_color, (h, w))
        self._free_locations: set[Coordinate] = get_locations(
            (h, w), max_obj_dimension, margin, 2
        )

        self._objects: List[Shape] = []

    @property
    def grid(self) -> Grid:
        return self._grid

    @grid.setter
    def grid(self, value: Grid):
        self._grid = value

    @property
    def objects(self) -> Tuple[Shape]:
        """
        The user is not expected to mutate this list.
        So we return the a tuple of the internal
        object list.
        """
        return tuple(self._objects)

    @property
    def free_locations(self) -> set[Coordinate]:
        return self._free_locations

    @free_locations.setter
    def free_locations(self, value: set[Coordinate]):
        self._free_locations = value

    @property
    def max_obj_dimension(self) -> int:
        return self._max_obj_dimension

    @property
    def margin(self) -> int:
        return self._margin

    @property
    def height(self) -> int:
        return self._h

    @property
    def width(self) -> int:
        return self._w

    @property
    def shape(self) -> Coordinate:
        return self._grid_shape

    @property
    def palette_schemes(self) -> Any:
        return self._palette_schemes

    def prune_locations(self, obj: Shape):
        """
        Basically prunes the avalaible locations
        given an object coordinates.
        """
        d = self.max_obj_dimension + 4 * self.margin
        for i, j in backdrop(obj):
            locations_pruned: set[Coordinate] = set()
            for a, b in self.free_locations:
                if i < a or i >= a + d or j < b or j >= b + d:
                    locations_pruned.add((a, b))
            self.free_locations = locations_pruned

    def place_object(self, obj: Shape) -> Shape:
        """
        Places the shape in a random position in the grid.

        Args:
            shape: Is a normalized object. (Must be) Test it
                with an assert.
        Returns:
            The object coordinates where it is placed in the
            grid.

        Raises:
            ValueError: For the if no possible locations are left.
        """
        if len(self.free_locations) == 0:
            raise ValueError("No available locations")

        shift_vector = double(astuple(self.margin, self.margin))
        shift_function = rbind(add, shift_vector)
        locations_shifted = apply(shift_function, self.free_locations)
        loc_base = random.choice(tuple(locations_shifted))
        offset_i = random.randint(0, self.max_obj_dimension - height(obj))
        offset_j = random.randint(0, self.max_obj_dimension - width(obj))
        offset = (offset_i, offset_j)
        location = add(loc_base, offset)
        obj = shift(obj, location)

        object_color = random.choice(self.palette_schemes)
        obj = recolor(object_color, obj)

        # Update the grid.
        self.grid = paint(self.grid, obj)
        self._objects.append(obj)

        # Update the available locations.
        self.prune_locations(obj)

        return obj
