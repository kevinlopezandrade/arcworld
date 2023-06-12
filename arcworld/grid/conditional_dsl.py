import random
from typing import Any, List, Tuple

from arcworld.deprecated.generator_utils import get_locations
from arcworld.dsl.arc_types import Coordinate, Grid, Shape, Shapes
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


# TODO: I can define this interface more abstractly since different grids
# might have different strategies to place the objects.
class ConditionalGridSampler:
    """
    Samples an input grid. Based on the conditions we want.
    """

    def __init__(
        self,
        margin: int,
        grid_dimensions_range: Coordinate,
        num_objects_range: Coordinate,
        max_obj_dimension: int,
        background_color_options: Tuple[int, ...],
        palette_schemes: Tuple[int, ...],
    ):
        self._margin = margin
        self._min_dim, self._max_dim = grid_dimensions_range
        self._num_objects_range = num_objects_range
        self._max_obj_dimension = max_obj_dimension
        self._background_color_options = background_color_options
        self._palette_schemes = palette_schemes

        # Speedup factor is also necessary, but I have to ask Michael.

    @property
    def min_dim(self) -> int:
        return self._min_dim

    @property
    def max_dim(self) -> int:
        return self._max_dim

    @property
    def background_color_options(self) -> Tuple[int]:
        return self._background_color_options

    @property
    def num_objects_range(self) -> Coordinate:
        return self._num_objects_range

    @property
    def max_obj_dimension(self) -> int:
        return self._max_obj_dimension

    @property
    def margin(self) -> int:
        return self._margin

    @property
    def palette_schemes(self) -> Tuple[int]:
        return self._palette_schemes

    def sample_input_grid(
        self, satisfying_objects: Shapes, unsatisfying_objects: Shapes
    ) -> GridDSL:
        h = random.randint(self.min_dim, self.max_dim)
        w = random.randint(self.min_dim, self.max_dim)
        background_color = random.choice(self.background_color_options)

        # Construct the Grid Object
        grid = GridDSL(
            h,
            w,
            self.max_obj_dimension,
            self.margin,
            background_color,
            self.palette_schemes,
        )

        min_num_objs, max_num_objs = self.num_objects_range
        num_objs = random.randint(min_num_objs, max_num_objs)

        # So far this value is random, but maybe the user
        # might be able to choose it.
        num_satisfying_objs = random.randint(1, num_objs - 1)
        num_unsatisfying_objs = num_objs - num_satisfying_objs

        # These dictionaries are required since we
        # are choosing the order of sampling at random later.
        groups = {
            "satisfying": satisfying_objects,
            "unsatisfying": unsatisfying_objects,
        }
        must_place = {
            "satisfying": num_satisfying_objs,
            "unsatisfying": num_unsatisfying_objs,
        }
        num_placed = {"satisfying": 0, "unsatisfying": 0}

        fill_order = ["satisfying", "unsatisfying"][:: random.choice([-1, 1])]

        for group in fill_order:
            while num_placed[group] < must_place[group]:
                random_object = random.choice(
                    tuple(groups[group])
                )  # I don't like casting to tuple, check how to it differently.

                try:
                    grid.place_object(random_object)
                except ValueError:
                    raise ValueError("Could not fullfil requirements")
                else:
                    num_placed[group] = num_placed[group] + 1

        return grid
