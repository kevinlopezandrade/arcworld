import random
from typing import Tuple

from arcworld.dsl.arc_types import Coordinate, Objects
from arcworld.grid.dsl.grid_dsl import GridDSLOld


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
        self, satisfying_objects: Objects, unsatisfying_objects: Objects
    ) -> GridDSLOld:
        h = random.randint(self.min_dim, self.max_dim)
        w = random.randint(self.min_dim, self.max_dim)
        background_color = random.choice(self.background_color_options)

        # Construct the Grid Object
        grid = GridDSLOld(
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
