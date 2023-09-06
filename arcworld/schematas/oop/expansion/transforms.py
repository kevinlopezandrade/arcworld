import random
from typing import List

from arcworld.dsl.arc_types import Shape
from arcworld.dsl.functional import backdrop, height, width
from arcworld.filters.functional.shape_filter import FunctionalFilter
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.schematas.oop.expansion.grid import ExpansionGridBuilder, LinesGrid
from arcworld.schematas.oop.expansion.intersection_line import POLICIES as LINE_POLICIES
from arcworld.schematas.oop.expansion.intersection_shape import (
    POLICIES as SHAPE_POLICIES,
)
from arcworld.schematas.oop.expansion.line import draw_standard_line

# TODO: I need to define what happens when I intersect an object and transform it.
# Should I update as a new unit of dot, or should I transform an not update anything ?
# For example assume a policy where I delete an object if I intersect it.


def is_bbox_odd(shape: Shape) -> bool:
    bbox = backdrop(shape)
    h = height(bbox)
    w = width(bbox)

    if h == w and h % 2 == 1:
        return True
    else:
        return False


class StandardExpansion:
    """
    Expands over all the dots.
    """

    DIRECTIONS = {
        "N": (-1, 0),
        "NE": (-1, 1),
        "E": (0, 1),
        "SE": (1, 1),
        "S": (1, 0),
        "SW": (1, -1),
        "W": (0, -1),
        "NW": (-1, -1),
    }

    def __init__(self) -> None:
        self.directions: List[str] = random.sample(list(self.DIRECTIONS.keys()), k=2)
        self.line_policy = random.choice(list(LINE_POLICIES.keys()))
        self.shape_policy = random.choice(list(SHAPE_POLICIES.keys()))
        self.program = (
            f"{'_'.join(self.directions)}_{self.line_policy}_{self.shape_policy}"
        )

    @property
    def filters(self) -> List[ShapesFilter]:
        filter = FunctionalFilter(name="SQUARE_BBOX_ODD", func=is_bbox_odd)

        return [filter]

    def transform(self, input_grid: LinesGrid) -> LinesGrid:
        grid = input_grid.clone_no_shapes()
        for shape in input_grid.shapes:
            grid.add_shape(shape, no_bbox=True)

        for dot in grid.shapes:
            for dir in self.directions:
                draw_standard_line(
                    dot,
                    self.DIRECTIONS[dir],
                    grid,
                    LINE_POLICIES[self.line_policy],
                    SHAPE_POLICIES[self.shape_policy],
                )

        return grid

    def grid_sampler(self):
        bg_color = random.randint(0, 9)

        def sampler():
            h = random.randint(10, 25)
            w = random.randint(10, 25)
            max_dots = random.randint(2, 5)

            return ExpansionGridBuilder(
                height=h, width=w, max_dots=max_dots, bg_color=bg_color
            )

        return sampler
