import random
from typing import List, Optional, Set

from arcworld.dsl.arc_types import Object
from arcworld.dsl.functional import backdrop, height, width
from arcworld.filters.functional.shape_filter import FunctionalFilter
from arcworld.filters.objects_filter import ObjectsFilter
from arcworld.schematas.oop.expansion.grid import (
    ExpansionGridBuilder,
    IntersectionGridBuilder,
    LinesGrid,
)
from arcworld.schematas.oop.expansion.intersection_line import POLICIES as LINE_POLICIES
from arcworld.schematas.oop.expansion.intersection_shape import (
    POLICIES as SHAPE_POLICIES,
)
from arcworld.schematas.oop.expansion.line import STYLES, expand_shape


def is_bbox_odd(shape: Object) -> bool:
    bbox = backdrop(shape)
    h = height(bbox)
    w = width(bbox)

    if h == w and h % 2 == 1:
        return True
    else:
        return False


def dot_filter(shape: Object) -> bool:
    if width(shape) <= 1 and height(shape) <= 1:
        return True
    else:
        return False


class DotsExpansion:
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

    def __init__(
        self,
        linestyle: Optional[str] = None,
        directions: Optional[Set[str]] = None,
        line_policy: Optional[str] = None,
    ) -> None:
        if linestyle:
            self.linestyle = linestyle
        else:
            self.linestyle = random.choice(list(STYLES.keys()))

        if directions:
            self.directions = set(directions)
        else:
            self.directions = set(random.sample(list(self.DIRECTIONS.keys()), k=4))

        if line_policy:
            self.line_policy = line_policy
        else:
            self.line_policy = random.choice(list(LINE_POLICIES.keys()))

        # A line cannot intersect another dot in this transformation
        # it can only intersect dots.
        self.shape_policy = "no_op"

        self.program = (
            f"{self.__class__.__name__}"
            f"#{'_'.join(sorted(self.directions))}"
            f"#{self.linestyle}"
            f"#{self.line_policy}"
        )

    @property
    def filters(self) -> List[ObjectsFilter]:
        filter = FunctionalFilter(name="DIM_1", func=dot_filter)

        return [filter]

    def transform(self, input_grid: LinesGrid) -> LinesGrid:
        grid = input_grid.clone_no_shapes()
        for shape in input_grid.objects:
            grid.add_object(shape, no_bbox=True)

        # Start from the top dot.
        for dot in grid.objects[::-1]:
            for dir in self.directions:
                STYLES[self.linestyle](
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
            h = random.randint(20, 25)
            w = random.randint(20, 25)
            max_dots = random.randint(3, 4)

            return ExpansionGridBuilder(
                height=h,
                width=w,
                max_dots=max_dots,
                bg_color=bg_color,
                directions=self.directions,
            )

        return sampler


class ObjectsExpansion(DotsExpansion):
    @property
    def filters(self) -> List[ObjectsFilter]:
        filter = FunctionalFilter(name="IS_BBOX_ODD", func=is_bbox_odd)

        return [filter]

    def grid_sampler(self):
        bg_color = random.randint(0, 9)

        def sampler():
            h = random.randint(10, 12)
            w = random.randint(10, 12)
            max_dots = random.randint(2, 5)

            return IntersectionGridBuilder(
                height=h,
                width=w,
                max_dots=max_dots,
                bg_color=bg_color,
                directions=self.directions,
            )

        return sampler

    def transform(self, input_grid: LinesGrid) -> LinesGrid:
        grid = input_grid.clone_no_shapes()
        for shape in input_grid.objects:
            grid.add_object(shape, no_bbox=True)

        shapes = grid.objects[::-1]
        visited = [False for _ in shapes]

        for i in range(len(shapes)):
            for dir in self.directions:
                expand_shape(
                    i,
                    shapes,
                    visited,
                    self.DIRECTIONS[dir],
                    grid,
                    LINE_POLICIES[self.line_policy],
                    SHAPE_POLICIES[self.shape_policy],
                )

        return grid
