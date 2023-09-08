from typing import Callable, Set, Tuple, cast

from arcworld.dsl.arc_types import Coordinate, Shape
from arcworld.dsl.functional import add, backdrop, centerofmass, color, paint, toindices
from arcworld.schematas.oop.expansion.grid import LinesGrid


def draw_standard_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], None],
):
    """
    Draws a line starting from the center of mass of the shape. Each dot in the line
    is a pixel with the same color as the original shape. Intersection with a shape
    is defined only if a dot of the current expansion intersects with a cell of the
    shape, not with its bounding box. The line is drawed until the limits of the
    grid.
    """
    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_line: Set[Tuple[int, Coordinate]] = set()

    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)
        new_line.add(dot)

        if len({dot[1]} & grid.occupied) > 0:
            for shape in grid.shapes:
                if len({dot} & toindices(shape)) > 0:
                    shape_policy(frozenset({dot}), shape, grid)
        else:
            intersected = False
            for line in grid.lines:
                if len({dot[1]} & toindices(line)) > 0:
                    line_policy(frozenset({dot}), line, grid)
                    intersected = True

            if not intersected:
                grid.grid = paint(grid.grid, frozenset({dot}))

        start = dot[1]

    grid.lines.append(frozenset(new_line))
