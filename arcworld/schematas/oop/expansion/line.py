from typing import Callable, Set, Tuple, cast

from arcworld.dsl.arc_types import Coordinate, Shape
from arcworld.dsl.functional import (
    add,
    backdrop,
    centerofmass,
    color,
    paint,
    recolor,
    toindices,
)
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
            for candidate_shape in grid.shapes:
                if (
                    len({dot[1]} & toindices(candidate_shape)) > 0
                    and candidate_shape != shape
                ):
                    shape_policy(frozenset({dot}), candidate_shape, grid)
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


def draw_dotted_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], None],
):
    """
    Draws a dotted line starting from the center of mass of the shape. Each
    dot in the line is a pixel with the same color as the original shape.
    Intersection with a shape is defined only if a dot of the current expansion
    intersects with a cell of the shape, not with its bounding box. The line is
    drawed until the limits of the grid.
    """
    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_line: Set[Tuple[int, Coordinate]] = set()

    # state = 0 := Not draw a dot
    # state = 1 := Draw a dot
    state = 0
    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)

        if state == 0:
            start = dot[1]
            state = 1
        else:
            new_line.add(dot)

            if len({dot[1]} & grid.occupied) > 0:
                for candidate_shape in grid.shapes:
                    if (
                        len({dot[1]} & toindices(candidate_shape)) > 0
                        and candidate_shape != shape
                    ):
                        shape_policy(frozenset({dot}), candidate_shape, grid)
            else:
                intersected = False
                for line in grid.lines:
                    if len({dot[1]} & toindices(line)) > 0:
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

                if not intersected:
                    grid.grid = paint(grid.grid, frozenset({dot}))

            start = dot[1]
            state = 0

    grid.lines.append(frozenset(new_line))


def draw_dashed_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], None],
):
    """
    Draws a dashed line starting from the center of mass of the shape. Each
    dot in the line is a pixel with the same color as the original shape.
    Intersection with a shape is defined only if a dot of the current expansion
    intersects with a cell of the shape, not with its bounding box. The line is
    drawed until the limits of the grid.
    """
    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_line: Set[Tuple[int, Coordinate]] = set()

    # state = A := Not draw a dot
    # state = B := Draw a dot
    # state = C := Draw a dot
    # Transitions: A -> B -> C -> A
    state = "A"
    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)

        if state == "A":
            start = dot[1]
            state = "B"
        else:
            new_line.add(dot)

            if len({dot[1]} & grid.occupied) > 0:
                for candidate_shape in grid.shapes:
                    if (
                        len({dot[1]} & toindices(candidate_shape)) > 0
                        and candidate_shape != shape
                    ):
                        shape_policy(frozenset({dot}), candidate_shape, grid)
            else:
                intersected = False
                for line in grid.lines:
                    if len({dot[1]} & toindices(line)) > 0:
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

                if not intersected:
                    grid.grid = paint(grid.grid, frozenset({dot}))

            start = dot[1]
            if state == "B":
                state = "C"
            elif state == "C":
                state = "A"

    grid.lines.append(frozenset(new_line))


def draw_dashed_dot_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], None],
):
    """
    Draws a dashed dotted line starting from the center of mass of the shape. Each
    dot in the line is a pixel with the same color as the original shape.
    Intersection with a shape is defined only if a dot of the current expansion
    intersects with a cell of the shape, not with its bounding box. The line is
    drawed until the limits of the grid.
    """
    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_line: Set[Tuple[int, Coordinate]] = set()

    # state = A := Not draw a dot
    # state = B := Draw a dot
    # state = C := Draw a dot
    # state = D := Not draw a dot
    # state = E := Draw a dot
    # Transitions: A -> B -> C -> D -> E -> A
    state = "A"
    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)

        if state == "A":
            start = dot[1]
            state = "B"
        elif state == "D":
            start = dot[1]
            state = "E"
        else:
            new_line.add(dot)

            if len({dot[1]} & grid.occupied) > 0:
                for candidate_shape in grid.shapes:
                    if (
                        len({dot[1]} & toindices(candidate_shape)) > 0
                        and candidate_shape != shape
                    ):
                        shape_policy(frozenset({dot}), candidate_shape, grid)
            else:
                intersected = False
                for line in grid.lines:
                    if len({dot[1]} & toindices(line)) > 0:
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

                if not intersected:
                    grid.grid = paint(grid.grid, frozenset({dot}))

            start = dot[1]
            if state == "B":
                state = "C"
            elif state == "C":
                state = "D"
            elif state == "E":
                state = "A"

    grid.lines.append(frozenset(new_line))


def draw_hidden_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], None],
):
    """
    Draws a hidden line starting from the center of mass of the shape. Each dot in
    the line is a pixel with the same color as the background color. Intersection
    with a shape is defined only if a dot of the current expansion intersects with
    a cell of the shape, not with its bounding box. The line is drawed until the
    limits of the grid.
    """
    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_hidden_line: Set[Tuple[int, Coordinate]] = set()

    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)
        new_hidden_line.add(dot)

        if len({dot[1]} & grid.occupied) > 0:
            for candidate_shape in grid.shapes:
                if (
                    len({dot[1]} & toindices(candidate_shape)) > 0
                    and candidate_shape != shape
                ):
                    shape_policy(frozenset({dot}), candidate_shape, grid)
        else:
            intersected = False
            for line in grid.lines:
                if len({dot[1]} & toindices(line)) > 0:
                    line_policy(frozenset({dot}), line, grid)
                    intersected = True

            if not intersected:
                grid.grid = paint(grid.grid, recolor(grid.bg_color, frozenset({dot})))

        start = dot[1]

    grid.lines.append(frozenset(new_hidden_line))


STYLES = {
    "standard": draw_standard_line,
    "dashed": draw_dashed_line,
    "dotted": draw_dotted_line,
    "dashdot": draw_dashed_dot_line,
}
