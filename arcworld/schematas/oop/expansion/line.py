from typing import Callable, List, Set, Tuple, cast

from arcworld.dsl.arc_types import Cell, Coordinate, Shape
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
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
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
            for _, v in grid.lines.items():
                for line in v:
                    if {dot[1]} & toindices(line):
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

            if not intersected:
                grid.grid = paint(grid.grid, frozenset({dot}))

        start = dot[1]

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_line)}


def draw_dotted_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
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
                for _, v in grid.lines.items():
                    for line in v:
                        if {dot[1]} & toindices(line):
                            line_policy(frozenset({dot}), line, grid)
                            intersected = True

                if not intersected:
                    grid.grid = paint(grid.grid, frozenset({dot}))

            start = dot[1]
            state = 0

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_line)}


def draw_dashed_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
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
                for _, v in grid.lines.items():
                    for line in v:
                        if {dot[1]} & toindices(line):
                            line_policy(frozenset({dot}), line, grid)
                            intersected = True

                if not intersected:
                    grid.grid = paint(grid.grid, frozenset({dot}))

            start = dot[1]
            if state == "B":
                state = "C"
            elif state == "C":
                state = "A"

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_line)}


def draw_dashed_dot_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
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
                for _, v in grid.lines.items():
                    for line in v:
                        if {dot[1]} & toindices(line):
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

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_line)}


def draw_hidden_line(
    shape: Shape,
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
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
            for _, v in grid.lines.items():
                for line in v:
                    if {dot[1]} & toindices(line):
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

            if not intersected:
                grid.grid = paint(grid.grid, recolor(grid.bg_color, frozenset({dot})))

        start = dot[1]

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_hidden_line)}


def expand_shape(
    index: int,
    shapes: List[Shape],
    visited: List[bool],
    direction: Coordinate,
    grid: LinesGrid,
    line_policy: Callable[[Shape, Shape, LinesGrid], None],
    shape_policy: Callable[[Shape, Shape, LinesGrid], Tuple[Shape, bool]],
):
    shape = shapes[index]

    if len(shape) == 0:
        return

    start = centerofmass(backdrop(shape))
    h = grid.height
    w = grid.width
    c = color(shape)

    new_line: Set[Tuple[int, Coordinate]] = set()

    while 0 <= add(start, direction)[0] < h and 0 <= add(start, direction)[1] < w:
        dot = cast(Coordinate, add(start, direction))
        dot = (c, dot)
        new_line.add(dot)

        # Intersection with a shape.
        if {dot[1]} & grid.occupied:
            for i, candidate_shape in enumerate(shapes):
                if candidate_shape == shape:
                    continue

                if visited[i]:
                    continue

                # Update shapes array.
                if {dot[1]} & toindices(candidate_shape):
                    # shapes[i] = shape_policy(frozenset({dot}), candidate_shape, grid)
                    new_shape, keep_base_shape = shape_policy(
                        frozenset({dot}), candidate_shape, grid
                    )

                    if keep_base_shape:
                        shapes[i] = candidate_shape
                    else:
                        shapes[i] = new_shape

                    visited[i] = True

                    # A transformation can change a shape, leaving the cell
                    # where they intersect empty or as a line, which should be
                    # then follow the line policy. So start again from the same
                    # position, the visited boolean array ensures not infinete
                    # loop.
                    dot = cast(Cell, (c, start))
        else:
            intersected = False
            for _, v in grid.lines.items():
                for line in v:
                    if {dot[1]} & toindices(line):
                        line_policy(frozenset({dot}), line, grid)
                        intersected = True

            if not intersected:
                grid.grid = paint(grid.grid, frozenset({dot}))

        start = dot[1]

    grid.lines[shape] = grid.lines[shape] | {frozenset(new_line)}


STYLES = {
    "standard": draw_standard_line,
    "dashed": draw_dashed_line,
    "dotted": draw_dotted_line,
    "dashdot": draw_dashed_dot_line,
}
