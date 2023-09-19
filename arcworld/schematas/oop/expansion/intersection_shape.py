from typing import Tuple, cast

from arcworld.dsl.arc_types import Coordinates, Shape
from arcworld.dsl.functional import (
    backdrop,
    centerofmass,
    color,
    fill,
    hmirror,
    normalize,
    outbox,
    paint,
    recolor,
    shift,
    toindices,
    ulcorner,
    vmirror,
)
from arcworld.schematas.oop.expansion.grid import LinesGrid

# Every Transformation should return the transformed shape plus a boolean
# value specifying if the object keeps its base form or not for the line
# drawing algorithm.


def no_op(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    return shape, True


def paint_outbox(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    new_shape = frozenset(recolor(color(shape), outbox(shape)) | shape)
    grid.grid = paint(grid.grid, new_shape)
    grid.occupied = grid.occupied | toindices(new_shape)
    return new_shape, True


def delete(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    grid.grid = fill(grid.grid, grid.bg_color, toindices(shape))
    grid.occupied = grid.occupied - toindices(shape)

    grid.grid = paint(grid.grid, dot)

    # Collect lines from other shapes
    extra: Coordinates = frozenset({})
    for k in grid.lines.keys():
        if k != shape:
            for line in grid.lines[k]:
                extra = extra | toindices(line)

    # Delete lines of the shape if it has some.
    lines = grid.lines[shape]
    for line in lines:
        grid.grid = fill(grid.grid, grid.bg_color, toindices(line) - extra)

    del grid.lines[shape]

    return frozenset({}), False


def extend_color(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    new_shape = recolor(color(dot), shape)
    grid.grid = paint(grid.grid, new_shape)

    return new_shape, True


def make_dot(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    grid.grid = fill(grid.grid, grid.bg_color, toindices(shape))

    # Construct the new shape
    new_shape = recolor(color(shape), frozenset({centerofmass(backdrop(shape))}))
    grid.grid = paint(grid.grid, new_shape)

    # Keeps its lines if it had any
    lines = frozenset(coord for line in grid.lines[shape] for coord in toindices(line))
    lines = backdrop(shape) & lines
    grid.grid = paint(grid.grid, recolor(color(shape), lines))

    # Udate the occupied
    grid.occupied = grid.occupied - toindices(shape)
    grid.occupied = grid.occupied | toindices(new_shape)

    return new_shape, False


def fill_bbox(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    new_shape = recolor(color(dot), backdrop(shape) - toindices(shape))
    grid.grid = paint(grid.grid, new_shape)
    grid.occupied = grid.occupied | toindices(new_shape)

    return new_shape, True


def flip_vertically(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    grid.grid = fill(grid.grid, grid.bg_color, toindices(shape))
    grid.occupied = grid.occupied - toindices(shape)

    new_shape = cast(Shape, vmirror(shape))
    grid.grid = paint(grid.grid, new_shape)
    grid.occupied = grid.occupied | toindices(new_shape)

    return new_shape, False


def flip_horizontally(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    grid.grid = fill(grid.grid, grid.bg_color, toindices(shape))
    grid.occupied = grid.occupied - toindices(shape)

    new_shape = cast(Shape, hmirror(shape))
    grid.grid = paint(grid.grid, new_shape)
    grid.occupied = grid.occupied | toindices(new_shape)

    return new_shape, False


def rotate_90(dot: Shape, shape: Shape, grid: LinesGrid) -> Tuple[Shape, bool]:
    grid.grid = fill(grid.grid, grid.bg_color, toindices(shape))
    grid.occupied = grid.occupied - toindices(shape)

    new_shape = recolor(color(shape), frozenset((-j, i) for i, j in toindices(shape)))
    new_shape = normalize(new_shape)
    new_shape = shift(new_shape, ulcorner(backdrop(shape)))
    new_shape = cast(Shape, new_shape)

    grid.grid = paint(grid.grid, new_shape)
    grid.occupied = grid.occupied | toindices(new_shape)

    return new_shape, False


POLICIES = {
    "no_op": no_op,
    "paint_outbox": paint_outbox,
    "delete": delete,
    "extend_color": extend_color,
    "make_dot": make_dot,
    "flip_vertically": flip_vertically,
    "flip_horizontally": flip_horizontally,
    "rotate_90": rotate_90,
    "fill_bbox": fill_bbox,
}
