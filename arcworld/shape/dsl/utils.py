from arcworld.dsl.arc_types import Coordinates, Grid, Shape
from arcworld.dsl.functional import (
    add,
    cover,
    lrcorner,
    move,
    paint,
    shift,
    subtract,
    toindices,
    ulcorner,
)


def proto_vbar(x: int, h: int) -> Coordinates:
    return frozenset((i, x) for i in range(h))


def switch_shapes(grid: Grid, shape_a: Shape, shape_b: Shape):
    # Delete shape_b
    grid = cover(grid, shape_b)

    # Move shape_a to pos_b
    grid = move(grid, shape_a, subtract(ulcorner(shape_b), ulcorner(shape_a)))

    # Move shape_b to pos_a witout deleting the already placed shape.
    grid = paint(grid, shift(shape_b, subtract(ulcorner(shape_a), ulcorner(shape_b))))

    return grid


def bbox(shape: Shape, padding: int = 0) -> Coordinates:
    """
    Computes the bounding box coodinates of a shape
    with an optional padding paremeter to increase
    the dimension of the bounding box.

    Args:
        shape: Shape for which to compute the bounding box
        margin: Extra padding to add to the bounding box
    """
    indices = toindices(shape)
    si, sj = add(ulcorner(indices), (-padding, -padding))
    ei, ej = add(lrcorner(indices), (padding, padding))

    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))
