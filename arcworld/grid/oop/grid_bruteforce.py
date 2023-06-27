import random
from typing import List

import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Coordinate, Coordinates, Shape
from arcworld.dsl.functional import (
    add,
    canvas,
    cast,
    height,
    lrcorner,
    paint,
    recolor,
    shift,
    toindices,
    ulcorner,
    width,
)
from arcworld.grid.oop.grid_oop import to_shape_object
from arcworld.internal.constants import DoesNotFitError
from arcworld.shape.oop.base import ShapeObject


def bounding_box(shape: Shape, padding: int = 0) -> Coordinates:
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


def rectangle(start: Coordinate, h: int, w: int) -> Coordinates:
    x, y = start

    rect: set[Coordinate] = set()
    for i in range(x, x + h):
        for j in range(y, y + w):
            rect.add((i, j))

    return frozenset(rect)


class GridBruteForce:
    def __init__(self, h: int, w: int, margin: int = 1) -> None:
        self._h = h
        self._w = w
        self._grid = canvas(0, (h, w))
        self._margin = margin
        self._occupied: set[Coordinate] = set()
        self._shapes: List[Shape] = []

    @property
    def height(self) -> int:
        return self._h

    @property
    def width(self) -> int:
        return self._w

    @property
    def grid(self) -> NDArray[np.int8]:
        return np.array(self._grid, dtype=np.int8)

    @property
    def shapes(self) -> List[ShapeObject]:
        return list(to_shape_object(shape) for shape in self._shapes)

    def place_object(self, shape: Shape) -> Shape:
        """
        Brute force algorithm to place shapes
        in a grid randomly.
        """
        # Define boundaries
        h = height(shape)
        w = width(shape)

        # The ideal rectangle
        rect = rectangle((0, 0), self.height - h + 1, self.width - w + 1)

        # Compute an early list of possible locations
        free = rect - self._occupied

        # Iterate over all indices in grid and discard the ones
        overlapping: set[Coordinate] = set()
        bbox = bounding_box(shape)
        for i, j in free:
            # Move bbox to (i, j) and check if intersects
            # with the occupied coodinates
            disp_bbox = cast(Coordinates, shift(bbox, (i, j)))

            if len(disp_bbox & self._occupied) > 0:
                overlapping.add((i, j))

        # Update the possible free locations
        free = free - overlapping

        if len(free) == 0:
            raise DoesNotFitError("Shape does not fit")

        # Place the shape in a random cell
        random_cell = random.choice(list(free))
        shifted_shape = cast(Shape, shift(shape, random_cell))
        shifted_shape = recolor(random.randint(1, 7), shifted_shape)

        self._grid = paint(self._grid, shifted_shape)
        self._occupied = self._occupied | bounding_box(shifted_shape, self._margin)

        self._shapes.append(shifted_shape)

        return shifted_shape
