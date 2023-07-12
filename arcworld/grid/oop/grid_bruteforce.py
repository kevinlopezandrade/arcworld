import random
from enum import Enum
from typing import List, Optional, Set, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from arcworld.dsl.arc_types import Coordinate, Coordinates, Shape
from arcworld.dsl.functional import (
    add,
    canvas,
    fill,
    height,
    lrcorner,
    paint,
    recolor,
    shift,
    toindices,
    ulcorner,
    width,
)
from arcworld.grid.oop.util import Node, Tree, bst_insert
from arcworld.internal.constants import DoesNotFitError


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
    def __init__(self, h: int, w: int, bg_color: int = 0, margin: int = 1) -> None:
        self._h = h
        self._w = w
        self._bg_color = bg_color
        self._grid = canvas(self._bg_color, (h, w))
        self._margin = margin
        self._occupied: frozenset[Coordinate] = frozenset()
        self._shapes: List[Shape] = []

    @property
    def height(self) -> int:
        return self._h

    @property
    def width(self) -> int:
        return self._w

    @property
    def margin(self) -> int:
        return self._margin

    @property
    def grid(self) -> NDArray[np.uint8]:
        return np.array(self._grid, dtype=np.uint8)

    @property
    def shapes(self) -> List[Shape]:
        return self._shapes

    def _update_grid(self, shape: Shape, padding: int = 0, no_bbox: bool = False):
        """
        Update the grid and the occupied set, with the passed shape.
        """
        self._grid = paint(self._grid, shape)

        if no_bbox:
            self._occupied = self._occupied | toindices(shape)
        else:
            self._occupied = self._occupied | bounding_box(shape, padding=padding)

        self._shapes.append(shape)

    def paint_occupied(self) -> NDArray[np.uint8]:
        painted = fill(self._grid, 9, frozenset(self._occupied))
        return np.array(painted, dtype=np.uint8)

    def place_object_deterministic(
        self, shape: Shape, pos: Coordinate, padding: int = 0, no_bbox: bool = False
    ):
        """
        Assumes a normalized shaped to be placed in the coordinates (y, x)
        in the grid.
        """
        shifted_shape = shift(shape, pos)
        shifted_shape = cast(Shape, shifted_shape)

        self._update_grid(shifted_shape, padding=padding, no_bbox=no_bbox)

    def place_object(
        self, shape: Shape, color_palette: Optional[Set[int]] = None
    ) -> Shape:
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

        if color_palette is not None:
            color = random.choice(list(color_palette))
            shifted_shape = recolor(color, shifted_shape)
        else:
            color = random.choice(list(set(range(9)) - {self._bg_color}))
            shifted_shape = recolor(color, shifted_shape)

        self._update_grid(shifted_shape, padding=self._margin)

        return shifted_shape


class BinaryRelation(Enum):
    BelowOf = 0
    LeftOf = 1


class BSTGridBruteForce(GridBruteForce):
    """
    Grid that stores the nodes in binary tree. And keeps the relative position.
    """

    def __init__(
        self,
        h: int,
        w: int,
        bg_color: int = 0,
        margin: int = 1,
        mode: BinaryRelation = BinaryRelation.BelowOf,
    ) -> None:
        super().__init__(h, w, bg_color, margin)
        self._shapes_tree: Tree[Shape] = Tree(None)

        if mode == BinaryRelation.BelowOf:
            self._binary_relation = self._is_below
        elif mode == BinaryRelation.LeftOf:
            self._binary_relation = self._is_left_of
        else:
            raise ValueError("Not supported binary relation")

    @override
    def _update_grid(self, shape: Shape, padding: int = 0, no_bbox: bool = False):
        self._grid = paint(self._grid, shape)

        if no_bbox:
            self._occupied = self._occupied | toindices(shape)
        else:
            self._occupied = self._occupied | bounding_box(shape, padding=padding)

        # Insert the shape in the tree
        bst_insert(self._shapes_tree, Node(shape), self._binary_relation)

    @property
    def tree(self) -> Tree[Shape]:
        return self._shapes_tree

    @staticmethod
    def _is_below(a: Node[Shape], b: Node[Shape]) -> bool:
        """
        Encodes the binary relation: a < b := a 'is below' b.
        """
        y_a = ulcorner(a.key)[0]
        y_b = ulcorner(b.key)[0]

        if y_a > y_b:
            return True
        else:
            return False

    @staticmethod
    def _is_left_of(a: Node[Shape], b: Node[Shape]) -> bool:
        """
        Encodes the binary relation: a < b := a 'is left of' b.
        """
        x_a = ulcorner(a.key)[1]
        x_b = ulcorner(b.key)[1]

        if x_a < x_b:
            return True
        else:
            return False
