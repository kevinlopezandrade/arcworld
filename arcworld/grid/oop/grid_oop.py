import random
from typing import List, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Shape
from arcworld.internal.constants import DoesNotFitError
from arcworld.shape.oop.base import ShapeObject
from arcworld.shape.oop.utils import grid_to_cropped_grid, grid_to_pc, shift_indexes


def to_shape_object(shape: Shape) -> ShapeObject:
    # shape = cast(Shape, normalize(shape))  # Cast because of union type.
    point_cloud = {}

    for color, (x, y) in shape:
        point_cloud[x, y] = color

    return ShapeObject(point_cloud)


def make_uniform_color(shape: ShapeObject, color: int) -> ShapeObject:
    new_pc = {}
    for x, y in shape.pc:
        new_pc[x, y] = color

    return ShapeObject(new_pc)


class GridObject:
    def __init__(self, h: int, w: int) -> None:
        self._h = h
        self._w = w
        self._grid_shape = (h, w)
        # TODO: Check if I can deifine this as a np.uint8
        self._grid = np.zeros(shape=self._grid_shape)

        self._shapes: List[ShapeObject] = []

    @property
    def height(self) -> int:
        return self._h

    @property
    def width(self) -> int:
        return self._w

    @property
    def shape(self) -> Tuple[int, int]:
        return self._grid_shape

    @property
    def grid(self) -> NDArray[np.float64]:
        return self._grid

    @property
    def shapes(self) -> List[ShapeObject]:
        return self._shapes

    def place_object(
        self,
        shape: ShapeObject,
        background: int = 0,
        allow_touching_objects: bool = False,
    ) -> ShapeObject:
        """Randomly chooses position for the shape in the grid"""
        shape = ShapeObject(shape)
        zeroedworld = self.grid.copy()
        zeroedworld[self.grid == background] = 0

        if not allow_touching_objects:
            dilated_shape = scipy.ndimage.morphology.binary_dilation(
                shape.grid, structure=scipy.ndimage.generate_binary_structure(2, 2)
            ).astype(int)
            positions = self._find_possible_positions(zeroedworld, dilated_shape)
        else:
            positions = self._find_possible_positions(zeroedworld, shape.grid)

        if len(positions) == 0:
            raise DoesNotFitError("Shape does not fit")

        position = random.choice(positions)
        shape.move_to_position(position)
        shape_grid_at_world_size = shape.grid[
            : self.grid.shape[0], : self.grid.shape[1]
        ]

        # Update the grid
        self.grid[shape_grid_at_world_size > 0] = shape_grid_at_world_size[
            shape_grid_at_world_size > 0
        ]

        # Update the objects list.
        self._shapes.append(shape)

        return shape

    @staticmethod
    def _find_possible_positions(
        world: NDArray[np.float64], grid: NDArray[np.float64], allow_holes: bool = True
    ) -> List[Tuple[int, int]]:
        world = world.copy()
        grid = grid_to_cropped_grid(grid)
        world[world != 0] = 1
        grid[grid != 0] = 1

        if not allow_holes:
            grid = scipy.ndimage.binary_fill_holes(grid).astype(int)
            world = scipy.ndimage.binary_fill_holes(world).astype(int)
        if world.shape[0] < grid.shape[0] or world.shape[1] < world.shape[1]:
            return []

        res = (
            scipy.signal.correlate2d(world, grid, mode="same", fillvalue=1) == 0
        ).astype(int)
        # values that are 0 are possible positions, but they use the middle as position
        # and not the top left corner, so shift to get top left corners
        dx = (grid.shape[0] - 1) // 2 * -1
        dy = (grid.shape[1] - 1) // 2 * -1
        indexes = shift_indexes(grid_to_pc(res).indexes, dx, dy)

        return indexes
