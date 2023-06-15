from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray

from arcworld.shape.oop.base import ShapeObject


def select_most_frequent_color_in_grid(grid):
    grid_as_shape = ShapeObject(grid)
    grid_colors = grid_as_shape.colors
    return max(set(grid_colors), key=grid_colors.count)


def select_least_frequent_color_in_grid(grid):
    grid_as_shape = ShapeObject(grid)
    grid_colors = grid_as_shape.colors
    return min(set(grid_colors), key=grid_colors.count)


transformations_dict: Dict[str, Callable[[NDArray[np.float64]], int]] = {
    "select_most_frequent_color_in_grid": select_most_frequent_color_in_grid,
    "select_least_frequent_color_in_grid": select_least_frequent_color_in_grid,
}
