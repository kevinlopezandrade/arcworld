from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray


def select_top_left_quadrant(grid):
    quadrant_grid_row_limit = int(grid.shape[0] / 2)
    quadrant_grid_col_limit = int(grid.shape[1] / 2)
    return grid[:quadrant_grid_row_limit, :quadrant_grid_col_limit]


def select_top_right_quadrant(grid):
    quadrant_grid_row_limit = int(grid.shape[0] / 2)
    quadrant_grid_col_limit = int(grid.shape[1] / 2)
    return grid[:quadrant_grid_row_limit, quadrant_grid_col_limit:]


def select_bottom_left_quadrant(grid):
    quadrant_grid_row_limit = int(grid.shape[0] / 2)
    quadrant_grid_col_limit = int(grid.shape[1] / 2)
    return grid[quadrant_grid_row_limit:, :quadrant_grid_col_limit]


def select_bottom_right_quadrant(grid):
    quadrant_grid_row_limit = int(grid.shape[0] / 2)
    quadrant_grid_col_limit = int(grid.shape[1] / 2)
    return grid[quadrant_grid_row_limit:, quadrant_grid_col_limit:]


def select_upper_half(grid):
    grid_row_limit = int(grid.shape[0] / 2)
    return grid[:grid_row_limit, :]


def select_lower_half(grid):
    grid_row_limit = int(grid.shape[0] / 2)
    return grid[grid_row_limit:, :]


def select_left_half(grid):
    grid_col_limit = int(grid.shape[1] / 2)
    return grid[:, :grid_col_limit]


def select_right_half(grid):
    grid_col_limit = int(grid.shape[1] / 2)
    return grid[:, grid_col_limit:]


transformations_dict: Dict[
    str, Callable[[NDArray[np.float64]], NDArray[np.float64]]
] = {
    "select_top_left_quadrant": select_top_left_quadrant,
    "select_top_right_quadrant": select_top_right_quadrant,
    "select_bottom_left_quadrant": select_bottom_left_quadrant,
    "select_bottom_right_quadrant": select_bottom_right_quadrant,
    "select_upper_half": select_upper_half,
    "select_lower_half": select_lower_half,
    "select_left_half": select_left_half,
    "select_right_half": select_right_half,
}
