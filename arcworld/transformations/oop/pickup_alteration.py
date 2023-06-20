import random
from typing import Callable, Dict, cast

import numpy as np
from numpy.typing import NDArray

from arcworld.shape.oop.base import ShapeObject
from arcworld.transformations.oop.single_shape_transformations import (
    transformations_dict as single_shape_transformations,
)


def show_as_is(grid, fixed_task_seed):
    return grid


def duplicate_grid_2_times_vertically(grid, fixed_task_seed):
    grid_row = grid.shape[0] * 2
    grid_col = grid.shape[1]
    output = np.zeros((grid_row, grid_col))
    output[: grid.shape[0], :] = grid
    output[grid.shape[0] :, :] = grid
    return output


def duplicate_grid_2_times_horizontally(grid, fixed_task_seed):
    grid_row = grid.shape[0]
    grid_col = grid.shape[1] * 2
    output = np.zeros((grid_row, grid_col))
    output[:, grid.shape[1] :] = grid
    output[:, : grid.shape[1]] = grid
    return output


def duplicate_grid_4_times_vertically(grid, fixed_task_seed):
    grid_row = grid.shape[0] * 4
    grid_col = grid.shape[1]
    output = np.zeros((grid_row, grid_col))
    output[: int(output.shape[0] / 4), :] = grid
    output[int(output.shape[0] / 4) : int(output.shape[0] / 2), :] = grid
    output[int(output.shape[0] / 2) : int(output.shape[0] * 3 / 4), :] = grid
    output[int(output.shape[0] * 3 / 4) :, :] = grid
    return output


def duplicate_grid_4_times_horizontally(grid, fixed_task_seed):
    grid_row = grid.shape[0]
    grid_col = grid.shape[1] * 4
    output = np.zeros((grid_row, grid_col))
    output[:, : int(output.shape[1] / 4)] = grid
    output[:, int(output.shape[1] / 4) : int(output.shape[1] / 2)] = grid
    output[:, int(output.shape[1] / 2) : int(output.shape[1] * 3 / 4)] = grid
    output[:, int(output.shape[1] * 3 / 4) :] = grid
    return output


def circle_grid_with_color(grid, fixed_task_seed):
    return


def crop_shape(grid, fixed_task_seed):
    return


def recolor_grid(grid, fixed_task_seed):
    random.seed(fixed_task_seed)
    color = random.randint(1, 9)
    grid[grid != 0] = color
    return grid


def rot90_grid(grid, fixed_task_seed):
    return single_shape_transformations["rot90"](ShapeObject(grid)).as_shape_only_grid


def mirror_vertical_grid(grid, fixed_task_seed):
    return single_shape_transformations["mirror_vertical"](
        ShapeObject(grid)
    ).as_shape_only_grid


def mirror_horizontal_grid(grid, fixed_task_seed):
    return single_shape_transformations["mirror_horizontal"](
        ShapeObject(grid)
    ).as_shape_only_grid


def duplicate_grid_4_times_2by2(grid, fixed_task_seed):
    grid_row = grid.shape[0] * 2
    grid_col = grid.shape[1] * 2
    output = np.zeros((grid_row, grid_col))
    output[: grid.shape[0], : grid.shape[1]] = grid
    output[grid.shape[0] :, : grid.shape[1]] = grid
    output[: grid.shape[0], grid.shape[1] :] = grid
    output[grid.shape[0] :, grid.shape[1] :] = grid
    return output


PICKUP_ALTERATIONS = cast(
    Dict[str, Callable[[NDArray[np.float64], int], NDArray[np.float64]]],
    {
        "show_as_is": show_as_is,
        "duplicate_grid_4_times_2by2": duplicate_grid_4_times_2by2,
        "duplicate_grid_2_times_vertically": duplicate_grid_2_times_vertically,
        "duplicate_grid_2_times_horizontally": duplicate_grid_2_times_horizontally,
        "duplicate_grid_4_times_vertically": duplicate_grid_4_times_vertically,
        "duplicate_grid_4_times_horizontally": duplicate_grid_4_times_horizontally,
        "rot90_grid": rot90_grid,
        "mirror_vertical_grid": mirror_vertical_grid,
        "mirror_horizontal_grid": mirror_horizontal_grid,
        "recolor_grid": recolor_grid,
    },
)
