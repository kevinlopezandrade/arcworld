import numpy as np

from arcworld.internal.constants import ShapeOutOfBoundsError


def pc_to_full_sized_grid(pc, n_cols=30, n_rows=30) -> np.ndarray:
    grid = np.zeros((n_cols, n_rows), dtype=int)
    try:
        for idx, color in pc.items():
            grid[idx] = color
    except IndexError:
        raise ShapeOutOfBoundsError(
            f"Can not convert this pc into a grid of size ({n_cols}, {n_rows})"
        )
    return grid


def pc_to_shape_only_grid(pc) -> np.ndarray:
    grid = np.zeros((pc.n_rows, pc.n_cols), dtype=int)
    dx = pc.min_x
    dy = pc.min_y
    for idx, color in pc.items():
        x, y = idx
        grid[(x - dx, y - dy)] = color
    return grid
