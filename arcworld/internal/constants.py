from typing import Dict, List

import numpy as np
from matplotlib import colors
from numpy.typing import NDArray

MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 1
ALLOWED_COLORS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
PADDING = -1

COLORMAP = colors.ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
NORM = colors.Normalize(vmin=0, vmax=9)


class DoesNotFitError(Exception):
    pass


class ShapeOutOfBoundsError(Exception):
    pass


class GridConstructionError(RuntimeError):
    pass


# Example: {"train": [{"input": np.array, "output": np.array}]}
TASK_DICT = Dict[str, List[Dict[str, NDArray[np.uint8]]]]
