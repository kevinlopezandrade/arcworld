import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm import tqdm

from arcworld.internal.constants import Task
from arcworld.storage.fingerprint import decode_normalized_grid, normalize_task
from arcworld.utils import decode_json_task


def _read_arc_json_files(path: str) -> List[Task]:
    """
    Given a directory path returns a list of arc
    Tasks.
    """
    files = sorted(os.listdir(path))

    tasks: List[Task] = []
    for file in tqdm(files):
        try:
            task = decode_json_task(os.path.join(path, file))
        except Exception as e:
            print(e)
        else:
            tasks.append(task)

    print(f"Decoded tasks: {len(tasks)}")

    return tasks


def _encode_colors(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Given a grid encodes the colors following a binary
    encoding where each channel represents a color.

    Args:
        grid: Normalized numpy array representing a grid.

    Returns:
        A new array X with shape [10, H, W] where
        X[c, i, j] = 0 if the original grid does not have
        color 'c' at position (i, j) and X[c, i,  j] = 1
        if it has that color. Where c is in 0<=c<=9.
    """
    new_grid = np.zeros((11, *grid.shape), dtype=np.uint8)
    for color in range(0, 10):
        x, y = np.where(grid == color)
        new_grid[color, x, y] = 1

    # Padding channel.
    x, y = np.where(grid == 255)
    new_grid[10, x, y] = 1

    return new_grid


def _decode_colors(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    _, h, w = grid.shape
    decoded_grid = np.zeros(shape=(h, w), dtype=np.uint8)

    for i, color in enumerate(list(range(0, 10)) + [255]):
        decoded_grid = decoded_grid + (color * grid[i, :, :])

    decoded_grid = decode_normalized_grid(decoded_grid)

    return decoded_grid


class TransformerOriginalDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = _read_arc_json_files(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        A normalized task is a multidimensional array, where the shape is as follows:
        [N_Example, 0 | 1, 30, 30], where 0 := input example, 1 := output example.
        The value 255 marks a coordinate not belonging to the grid.

        Returns:
            X: Tensor with shape [N_input_output_pairs * 2, C, H, W]
            inp_test: Tensor with shape [C, H, W]
            out_test: Tensor with shape [C, H, W]
        """
        task = normalize_task(self.data[idx])

        # In Stefan architecture the inputs are passed
        # as [6, C, H, W] where the input and output pairs
        # are arranged contiguously.
        X = np.zeros((3 * 2, 11, 30, 30), dtype=np.uint8)  # noqa
        for i, example in enumerate(task[:-1]):
            X[i * 2, :, :, :] = _encode_colors(example[0, :, :])
            X[i * 2 + 1, :, :, :] = _encode_colors(example[1, :, :])

        X = torch.Tensor(X)  # noqa
        inp_test = torch.Tensor(_encode_colors(task[-1, 0, :, :]))
        out_test = torch.Tensor(_encode_colors(task[-1, 1, :, :]))

        return X, inp_test, out_test
