import logging
import os
import random
import string
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from arcworld.internal.constants import Example, Task
from arcworld.storage.fingerprint import normalize_task
from arcworld.utils import decode_json_task

logger = logging.getLogger(__name__)

ARC_TENSOR = Tuple[Tensor, Tensor, Tensor]


def _read_arc_json_files(path: str) -> List[Task]:
    """
    Given a directory path returns a list of arc
    Tasks.
    """
    files = sorted(os.listdir(path))

    tasks: List[Task] = []
    for file in tqdm(files, desc="Decoding json files", leave=False):
        try:
            task = decode_json_task(os.path.join(path, file))
        except Exception as e:
            print(e)
        else:
            tasks.append(task)

    logger.info(f"Decoded tasks: {len(tasks)}")

    return tasks


def encode_colors(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
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
    for color in range(0, 11):
        x, y = np.where(grid == color)
        new_grid[color, x, y] = 1

    return new_grid


def decode_colors(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Given a grid with shape [C, H, W] where each channel
    encodes binary a color, returns the grid with shape
    [H, W] where colors are not binarized anymore.
    """
    _, h, w = grid.shape
    decoded_grid = np.zeros(shape=(h, w), dtype=np.uint8)

    for color in range(0, 11):
        decoded_grid = decoded_grid + (color * grid[color, :, :])

    return decoded_grid


def encode_task(normalized_task: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    In Stefan architecture the inputs are passed as [N_Examples * 2, C, H, W]
    where the input and output pairs are arranged contiguously. Colors of
    original grid are encoded as binary.

    Args:
        normalized_task: Is a multidimensional array, where the shape is as follows:
        [N_Example, 0 | 1, 30, 30], where 0 := input example, 1 := output example.
        The value 10 marks a coordinate not belonging to the grid

    Returns:
        Array following Stefan shape, [N_Examples * 2, C, H, W].
    """
    n, _, h, w = normalized_task.shape

    X = np.zeros((n * 2, 11, h, w), dtype=np.uint8)  # noqa
    for i, example in enumerate(normalized_task):
        X[i * 2, :, :, :] = encode_colors(example[0, :, :])
        X[i * 2 + 1, :, :, :] = encode_colors(example[1, :, :])

    return X


class ARCDataset(Dataset[ARC_TENSOR]):
    def __init__(
        self,
        path: str,
        h_bound: int = 30,
        w_bound: int = 30,
        max_input_output_pairs: int = 4,
    ):
        """
        Args:
            path: Path of the directory containing the json files
            h_bound: Height to which normalize the grids.
            w_bound: Width to which normalize the grids.
            max_input_otput_pairs: Maximum number of input output pairs used for the
                normalization of a task.
        """
        self.path = path
        self.data = _read_arc_json_files(path)
        self.h_bound = h_bound
        self.w_bound = w_bound
        self.max_input_output_pairs = max_input_output_pairs

        self._id = os.path.normpath(self.path)

    @property
    def id(self) -> str:
        return self._id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        A normalized task is a multidimensional array, where the shape is as follows:
        [N_Example, 0 | 1, 30, 30], where 0 := input example, 1 := output example.
        The value 10 marks a coordinate not belonging to the grid.

        Returns:
            X: Tensor with shape [N_input_output_pairs * 2, C, H, W]
            inp_test: Tensor with shape [C, H, W]
            out_test: Tensor with shape [C, H, W]
        """
        N = len(self.data[idx])  # noqa
        data = self.data[idx]

        if N > self.max_input_output_pairs:
            # The last element is always the test sample.
            data = random.sample(data[:-1], k=self.max_input_output_pairs - 1) + [
                data[-1]
            ]

        task = normalize_task(
            data,
            h=self.h_bound,
            w=self.w_bound,
            max_input_output_pairs=len(data),
        )

        X = torch.Tensor(encode_task(task[:-1]))  # noqa
        inp_test = torch.Tensor(encode_colors(task[-1, 0, :, :]))

        # Output test, should not be encoded for later computing the
        # cross entropy loss.
        out_test = torch.LongTensor(task[-1, 1, :, :])

        return X, inp_test, out_test


class ARCDatasetTest(Dataset[ARC_TENSOR]):
    """
    This class is made purely for testing.
    It should live in the 'tests' directory we have in the root project.
    But we leave it here for convenience.
    """

    def __init__(
        self,
        path: str,
        h_bound: int = 30,
        w_bound: int = 30,
        max_input_otput_pairs: int = 4,
        len: int = 512,
    ) -> None:
        self.h_bound = h_bound
        self.w_bound = w_bound
        self.max_input_otput_pairs = max_input_otput_pairs
        self.len = len
        self._id = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

        self.data = self._create_fake_data(self.len, self.max_input_otput_pairs)

        logger.info(f"Using debug dataset {self._id}")

    @staticmethod
    def _create_fake_data(len: int, max_input_otput_pairs: int) -> List[Task]:
        tasks: List[Task] = []
        for _ in trange(len, desc="Creating tasks"):
            task: Task = []
            for _ in range(max_input_otput_pairs):
                inp = np.zeros(shape=(30, 30), dtype=np.uint8)
                out = np.zeros(shape=(30, 30), dtype=np.uint8)
                task.append(Example(input=inp, output=out))

            tasks.append(task)

        return tasks

    @property
    def id(self) -> str:
        return self._id

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> ARC_TENSOR:
        data = self.data[index]

        task = normalize_task(
            data,
            h=self.h_bound,
            w=self.w_bound,
            max_input_output_pairs=len(data),
        )

        X = torch.Tensor(encode_task(task[:-1]))  # noqa
        inp_test = torch.Tensor(encode_colors(task[-1, 0, :, :]))

        # Output test, should not be encoded for later computing the
        # cross entropy loss.
        out_test = torch.LongTensor(task[-1, 1, :, :])

        return X, inp_test, out_test
