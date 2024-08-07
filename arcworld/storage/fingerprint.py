import hashlib
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from arcworld.internal.constants import Example, Task


def _all_permutations(pre: Task, post: Task, res: List[Task]):
    if len(pre) == 0:
        res.append(post)
    else:
        for i in range(len(pre)):
            _all_permutations(pre[:i] + pre[i + 1 :], post + [pre[i]], res)


def _flatten(example: Example):
    return np.hstack((example.input.flatten(), example.output.flatten()))


def hash_task(task: Task) -> str:
    """
    Given a Task. Uses SHA2-256 algorithm to create a 'unique' fingerprint.
    """
    if len(task) == 0:
        raise ValueError("Task with no examples.")

    res = _flatten(task[0])
    for i in range(1, len(task)):
        res = np.hstack((res, _flatten(task[i])))

    hash = hashlib.sha256(res.tobytes()).hexdigest()

    return hash


def normalize_task(
    task: Task, h: int = 30, w: int = 30, max_input_output_pairs: int = 4
) -> NDArray[np.uint8]:
    """
    Given a task normalize the task examples so that all of them are placed
    at the top left corner of a hxw grid. We use 10 as value to flag a pixel
    not being part of the task. Dimensions: [N_Example, 0 | 1, h, w], where
    0 := input, 1 := output. e.g To get output of the example 2, X[1, 1, :, :]

    Args:
        task: Task to be normalized
        h: Height of the normalized grid
        w: Width of the normalized grid
        max_input_otput_pairs: Maximum input output pairs of the normalized task.
    """
    MAX_PAIRS = max_input_output_pairs  # noqa

    N = len(task)  # noqa

    if N > MAX_PAIRS:
        raise ValueError(f"A task cannot have more than {MAX_PAIRS} examples")

    res = np.empty(shape=(MAX_PAIRS, 2, h, w), dtype=np.uint8)

    # TODO: We use the 10 value as a flag. Maybe I need
    # to test if the examples contain values that are not
    # within [0, 9].
    res.fill(10)

    for i in range(N):
        inp_shape = task[i].input.shape
        out_shape = task[i].output.shape

        if inp_shape[0] > h or inp_shape[1] > w:
            raise ValueError(f"Input {i} exceeds the dimension of the normalized grid.")

        if out_shape[0] > h or out_shape[1] > w:
            raise ValueError(
                f"Output {i} exceeds the dimension of the normalized grid."
            )

        res[i, 0, : inp_shape[0], : inp_shape[1]] = task[i].input
        res[i, 1, : out_shape[0], : out_shape[1]] = task[i].output

    return res


def _find_bounds(array: NDArray[np.uint8]) -> Tuple[int, int]:
    """
    Returns indices of the first 10 value found in the first and second
    dimension.
    """
    h_bound = 0
    for i in range(array.shape[0]):
        if array[i, 0] < 10:
            h_bound += 1
        else:
            break

    w_bound = 0
    for i in range(array.shape[1]):
        if array[0, i] < 10:
            w_bound += 1

    return (h_bound, w_bound)


def from_array_to_task(grid: NDArray[np.uint8]) -> Task:
    task: Task = []
    for pair in grid:
        example = Example(input=pair[0], output=pair[1])
        task.append(example)

    return task


def decode_normalized_grid(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Given a normalized grid, returns the subgrid where it does not contain
    padding pixels.
    """
    h, w = _find_bounds(grid)

    return grid[:h, :w]


def decode_normalized_task_sqlite(
    task_normalized: NDArray[np.uint8], max_input_otput_pairs: int = 4
) -> Task:
    """
    Given a numpy array, where each example is normalized to fit into a 2x30x30 grid,
    returns a Task.
    """
    MAX_PAIRS = max_input_otput_pairs  # noqa
    tasks: Task = []

    try:
        task_normalized.shape
    except AttributeError:
        task_normalized = np.frombuffer(task_normalized, dtype=np.uint8).reshape(
            MAX_PAIRS, 2, 30, 30
        )

    for i in range(task_normalized.shape[0]):
        input, output = task_normalized[i]
        h_input, w_input = _find_bounds(input)

        if h_input == 0 and w_input == 0:
            # (0, 0) bounds mean that there is no example.
            continue

        h_output, w_output = _find_bounds(output)
        example = Example(
            input=input[:h_input, :w_input], output=output[:h_output, :w_output]
        )

        tasks.append(example)

    return tasks
