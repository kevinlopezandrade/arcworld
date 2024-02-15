import json
import os
from typing import List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Coordinates, Object
from arcworld.dsl.functional import canvas, height, normalize, paint, recolor, width
from arcworld.internal.constants import COLORMAP, NORM, TASK_DICT, Example, Task


def decode_json_task(file_path: str) -> Task:
    with open(file_path) as f:
        data = json.load(f)

    examples = data["train"] + data["test"]

    task: Task = []
    for example in examples:
        input = example["input"]
        output = example["output"]
        example = Example(
            input=np.array(input, dtype=np.uint8),
            output=np.array(output, dtype=np.uint8),
        )
        task.append(example)

    return task


def plot_object(shape: Object):
    shape = cast(Object, normalize(shape))
    h = height(shape)
    w = width(shape)
    grid = canvas(0, (h, w))
    grid = paint(grid, shape)

    fig, axe = plt.subplots()
    axe.imshow(grid, cmap=COLORMAP, norm=NORM)
    axe.grid(True, which="both", color="lightgrey", linewidth=0.5)
    # axe.set_xticks([x - 0.5 for x in range(width(shape))])
    # axe.set_yticks([x - 0.5 for x in range(height(shape))])
    axe.set_xticks([x - 0.5 for x in range(w)])
    axe.set_yticks([x - 0.5 for x in range(h)])
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    plt.show()


def plot_objects(*shapes):
    h = max(height(shape) for shape in shapes)
    w = max(height(shape) for shape in shapes)

    fig, axes = plt.subplots(1, len(shapes))

    for i, shape in enumerate(shapes):
        grid = canvas(0, (h, w))
        grid = paint(grid, shape)
        axes[i].imshow(grid, cmap=COLORMAP, norm=NORM)
        axes[i].grid(True, which="both", color="lightgrey", linewidth=0.5)
        axes[i].set_xticks([x - 0.5 for x in range(w)])
        axes[i].set_yticks([x - 0.5 for x in range(h)])
        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])

    plt.show()


def plot_proto_shape(proto_shape: Coordinates):
    shape = recolor(7, proto_shape)
    plot_object(shape)


def plot_grid(grid: NDArray[np.uint8], return_fig: bool = False):
    fig, axe = plt.subplots()

    axe.imshow(grid, cmap=COLORMAP, norm=NORM)
    axe.grid(True, which="both", color="lightgrey", linewidth=0.5)

    axe.set_xticks([x - 0.5 for x in range(grid.shape[1])])
    axe.set_yticks([x - 0.5 for x in range(grid.shape[0])])
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()


def plot_grids(
    *grids,
    titles: Optional[List[str]] = None,
    sup_title: Optional[str] = None,
    return_fig: bool = False,
):
    fig, axes = plt.subplots(1, len(grids))

    for i, grid in enumerate(grids):
        h = grid.shape[0]
        w = grid.shape[1]
        axes[i].imshow(grid, cmap=COLORMAP, norm=NORM)
        axes[i].grid(True, which="both", color="lightgrey", linewidth=0.5)
        axes[i].set_xticks([x - 0.5 for x in range(w)])
        axes[i].set_yticks([x - 0.5 for x in range(h)])
        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])

        if titles:
            if i < len(titles):
                axes[i].set_title(titles[i])

    if sup_title:
        fig.suptitle(sup_title)

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()


def plot_json_task(file_path: str):
    """
    Plots a task in the json format given by the original ARC dataset.
    Args:
        file: Path to the json file containing the task.
    """

    with open(file_path) as f:
        data = json.load(f)

    n_train = len(data["train"])
    samples = data["train"] + data["test"]

    fig, axes = plt.subplots(2, len(samples))

    for i, subtask in enumerate(samples):
        for j, grid in enumerate([subtask["input"], subtask["output"]]):
            h = len(grid)
            w = len(grid[0])

            title = ""

            if i < n_train:
                if j == 0:
                    title += f"input {i}"
                else:
                    title += f"output {i}"
            else:
                if j == 0:
                    title += "test input"
                else:
                    title += "test output"

            axes[j, i].imshow(grid, cmap=COLORMAP, norm=NORM)
            axes[j, i].grid(True, which="both", color="lightgrey", linewidth=0.5)
            axes[j, i].set_xticks([x - 0.5 for x in range(w)])
            axes[j, i].set_yticks([x - 0.5 for x in range(h)])
            axes[j, i].set_yticklabels([])
            axes[j, i].set_xticklabels([])

            axes[j, i].set_title(title)

    fig.suptitle(f"{os.path.basename(file_path)}")

    plt.show()


def plot_task(task: Task, title: Optional[str] = None, return_fig: bool = False):
    """
    Plots a task in the json format given by the original ARC dataset,
    where only train tasks appears.
    Args:
        file: Path to the json file containing the task.
    """
    N = len(task)  # noqa

    fig, axes = plt.subplots(2, N, squeeze=False)

    for i, example in enumerate(task):
        for j, grid in enumerate([example.input, example.output]):
            h = grid.shape[0]
            w = grid.shape[1]

            label = ""

            if j == 0:
                label += f"input {i}"
            else:
                label += f"output {i}"

            axes[j, i].imshow(grid, cmap=COLORMAP, norm=NORM)
            axes[j, i].grid(True, which="both", color="lightgrey", linewidth=0.5)
            axes[j, i].set_xticks([x - 0.5 for x in range(w)])
            axes[j, i].set_yticks([x - 0.5 for x in range(h)])
            axes[j, i].set_yticklabels([])
            axes[j, i].set_xticklabels([])

            axes[j, i].set_title(label)

    if title:
        fig.suptitle(title)

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()


def task_to_json(task: Task) -> TASK_DICT:
    train_examples = task[:-1]
    test_example = task[-1]

    task_dict: TASK_DICT = {"train": [], "test": []}

    for example in train_examples:
        task_dict["train"].append(
            {"input": example.input.tolist(), "output": example.output.tolist()}
        )

    task_dict["test"].append(
        {"input": test_example.input.tolist(), "output": test_example.output.tolist()}
    )

    return task_dict
