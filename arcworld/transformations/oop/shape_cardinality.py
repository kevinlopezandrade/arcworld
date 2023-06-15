from collections import Counter
from typing import Callable, Dict, List, cast

from arcworld.shape.oop.base import ShapeObject


def select_shape_with_most_cells(shapes):
    max_cell = 0
    biggest_shape = shapes[0]
    for s in shapes:
        if s.num_points > max_cell:
            max_cell = s.num_points
            biggest_shape = s
    return biggest_shape


def select_shape_with_least_cells(shapes):
    min_cell = 100
    smallest_shape = shapes[0]
    for s in shapes:
        if s.num_points < min_cell:
            min_cell = s.num_points
            smallest_shape = s
    return smallest_shape


def select_shape_with_median_cells(shapes):
    cell_number = {}
    for i, s in enumerate(shapes):
        cell_number[i] = s.num_points

    cell_number = {
        k: v for k, v in sorted(cell_number.items(), key=lambda item: item[1])
    }
    median_shape_index = int(len(shapes) / 2)
    median_shape = shapes[list(cell_number.keys())[median_shape_index]]
    return median_shape


def select_shape_with_most_rows(shapes):
    max_rows = 0
    biggest_shape = shapes[0]
    for s in shapes:
        if s.n_rows > max_rows:
            max_rows = s.n_rows
            biggest_shape = s
    return biggest_shape


def select_shape_with_least_rows(shapes):
    min_rows = 100
    smallest_shape = shapes[0]
    for s in shapes:
        if s.n_rows < min_rows:
            min_rows = s.n_rows
            smallest_shape = s
    return smallest_shape


def select_shape_with_median_rows(shapes):
    cell_number = {}
    for i, s in enumerate(shapes):
        cell_number[i] = s.n_rows

    cell_number = {
        k: v for k, v in sorted(cell_number.items(), key=lambda item: item[1])
    }
    median_shape_index = int(len(shapes) / 2)
    median_shape = shapes[list(cell_number.keys())[median_shape_index]]
    return median_shape


def select_shape_with_most_cols(shapes):
    max_cols = 0
    biggest_shape = shapes[0]
    for s in shapes:
        if s.n_cols > max_cols:
            max_cols = s.n_cols
            biggest_shape = s
    return biggest_shape


def select_shape_with_least_cols(shapes):
    min_cols = 100
    smallest_shape = shapes[0]
    for s in shapes:
        if s.n_cols < min_cols:
            min_cols = s.n_cols
            smallest_shape = s
    return smallest_shape


def select_shape_with_median_cols(shapes):
    cell_number = {}
    for i, s in enumerate(shapes):
        cell_number[i] = s.n_cols

    cell_number = {
        k: v for k, v in sorted(cell_number.items(), key=lambda item: item[1])
    }
    median_shape_index = int(len(shapes) / 2)
    median_shape = shapes[list(cell_number.keys())[median_shape_index]]
    return median_shape


def select_most_frequent_shape(shapes):
    list_of_stringed_arrays = []
    for s in shapes:
        list_of_stringed_arrays.append(str(s.as_shape_only_grid))
    dic_of_freq = Counter(list_of_stringed_arrays)
    sorted_dic_of_freq = sorted(dic_of_freq.items(), key=lambda x: x[1])
    stringed_max_shape = sorted_dic_of_freq[-1][0]  # Element -1 is max frequency index.
    for s in shapes:
        if str(s.as_shape_only_grid) == stringed_max_shape:
            return s


def select_least_frequent_shape(shapes):
    list_of_stringed_arrays = []
    for s in shapes:
        list_of_stringed_arrays.append(str(s.as_shape_only_grid))
    dic_of_freq = Counter(list_of_stringed_arrays)
    sorted_dic_of_freq = sorted(dic_of_freq.items(), key=lambda x: x[1])
    stringed_min_shape = sorted_dic_of_freq[0][0]  # Element 0 is min frequency index.
    for s in shapes:
        if str(s.as_shape_only_grid) == stringed_min_shape:
            return s


def select_median_frequent_shape(shapes):
    list_of_stringed_arrays = []
    for s in shapes:
        list_of_stringed_arrays.append(str(s.as_shape_only_grid))
    dic_of_freq = Counter(list_of_stringed_arrays)
    sorted_dic_of_freq = sorted(dic_of_freq.items(), key=lambda x: x[1])
    stringed_median_shape = sorted_dic_of_freq[1][0]  # Element 1 is median index.
    for s in shapes:
        if str(s.as_shape_only_grid) == stringed_median_shape:
            return s


transformations_dict = cast(
    Dict[str, Callable[[List[ShapeObject]], ShapeObject]],
    {
        "select_shape_with_most_cells": select_shape_with_most_cells,
        "select_shape_with_least_cells": select_shape_with_least_cells,
        "select_shape_with_median_cells": select_shape_with_median_cells,
        "select_shape_with_most_rows": select_shape_with_most_rows,
        "select_shape_with_least_rows": select_shape_with_least_rows,
        "select_shape_with_median_rows": select_shape_with_median_rows,
        "select_shape_with_most_cols": select_shape_with_most_cols,
        "select_shape_with_least_cols": select_shape_with_least_cols,
        "select_shape_with_median_cols": select_shape_with_median_cols,
        "select_most_frequent_colored_shape": select_most_frequent_shape,
        "select_least_frequent_colored_shape": select_least_frequent_shape,
        "select_median_frequent_colored_shape": select_median_frequent_shape,
    },
)
