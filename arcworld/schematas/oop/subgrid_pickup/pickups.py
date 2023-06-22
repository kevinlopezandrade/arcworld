from typing import Dict, List, Type

import numpy as np
from numpy.typing import NDArray

from arcworld.grid.oop.grid_oop import GridObject
from arcworld.schematas.oop.subgrid_pickup.base_pickup import SubgridPickup
from arcworld.transformations.oop.color_cardinality import (
    transformations_dict as color_cardinality_transformation_dict,
)
from arcworld.transformations.oop.grid_picking import (
    transformations_dict as grid_picking_transformation_dict,
)
from arcworld.transformations.oop.shape_cardinality import (
    transformations_dict as shape_cardinality_transformation_dict,
)


class TopLeftQuadrant(SubgridPickup):
    CONSTRAINTS = {
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
        "grid_size_even": True,
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_top_left_quadrant"](grid.grid)


class TopRightQuadrant(SubgridPickup):
    CONSTRAINTS = {
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
        "grid_size_even": True,
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_top_right_quadrant"](grid.grid)


class BottomLeftQuadrant(SubgridPickup):
    CONSTRAINTS = {
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
        "grid_size_even": True,
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_bottom_left_quadrant"](
            grid.grid
        )


class BottomRightQuadrant(SubgridPickup):
    CONSTRAINTS = {
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
        "grid_size_even": True,
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_bottom_right_quadrant"](
            grid.grid
        )


class UpperHalf(SubgridPickup):
    CONSTRAINTS = {"grid_rows_range": (2, 30), "grid_size_even": True}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_upper_half"](grid.grid)


class LowerHalf(SubgridPickup):
    CONSTRAINTS = {"grid_rows_range": (2, 30), "grid_size_even": True}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_lower_half"](grid.grid)


class LeftHalf(SubgridPickup):
    CONSTRAINTS = {"grid_cols_range": (2, 30), "grid_size_even": True}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_left_half"](grid.grid)


class RightHalt(SubgridPickup):
    CONSTRAINTS = {"grid_cols_range": (2, 30), "grid_size_even": True}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid_picking_transformation_dict["select_right_half"](grid.grid)


class CropToShape(SubgridPickup):
    CONSTRAINTS = {"num_shapes_range": (1, 1)}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid.shapes[0].as_shape_only_grid


# COLOR CARDINALITY


class MostFrequentColor(SubgridPickup):
    CONSTRAINTS = {"shape_parameter_to_distinguish": "grid_color"}  # Not used.

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return np.array(
            [
                [
                    color_cardinality_transformation_dict[
                        "select_most_frequent_color_in_grid"
                    ](grid.grid)
                ]
            ]
        )


class LeastFrequentColor(SubgridPickup):
    CONSTRAINTS = {"shape_parameter_to_distinguish": "grid_color"}  # Not used

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return np.array(
            [
                [
                    color_cardinality_transformation_dict[
                        "select_least_frequent_color_in_grid"
                    ](grid.grid)
                ]
            ]
        )


# SHAPE CARDINALITY


class ShapeMostCells(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_cells",
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_most_cells"](
            grid.shapes
        ).as_shape_only_grid


class ShapeLeastCells(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_cells",
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_least_cells"](
            grid.shapes
        ).as_shape_only_grid


class ShapeMedianCells(SubgridPickup):
    CONSTRAINTS = {
        "median": True,
        "num_shapes_range": (3, 7),
        "grid_rows_range": (2, 30),
        "grid_cols_range": (2, 30),
        "num_shapes_even": False,
        "shape_parameter_to_distinguish": "n_cells",
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_median_cells"](
            grid.shapes
        ).as_shape_only_grid


class ShapeMostRows(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_rows",
        "grid_rows_range": (3, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_most_rows"](
            grid.shapes
        ).as_shape_only_grid


class ShapeLeastRows(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_rows",
        "grid_rows_range": (3, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_least_rows"](
            grid.shapes
        ).as_shape_only_grid


class ShapeMedianRows(SubgridPickup):
    CONSTRAINTS = {
        "median": True,
        "num_shapes_range": (3, 7),
        "grid_rows_range": (3, 30),
        "num_shapes_even": False,
        "shape_parameter_to_distinguish": "n_rows",
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_median_rows"](
            grid.shapes
        ).as_shape_only_grid


class ShapeMostCols(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_cols",
        "grid_cols_range": (3, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_most_cols"](
            grid.shapes
        ).as_shape_only_grid


class ShapeLeastCols(SubgridPickup):
    CONSTRAINTS = {
        "num_shapes_range": (2, 7),
        "shape_parameter_to_distinguish": "n_cols",
        "grid_cols_range": (3, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_least_cols"](
            grid.shapes
        ).as_shape_only_grid


class ShapeMedianCols(SubgridPickup):
    CONSTRAINTS = {
        "median": True,
        "num_shapes_range": (3, 7),
        "num_shapes_even": False,
        "shape_parameter_to_distinguish": "n_cols",
        "grid_cols_range": (3, 30),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict["select_shape_with_median_cols"](
            grid.shapes
        ).as_shape_only_grid


class MostFrequentShape(SubgridPickup):
    CONSTRAINTS = {"some_shapes_repeated_in_grid": True, "num_shapes_range": (3, 7)}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict[
            "select_most_frequent_colored_shape"
        ](grid.shapes).as_shape_only_grid


class LeastFrequentShape(SubgridPickup):
    CONSTRAINTS = {"some_shapes_repeated_in_grid": True, "num_shapes_range": (3, 7)}

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict[
            "select_least_frequent_colored_shape"
        ](grid.shapes).as_shape_only_grid


class MedianFrequentShape(SubgridPickup):
    CONSTRAINTS = {
        "median": True,
        "some_shapes_repeated_in_grid": True,
        "num_shapes_range": (3, 7),
    }

    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return shape_cardinality_transformation_dict[
            "select_median_frequent_colored_shape"
        ](grid.shapes).as_shape_only_grid


class WholeGrid(SubgridPickup):
    def __call__(self, grid: GridObject) -> NDArray[np.float64]:
        return grid.grid


SUBGRID_PICKUP_DICT: Dict[str, Type[SubgridPickup]] = {
    "select_shape_with_most_cells": ShapeMostCells,
    "select_shape_with_least_cells": ShapeLeastCells,
    "select_shape_with_median_cells": ShapeMedianCells,
    "select_shape_with_most_rows": ShapeMostRows,
    "select_shape_with_least_rows": ShapeLeastRows,
    "select_shape_with_median_rows": ShapeMedianRows,
    "select_shape_with_most_cols": ShapeMostCols,
    "select_shape_with_least_cols": ShapeLeastCols,
    "select_shape_with_median_cols": ShapeMedianCols,
    "select_most_frequent_shape": MostFrequentShape,
    "select_least_frequent_shape": LeastFrequentShape,
    "select_median_frequent_shape": MedianFrequentShape,
    "select_top_left_quadrant": TopLeftQuadrant,
    "select_top_right_quadrant": TopRightQuadrant,
    "select_bottom_left_quadrant": BottomLeftQuadrant,
    "select_bottom_right_quadrant": BottomRightQuadrant,
    "select_upper_half": UpperHalf,
    "select_lower_half": LowerHalf,
    "select_left_half": LeftHalf,
    "select_right_half": RightHalt,
    "select_most_frequent_color_in_grid": MostFrequentColor,
    "select_least_frequent_color_in_grid": LeastFrequentColor,
    "crop_to_shape": CropToShape,
    "select_whole_grid": WholeGrid,
}

DEFAULT_PICKUPS: List[Type[SubgridPickup]] = list(SUBGRID_PICKUP_DICT.values())
