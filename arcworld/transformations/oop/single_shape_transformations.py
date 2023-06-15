import numpy as np
import scipy

from arcworld.internal.constants import MAX_GRID_SIZE
from arcworld.shape.oop.base import ShapeObject

TRANSLATION_INCREMENT = 5


def erase(shape):
    return ShapeObject({})


def delete_shape_if_out_of_bound(shape):
    """verify the shape is in bound, delete shape otherwise. This is not a
    transformation but a helper function for the transformations"""
    if (
        shape.max_x >= MAX_GRID_SIZE
        or shape.max_y >= MAX_GRID_SIZE
        or shape.min_x < 0
        or shape.min_y < 0
    ):
        return ShapeObject({})
    else:
        return shape


def translate_up(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x - TRANSLATION_INCREMENT, y))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_down(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x + TRANSLATION_INCREMENT, y))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_right(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x, y + TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_left(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x, y - TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_up_right(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x - TRANSLATION_INCREMENT, y + TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_up_left(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x - TRANSLATION_INCREMENT, y - TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_down_right(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x + TRANSLATION_INCREMENT, y + TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def translate_down_left(shape):
    x, y = shape.current_position
    new_shape = ShapeObject(shape)
    new_shape.move_to_position((x + TRANSLATION_INCREMENT, y - TRANSLATION_INCREMENT))
    new_shape = delete_shape_if_out_of_bound(new_shape)
    return new_shape


def identity(shape):
    return ShapeObject(shape)


def rot90(shape):
    position = shape.current_position
    rotated = ShapeObject(np.rot90(shape.grid, 1))
    rotated.move_to_position(position)
    return rotated


def rot180(shape):
    position = shape.current_position
    rotated = ShapeObject(np.rot90(shape.grid, 2))
    rotated.move_to_position(position)
    return rotated


def rot270(shape):
    position = shape.current_position
    rotated = ShapeObject(np.rot90(shape.grid, 3))
    rotated.move_to_position(position)
    return rotated


def mirror_horizontal(shape):
    position = shape.current_position
    mirrored = ShapeObject(np.flipud(shape.grid))
    mirrored.move_to_position(position)
    return mirrored


def mirror_vertical(shape):
    position = shape.current_position
    mirrored = ShapeObject(np.fliplr(shape.grid))
    mirrored.move_to_position(position)
    return mirrored


def fill_holes(shape):  # Fills hole with the first color
    return ShapeObject(
        scipy.ndimage.binary_fill_holes(shape.grid).astype(int) * shape.colors[0]
    )


transformations_dict = {
    "erase": erase,
    "translate_up": translate_up,
    "translate_down": translate_down,
    "translate_right": translate_right,
    "translate_left": translate_left,
    "translate_up_right": translate_up_right,
    "translate_up_left": translate_up_left,
    "translate_down_right": translate_down_right,
    "translate_down_left": translate_left,
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "mirror_horizontal": mirror_horizontal,
    "mirror_vertical": mirror_vertical,
    "fill_holes": fill_holes,
}
