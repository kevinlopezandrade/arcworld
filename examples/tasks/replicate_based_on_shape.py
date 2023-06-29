import random

from arcworld.dsl.arc_types import Coordinate, Shape
from arcworld.dsl.functional import (
    add,
    backdrop,
    height,
    multiply,
    normalize,
    recolor,
    toindices,
    ulcorner,
    width,
)
from arcworld.filters.functional.shape_filter import FunctionalFilter
from arcworld.shape.dsl.generator import ShapeGeneratorDSL
from arcworld.utils import plot_shapes


def is_bbox_square(shape: Shape):
    bbox = backdrop(shape)
    h = height(bbox)
    w = width(bbox)

    if h == w:
        return True
    else:
        return False


def replicate_based_on_shape(shape: Shape) -> Shape:
    ur = ulcorner(shape)
    dim_bbox = height(backdrop(shape))

    new_shape: set[Coordinate] = set()
    for i, j in toindices(shape):
        d = multiply(dim_bbox, add((i, j), (-ur[0], -ur[1])))
        for i, j in toindices(shape):
            new_pos = add((i, j), d)
            new_shape.add(new_pos)

    return normalize(recolor(5, new_shape))


# def new(shaep: Shape) -> Shape:
# Think about paths
# shape  -> urcorner(shape)
#        -> height(backdrop(shape))
# compose(lbdind(multiply, height(backdrop(shape))), rbind(substract, urcorner(shape))
# apply(apply(), toindices(shape))


generator = ShapeGeneratorDSL(max_obj_dimension=5)
shapes = generator.generate_random_shapes()

filter = FunctionalFilter("is_bbox_square", is_bbox_square)
shapes = list(filter.filter(shapes))
random.shuffle(shapes)


# shapes = []
# shapes.append({(7, (0,0)), (7, (0, 1)), (7, (1, 2)), (7, (2, 0))})

N = 5
for i, shape in enumerate(shapes):
    tr_shape = replicate_based_on_shape(shape)
    plot_shapes(shape, tr_shape)

    if i == N - 1:
        break
