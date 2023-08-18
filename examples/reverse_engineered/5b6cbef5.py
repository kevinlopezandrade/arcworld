# 5b6cbef5 from evaluation set
import random
from typing import Optional

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from arcworld.dsl.arc_types import Coordinate, Shape, Shapes
from arcworld.dsl.functional import (
    add,
    backdrop,
    canvas,
    height,
    multiply,
    normalize,
    paint,
    recolor,
    toindices,
    ulcorner,
    width,
)
from arcworld.filters.functional.shape_filter import FunctionalFilter
from arcworld.internal.constants import ALLOWED_COLORS, Example, Task
from arcworld.shape.dsl.generator import ShapeGeneratorDSL
from arcworld.storage.fingerprint import hash_task, normalize_task
from arcworld.storage.table import Base, Engineered

AUTHOR = "kevinlo@student.ethz.ch"


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


def generate(N_tasks: int, N_examples: int, seed: Optional[int] = None):  # noqa
    random.seed(seed)

    # With a square bounding box of 5, max dim is 25.
    generator = ShapeGeneratorDSL(max_variations=30, max_obj_dimension=4)
    shapes: Shapes = generator.generate_random_shapes()
    print("Total number of generated shapes: ", len(shapes))

    for _ in tqdm(range(N_tasks), desc="Generating tasks"):
        for filter in [FunctionalFilter("is_bbox_square", is_bbox_square)]:
            shapes = filter.filter(shapes)

        task: Task = []
        shapes_list = list(shapes)
        try:
            sampled_shapes = random.choices(shapes_list, k=N_examples)
            bg_color = 0
            for shape in sampled_shapes:
                fg_color = random.choice(list(ALLOWED_COLORS - {bg_color}))

                shape = recolor(fg_color, shape)
                dim = max(height(shape), width(shape))
                input_grid = canvas(bg_color, (dim, dim))
                input_grid = paint(input_grid, shape)
                input_grid = np.array(input_grid, dtype=np.uint8)

                output_grid = canvas(bg_color, (dim * dim, dim * dim))
                output_grid = paint(
                    output_grid, recolor(fg_color, replicate_based_on_shape(shape))
                )
                output_grid = np.array(output_grid, dtype=np.uint8)

                example = Example(input=input_grid, output=output_grid)
                task.append(example)

        except Exception:
            pass
        else:
            yield task


engine = create_engine("sqlite:////Users/kev/arcworld/datasets/tasks.db", echo=False)
Base.metadata.create_all(engine)

with Session(engine) as session:
    for task in generate(256, 4, 7):
        session.add(
            Engineered(
                id=hash_task(task),
                author=AUTHOR,
                transformation="5b6cbef5",
                task=normalize_task(task),
                split="evaluation",
            )
        )
        session.commit()
