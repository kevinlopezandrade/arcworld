import random
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from arcworld.dsl.arc_types import Shapes
from arcworld.internal.constants import Example, Task
from arcworld.schematas.oop.expansion.transforms import DotsExpansion
from arcworld.shape.dsl.generator import ShapeGeneratorDSL
from arcworld.storage.fingerprint import hash_task, normalize_task
from arcworld.storage.table import Base, Schemata

AUTHOR = "kevinlo@student.ethz.ch"


class Expansion(Schemata):
    __tablename__ = "expansion"


def generate(N_tasks: int, N_examples: int, seed: Optional[int] = None):  # noqa
    if seed:
        print(f"Seed: {seed}")

    random.seed(seed)

    generator = ShapeGeneratorDSL(max_variations=30, max_obj_dimension=3)
    shapes: Shapes = generator.generate_random_shapes()
    print("Total number of generated shapes: ", len(shapes))

    for _ in tqdm(range(N_tasks), desc="Generating tasks"):
        transform = DotsExpansion()

        filtered_shapes = shapes
        for filter in transform.filters:
            filtered_shapes = filter.filter(filtered_shapes)

        sampler = transform.grid_sampler()

        task: Task = []
        try:
            for _ in range(N_examples):
                grid_builder = sampler()
                input_grid = grid_builder.build_input_grid(filtered_shapes)
                ouput_grid = transform.transform(input_grid)

                example = Example(input=input_grid.grid_np, output=ouput_grid.grid_np)

                task.append(example)
        except Exception as e:
            print(e)
        else:
            yield task, transform.program


engine = create_engine("sqlite:////Users/kev/tasks.db", echo=False)

N = 10
succesful = 0
# seed = random.randint(1, 1000)
seed = 13
store = False

with Session(engine) as session:
    if store:
        Base.metadata.create_all(engine)

    for task, program in generate(N, 4, seed):
        if store:
            session.add(
                Expansion(
                    id=hash_task(task),
                    author=AUTHOR,
                    transformation=program,
                    task=normalize_task(task),
                )
            )
            session.commit()
        succesful += 1

print(f"Succesfull ratio {succesful}/{N}")
