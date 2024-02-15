import random
from typing import Optional

from arcdsl.arc_types import Objects
from tqdm import tqdm

from arcworld.internal.constants import Example, Task
from arcworld.objects.dsl.generator import ObjectGeneratorDSL
from arcworld.schematas.oop.drop.transforms import (
    DropBidirectional,
    DropBidirectionalDots,
    Gravitate,
)
from arcworld.utils import plot_task


def generate(N_tasks: int, N_examples: int, seed: Optional[int] = None):  # noqa
    if seed:
        print(f"Seed: {seed}")

    random.seed(seed)

    generator = ObjectGeneratorDSL(max_variations=30)
    shapes: Objects = generator.generate_random_objects()
    print("Total number of generated shapes: ", len(shapes))

    for _ in tqdm(range(N_tasks), desc="Generating tasks"):
        transform = random.choice(
            [Gravitate, DropBidirectional, DropBidirectionalDots]
        )()

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
        except Exception:
            pass
        else:
            yield task, transform.program


N = 10
succesful = 0
# seed = random.randint(1, 1000)
seed = 13
store = False

for task, program in generate(N, 4, seed):
    plot_task(task)
    succesful += 1

print(f"Succesfull ratio {succesful}/{N}")
