import logging

from arcworld.schematas.oop.subgrid_pickup.pickups import DEFAULT_PICKUPS
from arcworld.schematas.oop.subgrid_pickup.task_generator import (
    SubgridPickupTaskGenerator,
)
from arcworld.shape.dsl.generator import ShapeGeneratorDSL
from arcworld.transformations.oop.base_pickup_alteration import DEFAULT_ALTERATIONS

# Change to DEBUG for more logging details.
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    generator = ShapeGeneratorDSL(max_variations=30)

    task_generator = SubgridPickupTaskGenerator(
        DEFAULT_PICKUPS,
        DEFAULT_ALTERATIONS,
    )
    task_generator.set_shape_generator(generator)

    for sample in task_generator.generate_random_dataset(10, 4, silent=False):
        # Do something here with the samples.
        pass
