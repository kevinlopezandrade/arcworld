import logging
import random
from typing import List, Type

from tqdm import tqdm

from arcworld.dsl.arc_types import Shapes
from arcworld.schematas.oop.subgrid_pickup.base_pickup import SubgridPickup
from arcworld.shape.base_generator import ShapeGenerator
from arcworld.transformations.base_transform import GridsTransform

logger = logging.getLogger(__name__)


class SubgridPickupTaskGenerator:
    """
    Class to generate a random dataset of training samples.
    A ShapeGenerator must be set before starting the generation.
    """

    def __init__(
        self, pickups: List[Type[SubgridPickup]], alterations: List[GridsTransform]
    ) -> None:
        """
        Args:
            pickups: A list with all the SubgridPickup classes
                from which to sample one when generating a single training
                sample.
            alterations: A list with all the



        """
        self.pickups = pickups
        self.alterations = alterations

    def set_shape_generator(self, generator: ShapeGenerator):
        self.generator = generator

    def _generate_sample(
        self,
        shapes: Shapes,
        pickup: SubgridPickup,
        transform: GridsTransform,
        n_tasks: int,
    ):
        sample = []
        grid_sampler = pickup.create_grid_sampler()

        for i in range(n_tasks):
            logger.debug(f"Trying to generate task {i}")
            input_grid = grid_sampler.sample_input_grid(shapes)
            ouput_grid = transform.transform([pickup(input_grid)], 7)[0]

            sample.append(
                {
                    "input": input_grid.grid,
                    "output": ouput_grid,
                    "shapes": input_grid.shapes,
                }
            )

        return sample

    def generate_random_dataset(
        self, n_samples: int, n_tasks: int, silent: bool = False
    ):
        """
        Generator for random samples.

        Args:
            n_samples: Desired number of training examples to generate.
            n_tasks: Number of tasks per training sample.
        """
        shapes = self.generator.generate_random_shapes()

        for filter in SubgridPickup.DEFAULT_CONDITIONS:
            shapes = filter.filter(shapes, silent=False)

        logger.info(f"Using {len(shapes)} shapes.")
        if silent:
            iterator = range(n_samples)
        else:
            iterator = tqdm(range(n_samples), desc="Generating training samples")

        for i in iterator:
            pickup = random.choice(self.pickups)()
            alteration = random.choice(self.alterations)

            logger.debug(f"Generating sample {i}")
            try:
                sample = self._generate_sample(shapes, pickup, alteration, n_tasks)
            except Exception:
                logger.debug(f"Could not generate sample {i}")
            else:
                logger.debug(
                    f"Generated sample {i}, with {pickup.name}, {alteration.name}"
                )
                yield sample
