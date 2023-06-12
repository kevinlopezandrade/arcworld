from typing import List

from arcworld.deprecated.generator_utils import read_base_programs
from arcworld.filters.dsl.dsl_filter import DSLFilter
from arcworld.grid.subgrid_pickup_sampler import GridObject, SubgridPickupGridSampler
from arcworld.internal.program import FilterProgram, TransformProgram
from arcworld.shape.dsl.generator import ShapeGenerator
from arcworld.transformations.dsl.dsl_transform import DSLTransform


class ConditionalSchemataDSL:
    """
    Kind of follows a strategy pattern desing,
    with dependency injections.
    """

    def __init__(self, conditions_path: str, transformations_path: str):
        self._shape_generator = ShapeGenerator()
        self._conditions = read_base_programs(conditions_path, test_run=True)
        self._transformations = read_base_programs(transformations_path, test_run=True)

    def execute(self):
        shapes = self._shape_generator.generate_random_shapes()

        for condition in self._conditions:
            for transformation in self._transformations:
                filter = DSLFilter(FilterProgram(*condition))
                transform = DSLTransform(TransformProgram(*transformation))

                filter.filter(shapes)
                transform.transform(shapes)


class SubgridPickup:
    """
    Yassine uses a different approach to Generate Shapes. He has generated a set
    of shapes, and store them already with the conditions he has defined. I don't
    want to reuse that. I want to use the shape generation process of Michael. So
    extract the part of Yassines code where he tests for Shapes and Do it with
    Michael filters. Later check how we deal with the checks for the Grid Itself.
    The first logic is to generate the objecst as before and then the
    transformations is done at the grid level not at the object level.

    Yassine constrcuts the Filters for the Objects after Processing some paremeters
    like shape of the grid and some constraints etc. After that what he does is to
    resample those generated shapes in order to satisfy some conditions. Based on
    the conditions.

    So We Need a Filter Program as Usual -> Then a Resmampler Program from those
    Already Filtered -> Placing the objects in the Grid.

    Let's try to do without using the shapes generated from Michael to test
    how well the current design supports interoperaibility.
    """

    def __init__(self, conditions: List[DSLFilter]) -> None:
        self._shape_generator = ShapeGenerator()
        self._conditions = conditions

    def execute(self) -> GridObject:
        shapes = self._shape_generator.generate_random_shapes(5)
        for filter in self._conditions:
            shapes = filter.filter(shapes)

        # Think about injecting the dependence instead of building the
        # SubgridPickupGridSampler here. Since actually the
        # SubgridPickupGridSampler you will only use it once. And then
        # reuse every time.
        grid_sampler = SubgridPickupGridSampler()
        grid = grid_sampler.sample_input_grid(shapes)

        return grid
