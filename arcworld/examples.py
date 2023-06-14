from typing import List

from arcworld.deprecated.generator_utils import read_base_programs
from arcworld.filters.dsl.dsl_filter import DSLFilter
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.grid.oop.grid_oop import GridObject
from arcworld.internal.program import FilterProgram, TransformProgram
from arcworld.schematas.oop.subgrid_pickup.base_pickup import SubgridPickup
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


class SubgridPickupSchemata:
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

    def __init__(self, shape_conditions: List[ShapesFilter], subgrid_pickup) -> None:
        self._shape_generator = ShapeGenerator()
        self._conditions = shape_conditions
        self._subgrid_pickup = subgrid_pickup

    def execute(self) -> GridObject:
        """
        The way Yassine does it. All the Parameters of the Grid Sampler, are passed
        based on the constraints of the SubgridPickup. So we could define an class
        of These Transformations that contain the paremeters and then do something like.

        However shape paremeter to distinguish is only for the Resampler, therefore I cannot really.

        Maybe I can have something like, args and extra-args methods ?

        So I have a class, that is actually in charge of creating. Setting the paremeters
        in the way thy actually want.

        So that I define something like Create Grid Sampler, which will already ensure the.
        """
        shapes = self._shape_generator.generate_random_shapes(5)
        # for filter in self._conditions:
        #     shapes = filter.filter(shapes)

        # Think about injecting the dependence instead of building the
        # SubgridPickupGridSampler here. Since actually the
        # SubgridPickupGridSampler you will only use it once. And then
        # reuse every time.

        # I create the sampler and then the pick ups
        # grid_sampler = SubgridPickupGridSampler()
        # grid_sampler.set_resampler()

        # This creates a grid sampler, with already the conditions satisfied.
        # Therefore the subgrid_pickup must either create a grid sample that already satisfies
        # the conditions or.
        grid_sampler = self._subgrid_pickup.create_grid_sampler()

        grid = grid_sampler.sample_input_grid(shapes)
        pickup = self._subgrid_pickup.apply(grid, grid.shapes)
        alteration_pickup = self.pickup_alteration(pickup)

        return grid

    def _sample_subgrid_pickup(self) -> SubgridPickup:
        ...

    def _sample_pickup_alteration(self) -> TransformProgram:
        ...

    def generate_task(self, shapes):
        # Ideally
        subgrid_pickup = self._sample_subgrid_pickup()
        pickup_alteration = self._sample_pickup_alteration()

        conditions = subgrid_pickup.conditions
        for condition in conditions:
            shapes = condition.filter(shapes)

        grid_sampler = subgrid_pickup.create_grid_sampler()

        grid = grid_sampler.sample_grid(shapes)
        pickup = subgrid_pickup.apply(subgrid_pickup)
        alteration = pickup_alteration.transform(alteration)

        return {}

        # subgrid_pickup

    def genereate_set_of_tasks(self, n_tasks: int):
        pass
