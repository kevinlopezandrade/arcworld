from typing import Callable, cast

from tqdm import tqdm

from arcworld.dsl.arc_types import Shape, Shapes
from arcworld.filters.functional.single_shape_conditionals import CONDITIONALS
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.grid.oop.grid_oop import to_shape_object


class FunctionalFilter(ShapesFilter):
    def __init__(
        self, name: str, func: Callable[..., bool], to_shape_object: bool = False
    ) -> None:
        super().__init__()
        self._name = name
        self._func = func
        self._to_shape_object = to_shape_object

    @property
    def name(self) -> str:
        return self._name

    def filter(self, objects: Shapes, silent: bool = True) -> Shapes:
        res: set[Shape] = set()

        if silent:
            iterator = objects
        else:
            iterator = tqdm(
                objects,
                desc=f"Filtering shapes with {self.name}",
                bar_format=ShapesFilter.BAR_FORMAT,
            )

        for shape in iterator:
            aux = shape

            if self._to_shape_object:
                aux = to_shape_object(shape)

            if self._func(aux):
                res.add(shape)

        return frozenset(res)


def get_filter(name: str) -> FunctionalFilter:
    """
    Given the name returns an object filter, from
    the set of possible filters already predefined.
    """

    if name not in CONDITIONALS.keys():
        raise ValueError(f"{name} does not correspond to any conditional")

    func = CONDITIONALS[name]
    func = cast(Callable[..., bool], func)

    return FunctionalFilter(name, func, to_shape_object=True)
