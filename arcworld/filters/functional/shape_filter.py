from typing import Callable, cast

from arcworld.dsl.arc_types import Shapes
from arcworld.filters.functional.single_shape_conditionals import CONDITIONALS
from arcworld.filters.objects_filter import ShapesFilter


class FunctionalFilter(ShapesFilter):
    def __init__(self, name: str, func: Callable[..., bool]) -> None:
        super().__init__()
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def filter(self, objects: Shapes) -> Shapes:
        res: Shapes = set()

        for shape in objects:
            if self._func(shape):
                res.add(shape)

        return res


def get_filter(name: str) -> FunctionalFilter:
    """
    Given the name returns an object filter, from
    the set of possible filters already predefined.
    """

    if name not in CONDITIONALS.keys():
        raise ValueError(f"{name} does not correspond to any conditional")

    func = CONDITIONALS[name]
    func = cast(Callable[..., bool], func)

    return FunctionalFilter(name, func)
