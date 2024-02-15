from typing import Callable

from tqdm import tqdm

from arcworld.dsl.arc_types import Object, Objects
from arcworld.filters.objects_filter import ObjectsFilter


class FunctionalFilter(ObjectsFilter):
    def __init__(self, name: str, func: Callable[..., bool]) -> None:
        super().__init__()
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def filter(self, objects: Objects, silent: bool = True) -> Objects:
        res: set[Object] = set()

        if silent:
            iterator = objects
        else:
            iterator = tqdm(
                objects,
                desc=f"Filtering shapes with {self.name}",
                bar_format=ObjectsFilter.BAR_FORMAT,
            )

        for shape in iterator:
            if self._func(shape):
                res.add(shape)

        return frozenset(res)
