import logging
from typing import cast

from tqdm import tqdm

import arcworld.dsl.functional as F
from arcworld.dsl.arc_types import Objects
from arcworld.filters.objects_filter import ObjectsFilter
from arcworld.internal.program import FilterProgram

logger = logging.getLogger(__name__)


class DSLFilter(ObjectsFilter):
    def __init__(self, program: FilterProgram):
        super().__init__()
        self._program = program

    @property
    def program(self) -> FilterProgram:
        return self._program

    def filter(self, objects: Objects, silent: bool = True) -> Objects:
        if silent:
            return cast(Objects, F.sfilter(objects, self.program))
        else:
            bar = tqdm(
                (e for e in objects if self.program(e)),
                total=len(objects),
                bar_format=ObjectsFilter.BAR_FORMAT,
                desc=f"Filtering shapes with {self.program.name}",
            )
            return frozenset(bar)
