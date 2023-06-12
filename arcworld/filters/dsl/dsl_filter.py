import arcworld.dsl.functional as F
from arcworld.dsl.arc_types import Shapes
from arcworld.filters.objects_filter import ShapesFilter
from arcworld.internal.program import FilterProgram


class DSLFilter(ShapesFilter):
    def __init__(self, program: FilterProgram):
        super().__init__()
        self._program = program

    @property
    def program(self) -> FilterProgram:
        return self._program

    def filter(self, objects: Shapes) -> Shapes:
        return F.sfilter(objects, self.program)
