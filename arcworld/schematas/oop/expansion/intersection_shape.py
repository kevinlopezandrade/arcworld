from arcworld.dsl.arc_types import Shape
from arcworld.schematas.oop.expansion.grid import LinesGrid


def no_op(dot: Shape, shape: Shape, grid: LinesGrid):
    pass


POLICIES = {"no_op": no_op}
