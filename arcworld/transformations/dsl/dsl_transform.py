from typing import Set, cast

import arcworld.dsl.functional as F
from arcworld.dsl.arc_types import Shape, Shapes
from arcworld.internal.program import TransformProgram
from arcworld.transformations.objects_transform import ObjectsTransform


class DSLTransform(ObjectsTransform):
    def __init__(self, program: TransformProgram):
        self._program = program

    @property
    def program(self) -> TransformProgram:
        return self._program

    def transform(self, objects: Shapes) -> Shapes:
        return cast(Set[Shape], F.apply(self.program, objects))
