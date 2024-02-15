from typing import cast

import arcworld.dsl.functional as F
from arcworld.dsl.arc_types import Object, Objects
from arcworld.internal.program import TransformProgram
from arcworld.transformations.base_transform import ObjectsTransform


class DSLTransform(ObjectsTransform):
    def __init__(self, program: TransformProgram):
        self._program = program

    @property
    def program(self) -> TransformProgram:
        return self._program

    def transform(self, objects: Objects) -> Objects:
        return cast(frozenset[Object], F.apply(self.program, objects))
