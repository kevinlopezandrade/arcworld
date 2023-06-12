# pyright: strict
from typing import Any, Callable, cast

from arcworld.dsl.arc_constants import ZERO
from arcworld.dsl.arc_types import Coordinates, Shape
from arcworld.dsl.functional import (
    backdrop,
    both,
    chain,
    compose,
    difference,
    equality,
    flip,
    fork,
    identity,
    matcher,
    outbox,
    positive,
    power,
    size,
    toindices,
)
from arcworld.dsl.util import build_function_from_program
from arcworld.internal.complexity import Complexity


class Program:
    def __init__(self, name: str, program_str: str):
        self._name = name
        self._program_str = program_str

    @property
    def name(self) -> str:
        return self._name

    @property
    def program_str(self) -> str:
        return self._program_str


class FilterProgram(Program):
    def __init__(
        self, name: str, program_str: str, complexity: Complexity = Complexity.EASY
    ):
        super().__init__(name, program_str)
        self._program = build_function_from_program(program_str)

    @property
    def program(self) -> Callable[..., Any]:
        return self._program

    def __call__(self, patch: Shape) -> bool:
        return self.program(patch)


class TransformProgram(Program):
    def __init__(
        self,
        name: str,
        program_str: str,
        margin: int = 2,
        complexity: Complexity = Complexity.EASY,
    ):
        super().__init__(name, program_str)

        raw_program = build_function_from_program(program_str)
        raw_program = cast(Callable[[Shape], Shape], raw_program)

        self._name = name
        self._program = raw_program
        self._is_valid_program = TransformProgram.build_is_valid_program(
            raw_program, margin
        )
        self._margin = margin
        self._complexity = Complexity.EASY

    @property
    def program(self) -> Callable[..., Any]:
        return self._program

    @property
    def name(self) -> str:
        return self._name

    def check_well_defined(self, shape: Shape) -> bool:
        return self._is_valid_program(shape)

    def __call__(self, shape: Shape) -> Shape:
        return self.program(shape)

    @staticmethod
    def build_is_valid_program(
        program: Callable[[Shape], Shape], margin: int
    ) -> Callable[[Shape], bool]:
        """
        Given a program transformation returns
        a test function that checks if the transformation
        is valid according to some criteria.
        """
        # First Test
        not_empty = chain(positive, size, program)

        # Second Test
        # TODO: Report To Github Pyright, to see why I need
        # two steps for this.
        f_is_equal = fork(equality, program, identity)
        not_identity = compose(flip, f_is_equal)

        # Third Test
        # Hack: I need to think how to handle this case better.
        f_outbox = cast(Callable[[Shape], Coordinates], power(outbox, margin))
        f_bbox = compose(backdrop, f_outbox)
        f_indices = compose(toindices, program)
        f_diff = fork(difference, f_bbox, f_indices)
        f_size_of_diff = compose(size, f_diff)

        not_out_of_bounds = matcher(
            f_size_of_diff,
            ZERO,
        )

        not_empty_or_identity = fork(both, not_empty, not_identity)
        is_valid = fork(both, not_empty_or_identity, not_out_of_bounds)

        return is_valid
