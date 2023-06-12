import ast
from typing import Any, Callable

import arcworld.dsl.arc_constants as arc_constants
import arcworld.dsl.functional as F

# TODO: Check how to DEAL with the renaming of T and F.
ALLOWED_NAMESPACE = vars(F) | vars(arc_constants)


def build_function_from_program(program: str) -> Callable[..., Any]:
    program_ast = ast.parse(program, mode="eval")

    for node in ast.walk(program_ast.body):
        if isinstance(node, ast.Name):
            name = node.id
            if not (hasattr(F, name) or hasattr(arc_constants, name)):
                raise ValueError(
                    "Command contains functions and constants outside the DSL"
                )

    compiled_program = compile(program_ast, filename="<string>", mode="eval")
    function = eval(compiled_program, ALLOWED_NAMESPACE)

    return function
