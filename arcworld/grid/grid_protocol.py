from abc import abstractmethod
from typing import Any, List, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Grid, Shape
from arcworld.shape.oop.base import ShapeObject

_T = TypeVar("_T", Shape, ShapeObject)
_S = TypeVar("_S", Grid, NDArray[np.int8])


# Define the Grid Protocol
class GridProtocol(Protocol):
    @property
    def height(self) -> int:
        ...

    @property
    def width(self) -> int:
        ...

    @property
    def grid(self) -> Any:
        ...

    @property
    def shapes(self) -> List[Any]:
        ...

    @abstractmethod
    def place_object(self, shape: Any) -> Any:
        ...
