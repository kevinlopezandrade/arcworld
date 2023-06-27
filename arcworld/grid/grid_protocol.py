from abc import abstractmethod
from typing import Any, Dict, List, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from arcworld.dsl.arc_types import Grid, Shape
from arcworld.shape.oop.base import ShapeObject

_T = TypeVar("_T", Shape, ShapeObject)
_S = TypeVar("_S", Grid, NDArray[np.int8])


# Define the Grid Protocol
class GridProtocol(Protocol):
    def __init__(self, h: int, w: int, **kwargs: Dict[str, Any]) -> None:
        super().__init__()

    @property
    def grid(self) -> Grid:
        ...

    @property
    def shapes(self) -> List[_T]:
        ...

    @abstractmethod
    def place_object(self, shape: _T) -> _T:
        ...
