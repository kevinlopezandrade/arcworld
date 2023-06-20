from abc import abstractmethod
from typing import Protocol, Tuple

from arcworld.dsl.arc_types import Grid, Shape


# Define the Grid Protocol
class GridProtocol(Protocol):
    @property
    def grid(self) -> Grid:
        ...

    @property
    def shapes(self) -> Tuple[Shape]:
        ...

    @abstractmethod
    def place_object(self, shape: Shape) -> Shape:
        ...