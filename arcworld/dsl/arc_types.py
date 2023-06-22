from typing import (
    Any,
    Container,
    FrozenSet,
    Iterable,
    Protocol,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

_T = TypeVar("_T")
_S = TypeVar("_S")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


# Custom Protocol
class IterableContainer(Iterable[_T], Container[_T], Sized, Protocol[_T]):
    # We define init here to make it compatible with the current containers.
    def __init__(self, __iterable: Iterable[_T]) -> None:
        ...


# NOTE: Extracted from typshed package.
# Comparison protocols
class SupportsDunderLT(Protocol[_T_contra]):
    def __lt__(self, __other: _T_contra) -> bool:
        ...


class SupportsDunderGT(Protocol[_T_contra]):
    def __gt__(self, __other: _T_contra) -> bool:
        ...


class SupportsDunderLE(Protocol[_T_contra]):
    def __le__(self, __other: _T_contra) -> bool:
        ...


class SupportsDunderGE(Protocol[_T_contra]):
    def __ge__(self, __other: _T_contra) -> bool:
        ...


SupportsRichComparison = Union[SupportsDunderLT[Any], SupportsDunderGT[Any]]


# Basic Types Aliases
Boolean = bool
Integer = int
Coordinate = Tuple[Integer, Integer]
Numerical = Union[Integer, Coordinate]
IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, Coordinate]

Shape = frozenset[Cell]
Shapes = frozenset[Shape]

Coordinates = frozenset[Coordinate]

IndicesSet = frozenset[Coordinates]

Patch = Union[Shape, Coordinates]
Element = Union[Shape, Grid]
Piece = Union[Grid, Patch]
