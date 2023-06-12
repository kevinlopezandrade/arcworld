from typing import (
    AbstractSet,
    Any,
    Container,
    FrozenSet,
    Generic,
    Iterable,
    Protocol,
    Set,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


# Custom Protocol
class IterableContainer(Iterable[_T], Container[_T], Sized, Protocol):
    def __init__(self, __iterable: Iterable[_T]) -> None:
        ...


class ARCFrozenSet(Generic[_T_co], AbstractSet[_T_co]):
    def __init__(self, __iterable: Iterable[_T_co] = []) -> None:
        self._frozenset = frozenset(__iterable)

    def __and__(self, __value: AbstractSet[_T_co]) -> "ARCFrozenSet[_T_co]":
        _aux = self._frozenset.__and__(__value)
        return ARCFrozenSet.init_from_set(_aux)

    @classmethod
    def init_from_set(cls, E: frozenset[_T_co]) -> "ARCFrozenSet[_T_co]":
        arcfrozenset = cls()
        arcfrozenset._frozenset = E
        return arcfrozenset


class ARCTuple(tuple[_T_co]):
    def __init__(self, __iterable: Iterable[_T_co] = []) -> None:
        super().__new__(ARCTuple, __iterable)


# class ARCTuple(Generic[_T_co], ABC):
#     def __new__(cls, __iterable: Iterable[_T_co] = []):
#         tuple.__new__(cls, __iterable)

#     def __init__(self, __iterable: Iterable[_T_co] = []) -> None:
#         self.__dict__ = tuple.__dict__

# ARCTuple.register(tuple)


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

# Avoid FrozenSets since we cannot use them in our protocol.
# TODO: MaybeCreate a Custom FrozenSet that satisfies our protocol.
# Objects have been rename to Shapes, and we use FrozenSet here as well.
Shape = Set[Cell]
Shapes = Set[Shape]

# For the moment use Sets until I figure out how to
# avoid the type checking errors.
Coordinates = Set[Coordinate]
IndicesSet = Set[Coordinates]

Patch = Union[Shape, Coordinates]
Element = Union[Shape, Grid]
Piece = Union[Grid, Patch]

# testing: IterableContainer[int] = ARCFrozenSet(i for i in range(10))
# tesi: IterableContainer[int] = ARCFrozenSet(i for i in range(10)) & testing
