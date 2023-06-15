from typing import (
    AbstractSet,
    Any,
    Container,
    FrozenSet,
    Iterable,
    Iterator,
    Protocol,
    Sized,
    Tuple,
    TypeVar,
    Union,
    cast,
)

_T = TypeVar("_T")
_S = TypeVar("_S")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


# Custom Protocol
class IterableContainer(Iterable[_T], Container[_T], Sized, Protocol):
    # We define init here to make it compatible with the current containers.
    def __init__(self, __iterable: Iterable[_T]) -> None:
        ...


# IMPORTANT: I need to add unit tests for this class.
# If Errors in the whole pipeline at last do the following
# arcfrozenset = frozenset
class arcfrozenset(frozenset[_T_co]):  # noqa: N801
    """
    In essence its just a frozenset where no modifications
    are made, besides from only adding an __init__. To make it
    compatible with our IterableContainer protocol. Therefore
    casting is secure.
    """

    def __init__(self, __iterable: Iterable[_T_co]) -> None:
        """
        Here we don't do anything since __new__ from frozenset is already called
        during object construction which gives a frozenset back to this __init__
        in self with type arcfrozenset.
        """
        ...

    def copy(self) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().copy())

    def difference(self, *s: Iterable[object]) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().difference(*s))

    def intersection(self, *s: Iterable[object]) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().intersection(*s))

    def isdisjoint(self, __s: Iterable[_T_co]) -> bool:
        return super().isdisjoint(__s)

    def issubset(self, __s: Iterable[object]) -> bool:
        return super().issubset(__s)

    def issuperset(self, __s: Iterable[object]) -> bool:
        return super().issuperset(__s)

    def symmetric_difference(self, __s: Iterable[_T_co]) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().symmetric_difference(__s))

    def union(self, *s: Iterable[_S]) -> "arcfrozenset[_T_co | _S]":
        return cast("arcfrozenset[_T_co | _S]", super().union(*s))

    def __len__(self) -> int:
        return super().__len__()

    def __contains__(self, __o: object) -> bool:
        return super().__contains__(__o)

    def __iter__(self) -> Iterator[_T_co]:
        return super().__iter__()

    def __and__(self, __value: AbstractSet[_T_co]) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().__and__(__value))

    def __or__(self, __value: AbstractSet[_S]) -> "arcfrozenset[_T_co | _S]":
        return cast("arcfrozenset[_T_co]", super().__or__(__value))

    def __sub__(self, __value: AbstractSet[_T_co]) -> "arcfrozenset[_T_co]":
        return cast("arcfrozenset[_T_co]", super().__sub__(__value))

    def __xor__(self, __value: AbstractSet[_S]) -> "arcfrozenset[_T_co | _S]":
        return cast("arcfrozenset[_T_co]", super().__xor__(__value))

    def __le__(self, __value: AbstractSet[object]) -> bool:
        return super().__le__(__value)

    def __lt__(self, __value: AbstractSet[object]) -> bool:
        return super().__lt__(__value)

    def __ge__(self, __value: AbstractSet[object]) -> bool:
        return super().__ge__(__value)

    def __gt__(self, __value: AbstractSet[object]) -> bool:
        return super().__gt__(__value)


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
# Shape = Set[Cell]
Shape = arcfrozenset[Cell]
Shapes = arcfrozenset[Shape]

# For the moment use Sets until I figure out how to
# avoid the type checking errors.
Coordinates = arcfrozenset[Coordinate]
IndicesSet = arcfrozenset[Coordinates]

Patch = Union[Shape, Coordinates]
Element = Union[Shape, Grid]
Piece = Union[Grid, Patch]

# testing: IterableContainer[int] = ARCFrozenSet(i for i in range(10))
# tesi: IterableContainer[int] = ARCFrozenSet(i for i in range(10)) & testing
