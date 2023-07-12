from typing import Callable, Generic, Optional, TypeVar

_T = TypeVar("_T")


class Node(Generic[_T]):
    def __init__(self, key: _T):
        self.left: Optional[Node[_T]] = None
        self.right: Optional[Node[_T]] = None
        self.parent: Optional[Node[_T]] = None
        self.key = key


class Tree(Generic[_T]):
    def __init__(self, root_node: Optional[Node[_T]]) -> None:
        self.root = root_node


def _default_st(a: Node[_T], b: Node[_T]) -> bool:
    return a.key < b.key


SmallerThan = Callable[[Node[_T], Node[_T]], bool]


def bst_insert(T: Tree[_T], z: Node[_T], st: SmallerThan[_T] = _default_st):  # noqa
    """
    Inserts a node in the binary search tree.

    Args:
        T: The Tree
        z: The node to be inserted.
        st: Optional function f(a,b) to be passed to use when comparing keys.
            if a < b then it must return True, otherwise False.
    """
    y = None
    x = T.root

    while x is not None:
        y = x
        if st(z, x):
            x = x.left
        else:
            x = x.right

    z.parent = y

    if y is None:
        T.root = z
    elif st(z, y):
        y.left = z
    else:
        y.right = z
