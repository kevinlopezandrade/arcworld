from abc import ABCMeta, abstractmethod

from arcworld.dsl.arc_types import Objects


class ObjectsFilter(metaclass=ABCMeta):
    """
    Interface for every shape filterer.
    """

    BAR_FORMAT = (
        "{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    @abstractmethod
    def filter(self, objects: Objects, silent: bool = True) -> Objects:
        pass
