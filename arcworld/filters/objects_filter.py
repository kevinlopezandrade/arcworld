from abc import ABCMeta, abstractmethod

from arcworld.dsl.arc_types import Shapes


class ShapesFilter(metaclass=ABCMeta):
    """
    Interface for every object filterer.
    """

    @abstractmethod
    def filter(self, objects: Shapes) -> Shapes:
        pass
