from abc import ABCMeta, abstractmethod

from arcworld.dsl.arc_types import Shapes


class ObjectsTransform(metaclass=ABCMeta):
    """
    Interface for every object transform
    """

    @abstractmethod
    def transform(self, objects: Shapes) -> Shapes:
        pass
