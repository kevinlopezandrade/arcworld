import logging
import random
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from arcworld.dsl.arc_types import Shape, Shapes
from arcworld.dsl.functional import height, size, width

logger = logging.getLogger(__name__)


def range_c(start: int, end: int):
    """
    Generator that generates a sequence in
    [start, end]

    Args:
        start: Lower bound
        end: Upper bound
    """
    i = start
    while i <= end:
        yield i
        i += 1


class Resampler(metaclass=ABCMeta):
    """
    Interface that all the resamplers for the subgridpickup
    schemata must follow.
    """

    @abstractmethod
    def resample(self, shapes: Shapes, n_shapes_per_grid: int) -> List[Shape]:
        """
        Derived classes should implement this method to define
        the resampling strategies.

        Args:
            shapes: List of shapes for which to resample.
            n_shapes_per_grid: Number of samples to resample.
        Returns:
            List of resampled shapes.
        """
        ...


class OnlyShapesRepeated(Resampler):
    """
    Resampler that given a list of shapes, chooses at random
    one shape, and creates copies of it.
    """

    def resample(self, shapes: Shapes, n_shapes_per_grid: int) -> List[Shape]:
        selected_shape = random.choice(list(shapes))

        res: List[Shape] = []
        for _ in range(n_shapes_per_grid):
            res.append(selected_shape)

        return res


class UniqueShapeParemeter(Resampler):
    """
    Resamples shapes in a way where a given parameter is controlled.
    It ensures we have no two shapes that have same number of e.g. n_cells.
    """

    ALLOWED_PARAMS = ("n_rows", "n_cols", "n_cells")

    def __init__(self, param: str, max_trials: int = 100):
        if param not in self.ALLOWED_PARAMS:
            raise ValueError("Unknow shape paremeter to control.")

        self.param = param
        self.max_trials = max_trials

    def _get_param(self, shape: Shape) -> int:
        if self.param == "n_rows":
            return height(shape)
        elif self.param == "n_cols":
            return width(shape)
        else:
            return size(shape)

    def resample(self, shapes: Shapes, n_shapes_per_grid: int) -> List[Shape]:
        # Permute the indices to make it random. It maybe be
        # inefficient if the list of shapes is to big.
        shapes_list = list(shapes)

        indices = list(range(0, len(shapes_list)))
        random.shuffle(indices)

        selected_shapes = [shapes_list[indices[0]]]
        seen = {self._get_param(selected_shapes[0])}

        index = 1
        while (
            len(selected_shapes) < n_shapes_per_grid
            and index < len(indices)
            and index < self.max_trials
        ):
            candidate_shape = shapes_list[indices[index]]
            value = self._get_param(candidate_shape)

            if value not in seen:
                selected_shapes.append(candidate_shape)
                seen.add(value)

            index += 1

        return selected_shapes


class RepeatedShapesResampler(Resampler):
    """
    Resamples a set of shapes to fullfill some frequency
    of shapes constraint.
    """

    def __init__(self, median: bool = False) -> None:
        """
        Args:
            median: Boolean flag normaly set by Subgridpickup objects.
        """
        self.median = median

    @staticmethod
    def _generate_three_freq(n: int) -> Tuple[int, int, int]:
        """
        Generatas a triple (p1, p2, p3) where the three numbers are all
        distinct and p1 + p2 + p3 = n.

        Ad-hoc algorithm derived by hand, not sure it is more readable
        than the previous brute force one of generating all the unique triplets.

        Args:
            n: The number the three generated numbers should add up to.

        Return:
            Triplet with the three random generated numbers.
        """
        # Upper limit, sample only up to n - 3.
        valid_numbers = set(range_c(1, n - 3))
        p1 = random.choice(list(valid_numbers))

        # Generate new set of possible numbers using previous number.
        valid_numbers = set(range_c(1, n - p1 - 1))

        # Remove myself from the set.
        valid_numbers = valid_numbers - {p1}

        # Remove my complement as well.
        valid_numbers = valid_numbers - {n - 2 * p1}

        # If even then remove my half as well.
        if (n - p1) % 2 == 0:
            valid_numbers = valid_numbers - {int((n - p1) / 2)}

        # Sample second number and compute remainder.
        p2 = random.choice(list(valid_numbers))
        p3 = (n - p1) - p2

        freq = (p1, p2, p3)

        assert sum(freq) == n

        return freq

    @staticmethod
    def _generate_two_freq(n: int) -> Tuple[int, int]:
        """
        Generatas a tuple (p1, p2) where the two numbers are
        distinct and p1 + p2 = n.

        Args:
            n: The number the two generated numbers should add up to.

        Returns:
            The tuple with the two random generated numbers.
        """
        valid_numbers = set(range_c(1, n - 1))

        if n % 2 == 0:
            valid_numbers = valid_numbers - {int(n / 2)}

        p = random.choice(list(valid_numbers))

        freq = (p, n - p)

        assert sum(freq) == p

        return freq

    def resample(self, shapes: Shapes, n_shapes_per_grid: int) -> List[Shape]:
        if self.median:
            # Otherwise there is no way we get something where shapes
            # are repeated at 3 different frequency.
            n_shapes_per_grid = n_shapes_per_grid if n_shapes_per_grid > 6 else 6
            n_combinations = 3
        else:
            n_combinations = random.randint(2, 3) if n_shapes_per_grid > 6 else 2

        if n_combinations == 3:
            frequencies = self._generate_three_freq(n_shapes_per_grid)
        else:
            frequencies = self._generate_two_freq(n_shapes_per_grid)

        selected_shapes = random.sample(shapes, n_combinations)

        res: List[Shape] = []
        for i, freq in enumerate(frequencies):
            shape = selected_shapes[i]

            for _ in range(freq):
                res.append(shape)

        random.shuffle(res)

        return res
