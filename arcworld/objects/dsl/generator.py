import logging
import random
from typing import Callable, List, Optional, cast

from arcdsl.arc_types import Coordinate, Coordinates, IterableContainer, Objects
from arcdsl.constants import FIVE
from arcdsl.dsl import (
    apply,
    both,
    compose,
    decrement,
    difference,
    dneighbors,
    fork,
    greater,
    height,
    lbind,
    merge,
    neighbors,
    normalize,
    recolor,
    sfilter,
    width,
)
from tqdm import tqdm

from arcworld.objects.dsl.augmentations import AUGMENTATION_OPTIONS

logger = logging.getLogger(__name__)


class ObjectGeneratorDSL:
    """
    Generates objects.
    """

    DEFAULT_AUGMENTATIONS = [
        "Identity",
        "AddVerticallyMirrored",
        "AddHorizontallyMirrored",
        "AddHVMirrored",
        "AddDiagonallyMirrored",
        "AddCounterdiagonallyMirrored",
        "AddDCMirrored",
        "AddMirrored",
        "AddBox",
        "AddOutBox",
        "AddDiagonalLine",
        "AddCounterdiagonalLine",
        "AddCross",
    ]

    def __init__(
        self,
        max_pixels: int = 30,
        max_variations: int = 10,
        max_obj_dimension: Optional[int] = None,
        augmentations: Optional[List[str]] = None,
    ):
        """
        Args:
            max_pixels: Maximum number of pixels of a shape before augmentations.
            max_variations: Given a fixed pixel size, the maximum number of shapes
                generated with that fixed pixel size.
            max_obj_dimension: After generating the random shapes with augmentations,
                max_obj_dimension to filter the shapes. If None no filtering is applied.
                Therefore no guarantee in the maximum dimension of the shapes.
            augmentations: List of augmentations to apply. If None then
                the default set of augmentations is used.
        """
        self._max_pixels = max_pixels
        self._max_variations = max_variations
        self._max_obj_dimension = max_obj_dimension

        if augmentations is None:
            self._augmentations = self.DEFAULT_AUGMENTATIONS
        else:
            self._augmentations = augmentations

    @property
    def max_pixels(self) -> int:
        return self._max_pixels

    @property
    def max_variations(self) -> int:
        return self._max_variations

    @property
    def max_obj_dimension(self) -> Optional[int]:
        return self._max_obj_dimension

    @property
    def augmentations(self) -> List[str]:
        return self._augmentations

    def generate_random_proto_shapes(self) -> frozenset[Coordinates]:
        """
        Generate an object using random neighbors and augmentations.
        We start at coordinates (0,0) and consider as neighbors evey
        pixel that is adjacent. We then choose a random neighbor an
        add it to the shape. After a full list of shapes has been
        generated, we apply the augmentations to increase diversity.

        Without augmentations and setting self.max_variations = 1 you
        will have one object with 1 pixel, one object with 2 pixels ...
        up to one object with self.max_pixels.

        Returns:
            frozenset with the generated colorless shapes all of them
            normalized.
        """
        random_shapes: set[Coordinates] = set()

        # At max we will have max_attempts_per_pixel_size objects for each
        # possible pixel size.
        upper_bound = self.max_pixels * self.max_variations * 2
        bar = tqdm(total=upper_bound, desc="Generating random shapes")

        for num_pixels in range(self.max_pixels):
            for f in [neighbors, dneighbors]:
                for _ in range(self.max_variations):
                    shape = self._generate_random_proto_shape(num_pixels, f)
                    # Normalize the shape.
                    norm_shape = normalize(frozenset(shape))
                    norm_shape = cast(Coordinates, norm_shape)
                    random_shapes.add(norm_shape)
                    bar.update()

        bar.close()

        assert len(random_shapes) <= upper_bound

        # Apply the transformations to each random shape.
        augmented_shapes: set[Coordinates] = set()
        for norm_shape in tqdm(random_shapes, desc="Augmenting shapes"):
            for name in self.augmentations:
                augmented = AUGMENTATION_OPTIONS[name](norm_shape)
                augmented_shapes.add(augmented)

        final_shapes = augmented_shapes | random_shapes
        logger.info(f"Total generated shapes: {len(final_shapes)}")

        return frozenset(augmented_shapes | random_shapes)

    @staticmethod
    def _generate_random_proto_shape(
        pixel_size: int, expander: Callable[[Coordinate], Coordinates]
    ) -> set[Coordinate]:
        shape: set[Coordinate] = {(0, 0)}

        for _ in range(pixel_size):
            d = apply(expander, shape)
            # Cast is necessary since my IterableContainer[_T] is invariant
            # in _T, since in merge I'm having a
            # IterableContainer[IterableContainer[_S]] my _T = IterableContainer[_S]
            # and if I'm trying to use a frozenset[_T] even though is virutal subclass
            # of IterableContainer[_T] it does not work because of invariance.
            d = cast(IterableContainer[IterableContainer[Coordinate]], d)
            neighborhood = merge(d)
            candidates = difference(neighborhood, shape)
            pixel = random.choice(tuple(candidates))
            shape.add(pixel)

        return shape

    def generate_random_objects(self) -> Objects:
        """
        Generates a set of random shapes all with the same color.

        Returns:
            A set with the generated shapes.
        """
        proto_shapes = self.generate_random_proto_shapes()

        if self.max_obj_dimension is not None:
            logger.info(f"Filtering by max dimension {self.max_obj_dimension}")
            # Filter shapes by dimension.
            bounding_function = lbind(greater, self.max_obj_dimension)
            small_enough = compose(bounding_function, compose(decrement, height))
            narrow_enough = compose(bounding_function, compose(decrement, width))
            filter_function = fork(both, small_enough, narrow_enough)
            proto_shapes = sfilter(proto_shapes, filter_function)

        recoloring_function = lbind(recolor, FIVE)
        shapes = apply(recoloring_function, proto_shapes)
        logger.info(f"Final number of shapes: {len(shapes)}")
        # TODO: Until lbin is proprerly type hinted
        # we must use cast.
        shapes = cast(Objects, shapes)

        return shapes
