import random
from typing import List, Optional, cast

from tqdm import tqdm

from arcworld.dsl.arc_constants import FIVE
from arcworld.dsl.arc_types import Coordinate, Coordinates, Shapes, arcfrozenset
from arcworld.dsl.functional import (
    apply,
    both,
    compose,
    decrement,
    difference,
    fork,
    greater,
    height,
    lbind,
    mapply,
    neighbors,
    normalize,
    recolor,
    sfilter,
    width,
)
from arcworld.shape.dsl.augmentations import AUGMENTATION_OPTIONS


class ShapeGenerator:
    """
    Generates shapes.
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
        augmentations: Optional[List[str]] = None,
    ):
        """
        Args:
            max_pixels: Maximum number of pixels of a shape before augmentations.
            max_variations: Given a fixed pixel size, the maximum number of shapes
                generated with that fixed pixel size.
            augmentations: List of augmentations to apply. If None then
                the default set of augmentations is used.
        """
        self._max_pixels = max_pixels
        self._max_variations = max_variations

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
    def augmentations(self) -> List[str]:
        return self._augmentations

    def generate_random_proto_shapes(self) -> arcfrozenset[Coordinates]:
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
        upper_bound = self.max_pixels * self.max_variations
        bar = tqdm(total=upper_bound, desc="Generating random shapes")

        for num_pixels in range(self.max_pixels):
            for _ in range(self.max_variations):
                shape = self._generate_random_proto_shape(num_pixels)
                # Normalize the shape.
                norm_shape = normalize(arcfrozenset(shape))
                norm_shape = cast(Coordinates, norm_shape)
                random_shapes.add(norm_shape)
                bar.update()

        bar.close()

        assert len(random_shapes) <= upper_bound

        # Apply the transformations to each random shape.
        augmented_shapes: set[Coordinates] = set()
        for norm_shape in random_shapes:
            for name in self.augmentations:
                augmented = AUGMENTATION_OPTIONS[name](norm_shape)
                augmented_shapes.add(augmented)

        final_shapes = augmented_shapes | random_shapes
        print(f"Total augmented shapes: {len(final_shapes)}")

        return arcfrozenset(augmented_shapes | random_shapes)

    @staticmethod
    def _generate_random_proto_shape(pixel_size: int) -> set[Coordinate]:
        shape: set[Coordinate] = {(0, 0)}

        for _ in range(pixel_size):
            neighborhood = mapply(neighbors, shape)
            candidates = difference(neighborhood, shape)
            pixel = random.choice(tuple(candidates))
            shape.add(pixel)

        return shape

    def generate_random_shapes(self, max_obj_dimension: int = 4) -> Shapes:
        """
        Generates a set of random shapes all with the same color.
        Args:
            max_obj_dimension: The maximum dimension of the shapes.

        Returns:
            A set with the generated shapes.
        """
        proto_shapes = self.generate_random_proto_shapes()

        # Filter shapes by dimension.
        bounding_function = lbind(greater, max_obj_dimension)
        small_enough = compose(bounding_function, compose(decrement, height))
        narrow_enough = compose(bounding_function, compose(decrement, width))
        filter_function = fork(both, small_enough, narrow_enough)
        shapes_filtered = sfilter(proto_shapes, filter_function)

        recoloring_function = lbind(recolor, FIVE)
        shapes = apply(recoloring_function, shapes_filtered)
        # TODO: Until lbin is proprerly type hinted
        # we must use cast.
        shapes = cast(Shapes, shapes)

        return shapes
