import random
from typing import FrozenSet, List

import tqdm

from arcworld.dsl.arc_constants import FIVE
from arcworld.dsl.arc_types import Coordinates, IndicesSet, Shapes
from arcworld.dsl.functional import (
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
    mapply,
    neighbors,
    normalize,
    recolor,
    sfilter,
    width,
)
from arcworld.internal.complexity import Complexity
from arcworld.shape.dsl.augmentations import AUGMENTATION_OPTIONS

# TODO: Define Unit Tests.


class ShapeGenerator:
    """
    Generates shapes attributes.
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
        min_obj_pixels: int = 6,
        max_obj_pixels_proxy: int = 16,
        num_objs_per_size_proxy: int = 42,
        augmentations: List[str] = [],
        complexity: Complexity = Complexity.EASY,
    ):
        self._min_obj_pixels = min_obj_pixels
        self._max_obj_pixels_proxy = max_obj_pixels_proxy
        self._num_objs_per_size_proxy = num_objs_per_size_proxy

        if len(augmentations) > 0:
            self._augmentations = augmentations
        else:
            self._augmentations = ShapeGenerator.DEFAULT_AUGMENTATIONS

        self._complexity = complexity

    @property
    def augmentations(self) -> List[str]:
        return self._augmentations

    @property
    def min_obj_pixels(self) -> int:
        return self._min_obj_pixels

    @property
    def max_obj_pixels_proxy(self) -> int:
        return self._max_obj_pixels_proxy

    @property
    def num_objs_per_size_proxy(self) -> int:
        return self._num_objs_per_size_proxy

    @property
    def complexity(self) -> Complexity:
        return self._complexity

    def generate_random_proto_shapes(self) -> FrozenSet[IndicesSet]:
        """
        A ProtoShape is a Shape without coloring only their coordinates
        """
        random_shapes: set[IndicesSet] = set()

        # Naming here is strange since, pixel_range is not even taken into account.
        pixel_range = range(self.min_obj_pixels, self.max_obj_pixels_proxy + 1)

        end_index = self.max_obj_pixels_proxy - self.min_obj_pixels
        pbar = tqdm.tqdm(pixel_range, desc="generating random shapes (0)")

        # This doesn't make sense since dneighbors is a subt of neighbors.
        neighboring_schemes = [neighbors, dneighbors]

        for i, n in enumerate(pbar):
            for k in range(self.num_objs_per_size_proxy):  # Assume 42
                for neighboring_scheme in neighboring_schemes:
                    shap = {(0, 0)}
                    # Number of times you are adding a random
                    # pixel to the initial pixel.
                    for i in range(k - 1):
                        neighborhood = mapply(neighboring_scheme, shap)
                        candidates = difference(neighborhood, shap)
                        pixel = random.choice(tuple(candidates))
                        shap.add(pixel)
                    shap = normalize(frozenset(shap))
                    random_shapes.add(shap)
                    for name in self.augmentations:
                        augmented = AUGMENTATION_OPTIONS[name](shap)
                        random_shapes.add(augmented)

            if i == end_index:
                desc = f"generated {len(random_shapes)} random shapes"
                pbar.set_description(desc)
            else:
                desc = f"generating random shapes ({len(random_shapes)})"
                pbar.set_description(desc)

        return frozenset(random_shapes)

    def generate_random_shapes_new(self) -> FrozenSet[IndicesSet]:
        """
        Generate an object using random neighbors and augmentations.
        We start at coordinates (0,0) and consider as neighbors
        every pixel that is adjacent. Without augmentations and just
        one iteration of the algorithm you will have one object with
        1 pixel, one object with 2 pixels., up to one object with max_pixels.
        """
        random_shapes: set[IndicesSet] = set()

        max_pixels = 40  # Max size of a generated random shape, without augmentations.
        max_attempts_per_pixel_size = 10

        for num_pixels in range(max_pixels):
            for _ in range(max_attempts_per_pixel_size):
                shape = {(0, 0)}
                for _ in range(num_pixels):
                    neighborhood = mapply(neighbors, shape)
                    candidates = difference(neighborhood, shape)
                    pixel = random.choice(tuple(candidates))
                    shape.add(pixel)

                shape = normalize(frozenset(shape))
                random_shapes.add(shape)

        # At max we will have max_attempts_per_pixel_size objects for each
        # possible pixel size.
        assert len(random_shapes) <= max_pixels * max_attempts_per_pixel_size

        # Apply the transformations to each random shape.
        augmented_shapes: set[IndicesSet] = set()
        for shape in random_shapes:
            for name in self.augmentations:
                augmented = AUGMENTATION_OPTIONS[name](shape)
                augmented_shapes.add(augmented)

        return frozenset(augmented_shapes | random_shapes)

    # Just for debugging.
    @staticmethod
    def generate_random_proto_shape(pixel_size: int) -> Coordinates:
        shape: Coordinates = {(0, 0)}
        for _ in range(pixel_size):
            neighborhood = mapply(neighbors, shape)
            candidates = difference(neighborhood, shape)
            pixel = random.choice(tuple(candidates))
            shape.add(pixel)

        return shape

    def generate_random_shapes(self, max_obj_dimension: int = 4) -> Shapes:
        """
        Colors random generated proto shapes.
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

        return shapes
