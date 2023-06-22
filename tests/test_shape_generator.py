import random

from arcworld.dsl.functional import normalize
from arcworld.shape.dsl.generator import ShapeGeneratorDSL


def test_no_augmentations_no_variations():
    max_pixels = random.randint(10, 100)

    generator = ShapeGeneratorDSL(
        max_pixels=max_pixels, max_variations=1, augmentations=[]
    )

    shapes = generator.generate_random_proto_shapes()

    assert len(shapes) <= max_pixels * 2

    sizes = set(i for i in range(1, max_pixels + 1))
    current_sizes = set(len(shape) for shape in shapes)

    assert sizes == current_sizes


def test_normalized_shapes():
    generator = ShapeGeneratorDSL()
    shapes = generator.generate_random_proto_shapes()

    assert all(normalize(shape) == shape for shape in shapes)
