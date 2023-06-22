import logging
import pickle
from typing import Protocol

from tqdm import tqdm

from arcworld.dsl.arc_types import Shapes
from arcworld.dsl.functional import lbind, recolor

logger = logging.getLogger(__name__)


class ShapeGenerator(Protocol):
    def generate_random_shapes(self) -> Shapes:
        ...


class ShapeGeneratorFromFile:
    def __init__(self, path: str) -> None:
        self.path = path

    def generate_random_shapes(self) -> Shapes:
        logger.info(f"Loading shapes from {self.path}")
        with open(self.path, "rb") as f:
            shapes = pickle.load(f)

        logger.info(f"Total number of shapes: {len(shapes)}")
        return shapes


class ShapeGeneratorFromProtoFile:
    def __init__(self, path: str) -> None:
        self.path = path

    def generate_random_shapes(self) -> Shapes:
        logger.info(f"Loading proto shapes from {self.path}")

        with open(self.path, "rb") as f:
            proto_shapes = pickle.load(f)

        logger.info(f"Total number of proto shapes: {len(proto_shapes)}")

        recoloring_function = lbind(recolor, 5)
        generator_expr = (
            recoloring_function(proto_shape) for proto_shape in proto_shapes
        )
        bar = tqdm(generator_expr, desc="Recoloring proto_shapes")
        shapes = frozenset(bar)

        return shapes
