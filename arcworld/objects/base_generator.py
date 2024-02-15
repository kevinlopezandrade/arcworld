import logging
import pickle
from typing import Protocol

from arcdsl.arc_types import Objects
from arcdsl.dsl import lbind, recolor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ObjectGenerator(Protocol):
    def generate_random_objects(self) -> Objects:
        ...


class ObjectGeneratorFromFile:
    def __init__(self, path: str) -> None:
        self.path = path

    def generate_random_objects(self) -> Objects:
        logger.info(f"Loading shapes from {self.path}")
        with open(self.path, "rb") as f:
            shapes = pickle.load(f)

        logger.info(f"Total number of shapes: {len(shapes)}")
        return shapes


class ObjectGeneratorFromProtoFile:
    def __init__(self, path: str) -> None:
        self.path = path

    def generate_random_objects(self) -> Objects:
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
