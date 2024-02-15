from arcdsl.dsl import shift, subtract, ulcorner

from arcworld.grid.dsl.grid_dsl import GridDSL
from arcworld.grid.oop.grid_bruteforce import BinaryRelation, BSTGridBruteForce
from arcworld.objects.dsl.generator import ShapeGeneratorDSL
from arcworld.objects.dsl.utils import proto_vbar, switch_shapes
from arcworld.utils import plot_grids

generator = ShapeGeneratorDSL(max_obj_dimension=2)
random_shapes = generator.generate_random_shapes()
input_grid = BSTGridBruteForce(20, 20, mode=BinaryRelation.LeftOf)

for shape in random_shapes:
    placed_shaped = input_grid.place_shape_random(shape)
    x = ulcorner(placed_shaped)[1]

    # Add fictional border.
    input_grid.occupied = input_grid.occupied | proto_vbar(x, input_grid.height)


output_grid = GridDSL(
    input_grid.height, input_grid.width, input_grid.bg_color, input_grid.margin
)

# Pairwise transposition
shapes = input_grid.shapes
for i in range(len(shapes) - 1):
    pre = shapes[i]
    pos = shapes[i + 1]
    output_grid.grid = switch_shapes(output_grid.grid, pre, pos)

    shapes[i + 1] = shift(pre, subtract(ulcorner(pos), ulcorner(pre)))


plot_grids(input_grid.grid_np, output_grid.grid_np)
