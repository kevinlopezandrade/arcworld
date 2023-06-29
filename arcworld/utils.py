import matplotlib.pyplot as plt

from arcworld.dsl.arc_types import Coordinates, Grid, Shape
from arcworld.dsl.functional import canvas, height, paint, recolor, width
from arcworld.internal.constants import COLORMAP, NORM


def plot_shape(shape: Shape):
    h = height(shape) * 2
    w = width(shape) * 2
    grid = canvas(0, (h, w))
    grid = paint(grid, shape)

    fig, axe = plt.subplots()
    axe.imshow(grid, cmap=COLORMAP, norm=NORM)
    axe.grid(True, which="both", color="lightgrey", linewidth=0.5)
    axe.set_xticks(x - 0.5 for x in range(width(shape)))
    axe.set_yticks(x - 0.5 for x in range(height(shape)))
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    plt.show()


def plot_shapes(*shapes):
    h = max(height(shape) for shape in shapes) * 2
    w = max(height(shape) for shape in shapes) * 2

    fig, axes = plt.subplots(1, len(shapes))

    for i, shape in enumerate(shapes):
        grid = canvas(0, (h, w))
        grid = paint(grid, shape)
        axes[i].imshow(grid, cmap=COLORMAP, norm=NORM)
        axes[i].grid(True, which="both", color="lightgrey", linewidth=0.5)
        axes[i].set_xticks([x - 0.5 for x in range(w)])
        axes[i].set_yticks([x - 0.5 for x in range(h)])
        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])

    plt.show()


def plot_proto_shape(proto_shape: Coordinates):
    shape = recolor(7, proto_shape)
    plot_shape(shape)


def plot_grid(grid: Grid):
    fig, axe = plt.subplots()

    axe.imshow(grid, cmap=COLORMAP, norm=NORM)
    axe.grid(True, which="both", color="lightgrey", linewidth=0.5)

    axe.set_xticks(x - 0.5 for x in range(width(grid)))
    axe.set_yticks(x - 0.5 for x in range(height(grid)))
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    plt.show()
