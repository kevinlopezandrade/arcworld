from arcworld.dsl.arc_types import Object
from arcworld.dsl.functional import color, outbox, paint, recolor
from arcworld.schematas.oop.expansion.grid import LinesGrid


def no_op(dot: Object, line: Object, grid: LinesGrid):
    pass


def paint_over(dot: Object, line: Object, grid: LinesGrid):
    grid.grid = paint(grid.grid, dot)


def paint_bg(dot: Object, line: Object, grid: LinesGrid):
    grid.grid = paint(grid.grid, recolor(grid.bg_color, dot))


def paint_outbox(dot: Object, line: Object, grid: LinesGrid):
    grid.grid = paint(grid.grid, recolor(color(dot), outbox(dot)))


POLICIES = {
    "no_op": no_op,
    "paint_over": paint_over,
    "paint_bg": paint_bg,
    "paint_outbox": paint_outbox,
}
