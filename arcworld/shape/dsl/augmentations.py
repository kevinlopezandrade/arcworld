from typing import Callable, Dict, cast

from arcdsl.dsl import (
    Coordinates,
    box,
    cmirror,
    combine,
    compose,
    connect,
    dmirror,
    fork,
    hmirror,
    identity,
    llcorner,
    lrcorner,
    normalize,
    outbox,
    ulcorner,
    urcorner,
    vmirror,
)

ShapeAugementator = Callable[[Coordinates], Coordinates]


def augm_scheme(f):
    return compose(normalize, fork(combine, identity, f))


AUGMENTATION_OPTIONS: Dict[str, ShapeAugementator] = cast(
    Dict[str, ShapeAugementator],
    {
        "Identity": normalize,
        "AddVerticallyMirrored": augm_scheme(vmirror),
        "AddHorizontallyMirrored": augm_scheme(hmirror),
        "AddHVMirrored": augm_scheme(
            fork(combine, augm_scheme(hmirror), augm_scheme(vmirror))
        ),
        "AddDiagonallyMirrored": augm_scheme(dmirror),
        "AddCounterdiagonallyMirrored": augm_scheme(cmirror),
        "AddDCMirrored": augm_scheme(
            fork(combine, augm_scheme(dmirror), augm_scheme(cmirror))
        ),
        "AddMirrored": augm_scheme(
            fork(
                combine,
                fork(combine, augm_scheme(hmirror), augm_scheme(vmirror)),
                fork(combine, augm_scheme(dmirror), augm_scheme(cmirror)),
            )
        ),
        "AddBox": augm_scheme(box),
        "AddOutBox": augm_scheme(outbox),
        "AddDiagonalLine": augm_scheme(fork(connect, ulcorner, lrcorner)),
        "AddCounterdiagonalLine": augm_scheme(fork(connect, llcorner, urcorner)),
        "AddCross": augm_scheme(
            fork(
                combine,
                fork(connect, ulcorner, lrcorner),
                fork(connect, llcorner, urcorner),
            )
        ),
    },
)
