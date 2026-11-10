"""Remove spurs: places where a route rides an edge and immediately rides it back."""

import numpy as np


def trim_spurs(arcs: np.ndarray) -> np.ndarray:
    """Drop every "arc, then its reverse" pair, repeatedly, so nested spurs unwind fully.

    Via points are the usual cause: the cheapest way to touch a point off the
    natural line of the route is to go there and come straight back.
    """
    kept: list[int] = []
    for arc in arcs.tolist():
        if kept and kept[-1] == arc ^ 1:
            kept.pop()
        else:
            kept.append(arc)
    return np.array(kept, dtype=np.uint32)
