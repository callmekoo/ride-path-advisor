"""Route through a sequence of nodes without retracing what was already ridden."""

from itertools import pairwise

import numpy as np

from rpa.routing.engine import Engine

# Riding an edge a second time, in either direction, costs this many times
# more. A penalty, not a ban: the only bridge over a river must stay usable.
USED_EDGE_FACTOR = 20.0


def route_through(
    engine: Engine, w: np.ndarray, nodes: list[int], may_turn_back_at: frozenset[int] = frozenset()
) -> np.ndarray | None:
    """Arcs of a walk visiting `nodes` in order, or None if some leg has no route.

    Works on a private copy of `w`. After each leg its edges become expensive
    in both directions, and turning straight back at the end of a leg is
    forbidden, so the walk tends to form a loop instead of an out-and-back.
    At the nodes in `may_turn_back_at` turning back is allowed: a place at the
    end of a road can only be visited that way.
    """
    w = w.copy()
    legs = []
    last_arc: int | None = None
    for a, b in pairwise(nodes):
        if last_arc is not None and a not in may_turn_back_at:
            w[last_arc ^ 1] = np.inf
        found = engine.kernel.route(a, b, w)
        if found is None:
            return None
        arcs, _ = found
        if len(arcs):
            edges = np.unique(arcs >> 1)
            w[2 * edges] *= USED_EDGE_FACTOR
            w[2 * edges + 1] *= USED_EDGE_FACTOR
            last_arc = int(arcs[-1])
        legs.append(arcs)
    return np.concatenate(legs) if legs else np.empty(0, dtype=np.uint32)
