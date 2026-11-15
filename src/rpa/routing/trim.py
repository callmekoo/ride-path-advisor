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


def cut_small_loops(
    arcs: np.ndarray,
    arc_tail: np.ndarray,
    arc_head: np.ndarray,
    edge_len: np.ndarray,
    max_m: float,
    keep: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Drop closed sub-walks no longer than `max_m` that pass none of the `keep` nodes.

    A walk that must not turn straight back can reach a point off its line, circle a
    block and come back to a node it already passed. Such a loop starts and ends at
    one node, so cutting it leaves a connected route. Returning to the start node with
    the last arc closes the ring and is not a loop.
    """
    if len(arcs) == 0:
        return np.array([], dtype=np.uint32)
    kept: list[int] = []
    nodes = [int(arc_tail[arcs[0]])]  # nodes[k] is where the route stands after kept[:k]
    metres = [0.0]  # metres[k] is the length of kept[:k]
    seen: dict[int, list[int]] = {nodes[0]: [0]}  # node -> its indices in `nodes`, latest last
    last = len(arcs) - 1
    for i, arc in enumerate(arcs.tolist()):
        head = int(arc_head[arc])
        total = metres[-1] + float(edge_len[arc >> 1])
        visits = seen.get(head)
        closes_ring = i == last and head == nodes[0]
        if visits and not closes_ring:
            at = visits[-1]
            if total - metres[at] <= max_m and keep.isdisjoint(nodes[at + 1 :]):
                # Forget the nodes the cut removes, so they cannot trigger a later cut.
                for gone in nodes[at + 1 :]:
                    seen[gone].pop()
                del kept[at:], nodes[at + 1 :], metres[at + 1 :]
                continue
        kept.append(arc)
        nodes.append(head)
        metres.append(total)
        seen.setdefault(head, []).append(len(nodes) - 1)
    return np.array(kept, dtype=np.uint32)


def tidy(
    arcs: np.ndarray,
    arc_tail: np.ndarray,
    arc_head: np.ndarray,
    edge_len: np.ndarray,
    max_m: float,
    keep: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Cut small loops and spurs until the route stops changing.

    Cutting a loop can leave an out-and-back at the seam, and trimming a spur can
    close a new small loop, so the two run in turn.
    """
    while True:
        tidied = trim_spurs(cut_small_loops(trim_spurs(arcs), arc_tail, arc_head, edge_len, max_m, keep))
        if len(tidied) == len(arcs):
            return tidied
        arcs = tidied
