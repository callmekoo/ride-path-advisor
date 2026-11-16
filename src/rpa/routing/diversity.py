"""Pick candidates that are good and also clearly different from each other."""

import numpy as np

from rpa.routing.score import Candidate


def core_edges(arcs: np.ndarray, edge_len: np.ndarray, skip_m: float) -> np.ndarray:
    """Distinct edges of a route, leaving out its first and last `skip_m` metres.

    Every loop from the same start shares the way out of town; counting that
    stretch would make all loops look alike.
    """
    edges = arcs >> 1
    metres = edge_len[edges].astype(np.float64)
    end = np.cumsum(metres)
    total = float(end[-1]) if len(end) else 0.0
    skip = min(skip_m, total / 6)
    inside = (end - metres >= skip) & (end <= total - skip)
    return np.unique(edges[inside])


def _compared_edges(candidate: Candidate, edge_len: np.ndarray, skip_m: float) -> np.ndarray:
    """What two candidates are compared on: the ring when the candidate has one, else its core edges.

    Rides around one object from a far start share the whole way there and back; only the
    rings tell them apart.
    """
    if candidate.core_arcs is not None:
        return np.unique(candidate.core_arcs >> 1)
    return core_edges(candidate.path.arcs, edge_len, skip_m)


def overlap(a: np.ndarray, b: np.ndarray, edge_len: np.ndarray) -> float:
    """Length-weighted Jaccard similarity of two edge sets: 0 = disjoint, 1 = identical."""
    union = float(edge_len[np.union1d(a, b)].sum())
    return float(edge_len[np.intersect1d(a, b)].sum()) / union if union else 0.0


def select_diverse(
    candidates: list[Candidate],
    n: int,
    edge_len: np.ndarray,
    max_overlap: float = 0.4,
    skip_m: float = 2000.0,
    chosen: list[Candidate] | None = None,
) -> list[Candidate]:
    """Best-first selection of up to `n` candidates no two of which overlap more than `max_overlap`.

    `chosen` are kept ahead of the rest, e.g. what a stricter selection already took.
    """
    chosen = list(chosen or [])
    chosen_edges = [_compared_edges(c, edge_len, skip_m) for c in chosen]
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(chosen) >= n:
            break
        if any(candidate is other for other in chosen):
            continue
        edges = _compared_edges(candidate, edge_len, skip_m)
        if all(overlap(edges, other, edge_len) <= max_overlap for other in chosen_edges):
            chosen.append(candidate)
            chosen_edges.append(edges)
    return chosen


def select_by_size(
    groups: list[list[Candidate]], edge_len: np.ndarray, max_overlap: float = 0.4, skip_m: float = 2000.0
) -> list[Candidate]:
    """One candidate per group of a size, in the groups' order (short to long).

    Each is the best of its group that overlaps the ones already taken by at
    most `max_overlap`; when none does, the best of the group all the same,
    since a different size is a different ride. An empty group gives nothing.
    """
    chosen: list[Candidate] = []
    chosen_edges: list[np.ndarray] = []
    for group in groups:
        ranked = sorted(group, key=lambda c: c.score, reverse=True)
        if not ranked:
            continue
        edges = [core_edges(c.path.arcs, edge_len, skip_m) for c in ranked]
        distinct = (
            i
            for i, mine in enumerate(edges)
            if all(overlap(mine, other, edge_len) <= max_overlap for other in chosen_edges)
        )
        best = next(distinct, 0)
        chosen.append(ranked[best])
        chosen_edges.append(edges[best])
    return chosen
