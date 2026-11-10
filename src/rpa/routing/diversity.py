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
) -> list[Candidate]:
    """Best-first selection of up to `n` candidates no two of which overlap more than `max_overlap`."""
    chosen: list[Candidate] = []
    chosen_edges: list[np.ndarray] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        edges = core_edges(candidate.path.arcs, edge_len, skip_m)
        if all(overlap(edges, other, edge_len) <= max_overlap for other in chosen_edges):
            chosen.append(candidate)
            chosen_edges.append(edges)
            if len(chosen) == n:
                break
    return chosen
