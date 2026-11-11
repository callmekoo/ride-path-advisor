"""Project map-matched rides onto the edges of a graph and count rides per edge.

A matched stretch says "this ride used OSM way W between these two points".
Both points are snapped to the nearest vertex of W in our graph, which gives a
range of vertices along the way; every edge of W covered by that range for at
least half of its length counts as ridden.

Known limitation: on a closed way (a roundabout) a stretch that passes the
closing node is attributed to the opposite arc. Such ways are short and the
effect on route preferences is negligible.
"""

import numpy as np

from rpa.graph.store import GraphData
from rpa.match.matcher import Matched
from rpa.tracks.clean import step_m

MIN_COVERED_SHARE = 0.5


def project(g: GraphData, matched: Matched) -> np.ndarray:
    """rides: u16[E], the number of distinct tracks that rode each edge."""
    n_edges = g.n_edges
    if len(matched.way_id) == 0:
        return np.zeros(n_edges, dtype=np.uint16)

    # Metres from the start of its edge to every geometry point.
    geom_xy = np.asarray(g.geom_xy)
    geom_off = np.asarray(g.geom_off).astype(np.int64)
    step = np.concatenate([[0.0], step_m(geom_xy / 1e7)])
    step[geom_off[:-1]] = 0.0
    along = np.cumsum(step)
    along -= np.repeat(along[geom_off[:-1]], np.diff(geom_off))

    # Edges grouped by way, and stretches grouped by way.
    edge_way = np.asarray(g.edge_way_id)
    edge_order = np.argsort(edge_way, kind="stable")
    ways, way_start = np.unique(edge_way[edge_order], return_index=True)
    way_end = np.r_[way_start[1:], len(edge_order)]
    way_first = np.asarray(g.edge_way_pos)[:, 0].astype(np.int64)
    way_last = np.asarray(g.edge_way_pos)[:, 1].astype(np.int64)
    edge_len = np.asarray(g.edge_len_m, dtype=np.float64)

    stretch_order = np.argsort(matched.way_id, kind="stable")
    stretch_ways, stretch_start = np.unique(matched.way_id[stretch_order], return_index=True)
    stretch_end = np.r_[stretch_start[1:], len(stretch_order)]
    slot = np.searchsorted(ways, stretch_ways)

    pair_keys = []
    for i, way in enumerate(stretch_ways):
        if slot[i] == len(ways) or ways[slot[i]] != way:
            continue  # the way is not part of the graph (not routable, or outside the main component)
        edges = edge_order[way_start[slot[i]] : way_end[slot[i]]]
        stretches = stretch_order[stretch_start[i] : stretch_end[i]]

        # Vertices of the way that the graph knows: their index along the way and position.
        first, last = way_first[edges], way_last[edges]
        counts = last - first + 1
        point = np.repeat(geom_off[edges], counts) + _ranges(counts)
        vertex = np.repeat(first, counts) + _ranges(counts)
        xy = geom_xy[point].astype(np.float64)
        scale = np.array([np.cos(np.radians(xy[0, 1] / 1e7)), 1.0])

        ends = matched.ends[stretches].astype(np.float64)
        a = vertex[_nearest(ends[:, 0:2] * scale, xy * scale)]
        b = vertex[_nearest(ends[:, 2:4] * scale, xy * scale)]
        low, high = np.minimum(a, b), np.maximum(a, b)

        # Overlap of every stretch's vertex range with every edge of the way, in metres.
        from_vertex = np.maximum(low[:, None], first[None, :])
        to_vertex = np.minimum(high[:, None], last[None, :])
        overlaps = to_vertex > from_vertex
        base = (geom_off[edges] - first)[None, :]
        covered = np.where(
            overlaps, along[np.where(overlaps, base + to_vertex, 0)] - along[np.where(overlaps, base + from_vertex, 0)], 0.0
        )
        which_stretch, which_edge = np.nonzero(covered >= MIN_COVERED_SHARE * edge_len[edges][None, :])
        track = matched.track[stretches[which_stretch]].astype(np.int64)
        pair_keys.append(track * n_edges + edges[which_edge])

    keys = np.unique(np.concatenate(pair_keys)) if pair_keys else np.empty(0, dtype=np.int64)
    rides = np.bincount(keys % n_edges, minlength=n_edges)
    return np.minimum(rides, np.iinfo(np.uint16).max).astype(np.uint16)


def _ranges(counts: np.ndarray) -> np.ndarray:
    """Concatenated arange(c) for each c in counts."""
    starts = np.cumsum(counts) - counts
    return np.arange(int(counts.sum())) - np.repeat(starts, counts)


def _nearest(queries: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the nearest of `points` for each of `queries` (both [n, 2])."""
    delta = queries[:, None, :] - points[None, :, :]
    return np.argmin((delta**2).sum(axis=2), axis=1)
