"""Find the graph node nearest to a point on the map."""

import numpy as np
from scipy.spatial import cKDTree

from rpa.graph.store import GraphData

EARTH_RADIUS_M = 6_371_008.8


def _unit_xyz(lonlat_deg: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lonlat_deg[..., 0]), np.radians(lonlat_deg[..., 1])
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class Snapper:
    """Nearest-node lookup. Builds a k-d tree over the nodes once.

    `rideable` is an optional bool mask over the graph's nodes: only the marked nodes can be found
    (e.g. not the ones reachable on foot only). Without it, every node can be found. A mask with
    no marked node is ignored rather than leaving nothing to find.
    """

    def __init__(self, g: GraphData, rideable: np.ndarray | None = None) -> None:
        lonlat = np.asarray(g.node_lonlat)
        self._nodes: np.ndarray | None = None  # tree index -> graph node; None when they coincide
        if rideable is not None and rideable.any():
            self._nodes = np.flatnonzero(rideable)
            lonlat = lonlat[self._nodes]
        self._tree = cKDTree(_unit_xyz(lonlat))

    def nearest_node(self, lon: float, lat: float) -> tuple[int, float]:
        """Return (node index, great-circle distance to it in metres)."""
        chord, node = self._tree.query(_unit_xyz(np.array([lon, lat])))
        node = node if self._nodes is None else self._nodes[node]
        return int(node), float(2 * EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1.0)))

    def nodes_within(self, lon: float, lat: float, radius_m: float) -> np.ndarray:
        """Indices of all nodes within `radius_m` metres of the point."""
        chord = 2 * np.sin(min(radius_m / EARTH_RADIUS_M, np.pi) / 2)
        found = np.array(self._tree.query_ball_point(_unit_xyz(np.array([lon, lat])), chord), dtype=np.int64)
        return found if self._nodes is None else self._nodes[found]
