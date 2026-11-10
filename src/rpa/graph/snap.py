"""Find the graph node nearest to a point on the map."""

import numpy as np
from scipy.spatial import cKDTree

from rpa.graph.store import GraphData

EARTH_RADIUS_M = 6_371_008.8


def _unit_xyz(lonlat_deg: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lonlat_deg[..., 0]), np.radians(lonlat_deg[..., 1])
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class Snapper:
    """Nearest-node lookup. Builds a k-d tree over the nodes once."""

    def __init__(self, g: GraphData) -> None:
        self._tree = cKDTree(_unit_xyz(np.asarray(g.node_lonlat)))

    def nearest_node(self, lon: float, lat: float) -> tuple[int, float]:
        """Return (node index, great-circle distance to it in metres)."""
        chord, node = self._tree.query(_unit_xyz(np.array([lon, lat])))
        return int(node), float(2 * EARTH_RADIUS_M * np.arcsin(min(chord / 2, 1.0)))
