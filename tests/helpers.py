"""Shared helpers for tests that work on the micro map (see test_build.py for a sketch)."""

from pathlib import Path

import numpy as np

from rpa.graph.store import GraphData

MICRO = Path(__file__).parent / "fixtures" / "micro.opl"


def node_index(g: GraphData, osm_id: int) -> int:
    return g.node_osm_id.tolist().index(osm_id)


def edge_index(g: GraphData, osm_u: int, osm_v: int) -> int:
    """Index of the edge stored as osm_u -> osm_v (in OSM way direction)."""
    ids = g.node_osm_id
    matches = np.flatnonzero((ids[g.edge_u] == osm_u) & (ids[g.edge_v] == osm_v))
    assert len(matches) == 1, f"expected exactly one edge {osm_u}->{osm_v}, found {len(matches)}"
    return int(matches[0])


def arc_index(g: GraphData, osm_from: int, osm_to: int) -> int:
    """Arc for travelling osm_from -> osm_to, whichever way the edge is stored."""
    ids = g.node_osm_id
    forward = np.flatnonzero((ids[g.edge_u] == osm_from) & (ids[g.edge_v] == osm_to))
    backward = np.flatnonzero((ids[g.edge_u] == osm_to) & (ids[g.edge_v] == osm_from))
    assert len(forward) + len(backward) == 1, f"ambiguous or missing edge {osm_from}-{osm_to}"
    return 2 * int(forward[0]) if len(forward) else 2 * int(backward[0]) + 1


def grid_graph(nx: int = 21, ny: int = 21, spacing_m: float = 500.0) -> GraphData:
    """A square street grid of plain two-way residential streets.

    Node (i, j) has index j * nx + i; i grows eastwards, j northwards.
    """
    from rpa.osm.tags import BACK_OK, FWD_OK, Highway, Surface

    lon0, lat0 = 60.0, 56.8
    dlat = spacing_m / 111_195.0
    dlon = spacing_m / (111_195.0 * np.cos(np.radians(lat0)))
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    node_lonlat = np.stack([lon0 + i.ravel() * dlon, lat0 + j.ravel() * dlat], axis=1)

    index = (j * nx + i).astype(np.uint32)
    edge_u = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
    edge_v = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
    n_edges = len(edge_u)

    a, b = np.radians(node_lonlat[edge_u]), np.radians(node_lonlat[edge_v])
    h = np.sin((b[:, 1] - a[:, 1]) / 2) ** 2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
    length = 2 * 6_371_008.8 * np.arcsin(np.sqrt(h))

    geom_xy = np.round(np.stack([node_lonlat[edge_u], node_lonlat[edge_v]], axis=1).reshape(-1, 2) * 1e7)
    graph = GraphData(
        node_lonlat=node_lonlat,
        node_osm_id=np.arange(1, nx * ny + 1, dtype=np.int64),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_len_m=length.astype(np.float32),
        edge_way_id=np.arange(1, n_edges + 1, dtype=np.int64),
        edge_way_pos=np.tile(np.array([0, 1], dtype=np.uint16), (n_edges, 1)),
        edge_highway=np.full(n_edges, Highway.RESIDENTIAL, dtype=np.uint8),
        edge_surface=np.full(n_edges, Surface.UNKNOWN, dtype=np.uint8),
        edge_flags=np.full(n_edges, FWD_OK | BACK_OK, dtype=np.uint16),
        geom_off=(2 * np.arange(n_edges + 1)).astype(np.uint32),
        geom_xy=geom_xy.astype(np.int32),
    )
    graph.validate()
    return graph
