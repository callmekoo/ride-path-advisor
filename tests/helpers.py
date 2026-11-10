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
