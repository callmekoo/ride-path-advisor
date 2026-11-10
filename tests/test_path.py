import numpy as np
import pytest
from helpers import arc_index, node_index

import rpa_kernel
from rpa.graph.snap import Snapper
from rpa.routing.cost import Prefs, arc_weights
from rpa.routing.path import build_path


def arcs_along(g, *osm_nodes: int) -> np.ndarray:
    pairs = zip(osm_nodes[:-1], osm_nodes[1:])
    return np.array([arc_index(g, a, b) for a, b in pairs], dtype=np.uint32)


def test_geometry_is_stitched_without_duplicate_joints(micro):
    path = build_path(micro, arcs_along(micro, 1, 3, 6))
    np.testing.assert_allclose(
        path.coords,
        [[60.000, 56.800], [60.001, 56.800], [60.002, 56.800], [60.002, 56.801], [60.002, 56.802]],
    )


def test_backward_arcs_reverse_the_edge_geometry(micro):
    path = build_path(micro, arcs_along(micro, 6, 3, 1))
    np.testing.assert_allclose(
        path.coords,
        [[60.002, 56.802], [60.002, 56.801], [60.002, 56.800], [60.001, 56.800], [60.000, 56.800]],
    )


def test_length_and_breakdowns(micro):
    path = build_path(micro, arcs_along(micro, 1, 3, 6))
    assert path.length_m == pytest.approx(2 * 60.89 + 2 * 111.19, abs=1.0)
    assert set(path.km_by_highway) == {"residential", "track"}
    assert path.km_by_highway["track"] == pytest.approx(0.2224, abs=0.001)
    assert sum(path.km_by_highway.values()) == pytest.approx(path.length_m / 1000)
    assert set(path.km_by_surface) == {"unknown", "unpaved"}
    assert sum(path.km_by_surface.values()) == pytest.approx(path.length_m / 1000)


def test_empty_path(micro):
    path = build_path(micro, np.array([], dtype=np.uint32))
    assert path.coords.shape == (0, 2)
    assert path.length_m == 0.0
    assert path.km_by_highway == {}


def test_snapper_finds_the_nearest_node(micro):
    snapper = Snapper(micro)
    node, distance = snapper.nearest_node(60.0021, 56.8001)  # ~6 m east, ~11 m north of node 3
    assert micro.node_osm_id[node] == 3
    assert distance == pytest.approx(12.7, abs=1.0)


def test_snapper_reports_large_distances(micro):
    node, distance = Snapper(micro).nearest_node(60.1, 56.8)
    assert micro.node_osm_id[node] == 7
    assert distance == pytest.approx(96 * 60.89, rel=0.01)


def test_end_to_end_route_on_the_micro_map(micro):
    kernel = rpa_kernel.Graph(micro.edge_u, micro.edge_v, micro.node_lonlat)
    w = arc_weights(micro, Prefs())
    arcs, _ = kernel.route(node_index(micro, 1), node_index(micro, 14), w)
    path = build_path(micro, arcs)
    np.testing.assert_allclose(path.coords[0], [60.000, 56.800])
    np.testing.assert_allclose(path.coords[-1], [60.004, 56.803])
    assert path.length_m > 500


def test_oneway_makes_the_dead_end_a_trap(micro):
    # Node 7 is reached by a one-way street and its only other way is a winter road.
    kernel = rpa_kernel.Graph(micro.edge_u, micro.edge_v, micro.node_lonlat)
    w = arc_weights(micro, Prefs())
    assert kernel.route(node_index(micro, 4), node_index(micro, 7), w) is not None
    assert kernel.route(node_index(micro, 7), node_index(micro, 4), w) is None
