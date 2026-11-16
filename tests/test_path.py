import numpy as np
import pytest
from helpers import arc_index, edge_index, node_index, with_edge

import rpa_kernel
from rpa.graph.snap import Snapper
from rpa.osm.tags import DISMOUNT
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


def walk_only_node(g, i: int, j: int):
    """A copy of the grid where every street at node (i, j) is walk-only, like a station platform."""
    import dataclasses

    at = j * 21 + i
    flags = g.edge_flags.copy()
    flags[(np.asarray(g.edge_u) == at) | (np.asarray(g.edge_v) == at)] |= DISMOUNT
    return dataclasses.replace(g, edge_flags=flags)


def test_snapper_with_rideable_mask_skips_walk_only_nodes(grid):
    walled = walk_only_node(grid, 10, 10)
    rideable = np.ones(walled.n_nodes, dtype=bool)
    rideable[10 * 21 + 10] = False
    point = tuple(grid.node_lonlat[10 * 21 + 10])

    assert Snapper(walled).nearest_node(*point) == (10 * 21 + 10, pytest.approx(0, abs=0.5))
    node, distance = Snapper(walled, rideable).nearest_node(*point)
    assert node != 10 * 21 + 10 and distance == pytest.approx(500, abs=5)
    near = Snapper(walled, rideable).nodes_within(*point, 600)
    assert sorted(near.tolist()) == sorted([10 * 21 + 9, 10 * 21 + 11, 9 * 21 + 10, 11 * 21 + 10])


def test_a_rideable_mask_with_no_node_marked_falls_back_to_all_nodes(grid):
    point = tuple(grid.node_lonlat[10 * 21 + 10])

    node, distance = Snapper(grid, np.zeros(grid.n_nodes, dtype=bool)).nearest_node(*point)

    assert node == 10 * 21 + 10 and distance == pytest.approx(0, abs=0.5)


def test_engine_snaps_only_to_rideable_nodes(grid):
    from rpa.routing.engine import Engine

    walled = walk_only_node(grid, 10, 10)
    engine = Engine(walled)
    node, _ = engine.snapper.nearest_node(*grid.node_lonlat[10 * 21 + 10])
    assert node != 10 * 21 + 10
    assert 10 * 21 + 10 not in engine.snapper.nodes_within(*grid.node_lonlat[10 * 21 + 10], 1000)


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


def walked(g, *edges: tuple[int, int]):
    """The micro graph with DISMOUNT on the edges between the given OSM node pairs."""
    for u, v in edges:
        g = with_edge(g, edge_index(g, u, v), edge_flags=DISMOUNT)
    return g


def test_no_walked_edges_means_no_spans(micro):
    assert build_path(micro, arcs_along(micro, 7, 4, 3, 1)).walk_spans == []
    assert build_path(micro, np.array([], dtype=np.uint32)).walk_spans == []


def test_a_walked_edge_in_the_middle_is_one_span_over_its_coordinates(micro):
    g = walked(micro, (1, 3))
    path = build_path(g, arcs_along(g, 6, 3, 1, 3, 4))
    # 6->3 has 3 points, 3->1 has 3, 1->3 has 3, 3->4 has 2: the walked stretches are 3->1 and 1->3.
    assert path.walk_spans == [(2, 6)]
    first, last = path.walk_spans[0]
    np.testing.assert_allclose(path.coords[first], [60.002, 56.800])
    np.testing.assert_allclose(path.coords[last], [60.002, 56.800])


def test_span_points_at_the_edge_also_when_it_is_ridden_backwards(micro):
    g = walked(micro, (3, 4))
    path = build_path(g, arcs_along(g, 7, 4, 3, 1))  # 4->3 runs against the stored direction
    ((first, last),) = path.walk_spans
    edge = g.edge_geometry(edge_index(g, 3, 4))[::-1]
    np.testing.assert_allclose(path.coords[first : last + 1], edge)
    assert 0 < first and last < len(path.coords) - 1


def test_neighbouring_walked_edges_merge_into_one_span(micro):
    g = walked(micro, (1, 3), (3, 4))
    path = build_path(g, arcs_along(g, 6, 3, 1, 3, 4, 7))
    # Edges: 6-3 (ridden), 3-1, 1-3, 3-4 (walked), 4-7 (ridden).
    ((first, last),) = path.walk_spans
    expected = np.concatenate([
        g.edge_geometry(edge_index(g, 1, 3))[::-1],
        g.edge_geometry(edge_index(g, 1, 3))[1:],
        g.edge_geometry(edge_index(g, 3, 4))[1:],
    ])
    np.testing.assert_allclose(path.coords[first : last + 1], expected)


def test_a_walked_first_edge_starts_at_index_zero(micro):
    g = walked(micro, (3, 6))
    path = build_path(g, arcs_along(g, 6, 3, 1))
    assert path.walk_spans == [(0, len(g.edge_geometry(edge_index(g, 3, 6))) - 1)]


def test_a_walked_last_edge_ends_at_the_last_coordinate(micro):
    g = walked(micro, (3, 6))
    path = build_path(g, arcs_along(g, 1, 3, 6))
    ((first, last),) = path.walk_spans
    assert last == len(path.coords) - 1 and first == len(g.edge_geometry(edge_index(g, 1, 3))) - 1
