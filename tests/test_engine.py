import dataclasses

import numpy as np
import pytest

from rpa.osm.tags import Highway
from rpa.routing.cost import Prefs
from rpa.routing.engine import Engine
from rpa.routing.geo import bearing_deg, destination


def test_destination_moves_the_right_distance_and_direction():
    lon, lat = destination(60.0, 56.8, 0.0, 1000.0)
    assert lon == pytest.approx(60.0, abs=1e-9)
    assert lat == pytest.approx(56.8 + 1000 / 111_195, abs=1e-5)
    lon, lat = destination(60.0, 56.8, 90.0, 1000.0)
    assert lon == pytest.approx(60.0 + 1000 / (111_195 * np.cos(np.radians(56.8))), abs=1e-5)
    assert lat == pytest.approx(56.8, abs=1e-5)


@pytest.mark.parametrize("bearing", [0.0, 45.0, 135.0, 250.0, 359.0])
def test_bearing_is_the_inverse_of_destination(bearing):
    lon, lat = destination(60.0, 56.8, bearing, 5000.0)
    difference = (bearing_deg(60.0, 56.8, lon, lat) - bearing + 180) % 360 - 180
    assert abs(difference) < 0.1


def test_engine_arc_endpoints(grid, grid_engine):
    e = 5
    assert grid_engine.arc_tail[2 * e] == grid.edge_u[e]
    assert grid_engine.arc_head[2 * e] == grid.edge_v[e]
    assert grid_engine.arc_tail[2 * e + 1] == grid.edge_v[e]
    assert grid_engine.arc_head[2 * e + 1] == grid.edge_u[e]


def test_engine_degree(grid_engine):
    # 21 x 21 grid: corners have 2 streets, edges 3, the interior 4.
    assert grid_engine.degree[0] == 2
    assert grid_engine.degree[1] == 3
    assert grid_engine.degree[21 + 1] == 4


def test_node_niceness_is_the_best_multiplier_leaving_the_node(grid):
    # Make every street a primary road except the ones touching node 0.
    highway = np.full(grid.n_edges, Highway.PRIMARY, dtype=np.uint8)
    touching = (grid.edge_u == 0) | (grid.edge_v == 0)
    highway[touching] = Highway.UNCLASSIFIED
    engine = Engine(dataclasses.replace(grid, edge_highway=highway))
    w = engine.weights(Prefs(avoid_major=1.0))
    nice = engine.node_niceness(w)
    # The grid has unknown surface, which costs x1.05 in the default surface mode.
    assert nice[0] == pytest.approx(1.05)
    assert nice[1] == pytest.approx(1.05)  # shares a quiet street with node 0
    assert nice[21 * 10 + 10] == pytest.approx((1 + 2.0 * 1.2) * 1.05)


def test_nodes_within_radius(grid, grid_engine):
    lon, lat = grid.node_lonlat[21 * 10 + 10]
    near = grid_engine.snapper.nodes_within(lon, lat, 600.0)
    # The centre node and its four neighbours 500 m away.
    assert sorted(near.tolist()) == sorted([220, 219, 221, 199, 241])
