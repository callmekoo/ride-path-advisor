"""Graph assembly, checked on a hand-written micro map.

tests/fixtures/micro.opl, roughly (ids are OSM node ids, * marks graph nodes):

    13 ---- 12* --- 14*      w16: closed path 6-11-12-13-6, w17: footway 12-14
     |       |
     6* --- 11
     |
     5                       w11: track 3-5-6
     |
    1* - 2 - 3* - 4* -> 7* .. 8     w10: residential 1-2-3-4, w12: oneway 4->7,
                                    w13: winter road 7-8 (must vanish)

    9 - 10                   w15: an island, dropped as a separate component
"""

from pathlib import Path

import numpy as np
import pytest

from rpa.graph.store import GraphData
from rpa.osm.build import RawWays, assemble, build_graph, read_ways
from rpa.osm.tags import BACK_OK, FWD_OK, SOFT_RESTRICTED, Highway, Surface

MICRO = Path(__file__).parent / "fixtures" / "micro.opl"


@pytest.fixture(scope="module")
def micro() -> GraphData:
    return build_graph(MICRO)


def edge_index(g: GraphData, osm_u: int, osm_v: int) -> int:
    ids = g.node_osm_id
    matches = np.flatnonzero((ids[g.edge_u] == osm_u) & (ids[g.edge_v] == osm_v))
    assert len(matches) == 1, f"expected exactly one edge {osm_u}->{osm_v}, found {len(matches)}"
    return int(matches[0])


def test_graph_is_consistent(micro):
    micro.validate()


def test_nodes_are_the_split_points_of_the_main_component(micro):
    assert micro.node_osm_id.tolist() == [1, 3, 4, 6, 7, 12, 14]


def test_edges_are_way_pieces_between_split_points(micro):
    ids = micro.node_osm_id
    edges = sorted(zip(ids[micro.edge_u].tolist(), ids[micro.edge_v].tolist(), micro.edge_way_id.tolist()))
    assert edges == [
        (1, 3, 10),
        (3, 4, 10),
        (3, 6, 11),
        (4, 7, 12),
        (6, 12, 16),
        (12, 6, 16),
        (12, 14, 17),
    ]


def test_winter_road_and_island_are_absent(micro):
    assert not set(micro.node_osm_id.tolist()) & {8, 9, 10}


def test_node_coordinates(micro):
    node = micro.node_osm_id.tolist().index(3)
    np.testing.assert_allclose(micro.node_lonlat[node], [60.002, 56.8])


def test_geometry_keeps_intermediate_points_in_travel_order(micro):
    np.testing.assert_allclose(
        micro.edge_geometry(edge_index(micro, 1, 3)),
        [[60.000, 56.8], [60.001, 56.8], [60.002, 56.8]],
    )
    np.testing.assert_allclose(
        micro.edge_geometry(edge_index(micro, 12, 6)),
        [[60.003, 56.803], [60.002, 56.803], [60.002, 56.802]],
    )


def test_edge_lengths_in_metres(micro):
    # 0.001 degree is 111.19 m along a meridian and 60.89 m along the 56.8 parallel.
    assert micro.edge_len_m[edge_index(micro, 1, 3)] == pytest.approx(2 * 60.89, abs=0.5)
    assert micro.edge_len_m[edge_index(micro, 3, 6)] == pytest.approx(2 * 111.19, abs=0.5)


def test_position_within_way(micro):
    assert micro.edge_way_pos[edge_index(micro, 1, 3)].tolist() == [0, 2]
    assert micro.edge_way_pos[edge_index(micro, 3, 4)].tolist() == [2, 3]
    assert micro.edge_way_pos[edge_index(micro, 6, 12)].tolist() == [0, 2]
    assert micro.edge_way_pos[edge_index(micro, 12, 6)].tolist() == [2, 4]


def test_attributes_come_from_the_way_tags(micro):
    track = edge_index(micro, 3, 6)
    assert micro.edge_highway[track] == Highway.TRACK
    assert micro.edge_surface[track] == Surface.UNPAVED

    oneway = edge_index(micro, 4, 7)
    assert micro.edge_highway[oneway] == Highway.TERTIARY
    assert micro.edge_flags[oneway] & (FWD_OK | BACK_OK) == FWD_OK

    footway = edge_index(micro, 12, 14)
    assert micro.edge_flags[footway] & SOFT_RESTRICTED
    assert micro.edge_flags[footway] & (FWD_OK | BACK_OK) == FWD_OK | BACK_OK


def raw_ways(ways: dict[int, list[int]], coords: dict[int, tuple[float, float]]) -> RawWays:
    refs = [ref for nodes in ways.values() for ref in nodes]
    return RawWays(
        refs=np.array(refs, dtype=np.int64),
        xy=np.array([[round(coords[r][0] * 1e7), round(coords[r][1] * 1e7)] for r in refs], dtype=np.int32),
        way_off=np.cumsum([0] + [len(nodes) for nodes in ways.values()]).astype(np.uint32),
        way_id=np.array(list(ways), dtype=np.int64),
        way_highway=np.full(len(ways), Highway.RESIDENTIAL, dtype=np.uint8),
        way_surface=np.full(len(ways), Surface.UNKNOWN, dtype=np.uint8),
        way_flags=np.full(len(ways), FWD_OK | BACK_OK, dtype=np.uint16),
    )


COORDS = {
    1: (60.000, 56.800),
    2: (60.001, 56.800),
    3: (60.001, 56.801),
    4: (60.000, 56.801),
    5: (60.002, 56.800),
}


def test_closed_way_touching_nothing_else_becomes_no_edge():
    # Way 20 is a ring hanging off node 1; it has no second split point, so it
    # would be a self-loop, which is useless for routing.
    g = assemble(raw_ways({20: [1, 2, 3, 4, 1], 21: [1, 5]}, COORDS))
    assert g.node_osm_id.tolist() == [1, 5]
    assert g.edge_way_id.tolist() == [21]


def test_way_crossing_itself_is_split_at_the_crossing():
    # Node 2 is visited twice by the same way: 1-2-3-4-2-5.
    coords = {**COORDS, 4: (60.002, 56.801)}
    g = assemble(raw_ways({30: [1, 2, 3, 4, 2, 5]}, coords))
    ids = g.node_osm_id
    edges = sorted(zip(ids[g.edge_u].tolist(), ids[g.edge_v].tolist()))
    # The 2-3-4-2 lobe is a self-loop and goes away.
    assert edges == [(1, 2), (2, 5)]


def test_duplicate_points_get_a_minimal_positive_length():
    coords = {1: (60.0, 56.8), 2: (60.0, 56.8), 3: (60.001, 56.8)}
    g = assemble(raw_ways({40: [1, 2], 41: [2, 3]}, coords))
    assert (g.edge_len_m > 0).all()


def test_no_routable_ways_is_an_error(tmp_path):
    path = tmp_path / "empty.opl"
    path.write_text("n1 x60.0 y56.8\nn2 x60.001 y56.8\nw1 Tbuilding=yes Nn1,n2,n1\n")
    with pytest.raises(ValueError, match="no routable ways"):
        build_graph(path)


def test_lone_ring_is_an_error_not_a_crash():
    with pytest.raises(ValueError, match="no routable ways"):
        assemble(raw_ways({50: [1, 2, 3, 4, 1]}, COORDS))


def test_nodes_missing_from_the_extract_are_skipped(tmp_path):
    # Ways cut by the border of an extract refer to nodes that are not in the file.
    path = tmp_path / "cut.opl"
    path.write_text(
        "n1 x60.000 y56.8\nn2 x60.001 y56.8\nn3 x60.002 y56.8\n"
        "w1 Thighway=residential Nn1,n2,n99,n3\n"
        "w2 Thighway=residential Nn3,n98\n"
    )
    raw = read_ways(path)
    assert raw.way_id.tolist() == [1]
    assert raw.refs.tolist() == [1, 2, 3]
