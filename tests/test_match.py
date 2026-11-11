import numpy as np
import osmium
import pytest
from helpers import MICRO, edge_index

from rpa.match.matcher import Matched, build_tiles, decode_polyline6, load_matched, match_tracks, save_matched
from rpa.match.project import project
from rpa.tracks.store import make_trackset

NODE = {1: (60.000, 56.800), 2: (60.001, 56.800), 3: (60.002, 56.800), 4: (60.003, 56.800), 6: (60.002, 56.802)}


def matched(*stretches: tuple[int, int, tuple, tuple]) -> Matched:
    """Stretches given as (track, way id, start lon/lat, end lon/lat)."""
    ends = [[*start, *end] for _, _, start, end in stretches]
    return Matched(
        way_id=np.array([way for _, way, _, _ in stretches], dtype=np.int64),
        ends=np.round(np.array(ends).reshape(-1, 4) * 1e7).astype(np.int32),
        track=np.array([track for track, _, _, _ in stretches], dtype=np.uint32),
        n_tracks=1 + max((track for track, _, _, _ in stretches), default=0),
    )


def rides_by_edge(micro, m: Matched) -> dict[tuple[int, int], int]:
    rides = project(micro, m)
    ids = micro.node_osm_id
    return {(int(ids[u]), int(ids[v])): int(r) for u, v, r in zip(micro.edge_u, micro.edge_v, rides) if r}


def test_decode_polyline6():
    # The classic example string; at precision 6 its values are a tenth of the precision-5 ones.
    np.testing.assert_allclose(
        decode_polyline6("_p~iF~ps|U_ulLnnqC_mqNvxq`@"), [[-12.02, 3.85], [-12.095, 4.07], [-12.6453, 4.3252]]
    )


def test_stretch_along_one_edge(micro):
    assert rides_by_edge(micro, matched((0, 10, NODE[1], NODE[3]))) == {(1, 3): 1}


def test_direction_of_travel_does_not_matter(micro):
    assert rides_by_edge(micro, matched((0, 10, NODE[3], NODE[1]))) == {(1, 3): 1}


def test_stretch_spanning_two_edges_of_a_way(micro):
    assert rides_by_edge(micro, matched((0, 10, NODE[1], NODE[4]))) == {(1, 3): 1, (3, 4): 1}


def test_a_track_counts_once_per_edge_and_tracks_add_up(micro):
    m = matched((0, 10, NODE[1], NODE[3]), (0, 10, NODE[3], NODE[1]), (1, 10, NODE[1], NODE[3]), (2, 11, NODE[3], NODE[6]))
    assert rides_by_edge(micro, m) == {(1, 3): 2, (3, 6): 1}


def test_ends_are_snapped_to_the_nearest_vertex_of_the_way(micro):
    near_1, near_3 = (60.00002, 56.80003), (60.00195, 56.79998)
    assert rides_by_edge(micro, matched((0, 10, near_1, near_3))) == {(1, 3): 1}


def test_an_edge_counts_only_if_most_of_it_was_ridden(micro):
    # Way 16 starts 6 - 11 - 12: its first edge is 61 m to node 11, then 111 m to node 12.
    node_11, node_12 = (60.003, 56.802), (60.003, 56.803)
    assert rides_by_edge(micro, matched((0, 16, node_11, node_12))) == {(6, 12): 1}
    assert rides_by_edge(micro, matched((0, 16, NODE[6], node_11))) == {}


def test_ways_missing_from_the_graph_are_ignored(micro):
    m = matched((0, 13, (60.004, 56.8), (60.005, 56.8)), (0, 999, NODE[1], NODE[3]))
    assert rides_by_edge(micro, m) == {}
    assert project(micro, matched()).sum() == 0


def test_matched_round_trip(tmp_path):
    m = matched((0, 10, NODE[1], NODE[3]), (3, 11, NODE[3], NODE[6]))
    save_matched(m, tmp_path / "match" / "matched.npz")
    loaded = load_matched(tmp_path / "match" / "matched.npz")
    np.testing.assert_array_equal(loaded.way_id, m.way_id)
    np.testing.assert_array_equal(loaded.ends, m.ends)
    np.testing.assert_array_equal(loaded.track, m.track)
    assert loaded.n_tracks == 4


@pytest.fixture(scope="module")
def valhalla_config(tmp_path_factory):
    """Valhalla tiles of the micro map."""
    folder = tmp_path_factory.mktemp("valhalla")
    pbf = folder / "micro.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for obj in osmium.FileProcessor(str(MICRO)):
            writer.add(obj)
    return build_tiles(pbf, folder)


def test_a_noisy_track_is_matched_to_the_ways_it_followed(micro, valhalla_config):
    # East along way 10 from node 1 to node 3, then north along way 11 to node 6.
    east = np.stack([np.linspace(60.0, 60.002, 14), np.full(14, 56.8)], axis=1)
    north = np.stack([np.full(24, 60.002), np.linspace(56.8, 56.802, 24)], axis=1)
    ride = np.concatenate([east, north[1:]])
    ride += np.random.default_rng(1).normal(0, 3e-5, ride.shape)  # a few metres of GPS noise
    tracks = make_trackset([{"id": "t"}], [[ride]])

    result, report = match_tracks(tracks, valhalla_config, jobs=1, log=lambda _: None)
    assert report.failed == 0
    assert report.matched_km == pytest.approx(0.344, abs=0.03)
    # GPS noise at the far end may touch the next way; projection discards such scraps.
    assert {10, 11} <= set(result.way_id.tolist())

    rides = project(micro, result)
    assert rides[edge_index(micro, 1, 3)] == 1
    assert rides[edge_index(micro, 3, 6)] == 1
    assert rides.sum() == 2
