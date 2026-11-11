"""Route generation on a synthetic 10 km x 10 km street grid (500 m blocks)."""

import numpy as np
import pytest

from rpa.routing.cost import Prefs
from rpa.routing.diversity import core_edges, overlap
from rpa.routing.engine import Engine
from rpa.routing.generate import generate
from rpa.routing.spec import AvoidArea, LengthKm, Point, RouteSpec


def node(i: int, j: int) -> int:
    return j * 21 + i


def point(grid, i: int, j: int) -> Point:
    lon, lat = grid.node_lonlat[node(i, j)]
    return Point(lon=float(lon), lat=float(lat))


def loop_spec(grid, i: int, j: int, km: float, n: int = 3) -> RouteSpec:
    return RouteSpec(start=point(grid, i, j), shape="loop", length_km=LengthKm.around(km), n_candidates=n)


def assert_sound(engine: Engine, arcs: np.ndarray, start: int, end: int) -> None:
    """A rideable walk from `start` to `end` with no immediate turn-backs."""
    assert len(arcs) > 0
    assert engine.arc_tail[arcs[0]] == start
    assert engine.arc_head[arcs[-1]] == end
    assert (engine.arc_head[arcs[:-1]] == engine.arc_tail[arcs[1:]]).all()
    assert not (arcs[1:] == (arcs[:-1] ^ 1)).any()
    assert np.isfinite(engine.weights(Prefs())[arcs]).all()


@pytest.mark.parametrize("km", [8.0, 16.0, 24.0])
def test_loops_return_to_the_start_at_the_requested_length(grid, grid_engine, km):
    candidates = generate(grid_engine, loop_spec(grid, 10, 10, km))
    assert len(candidates) >= 2
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))
        assert km * 0.9 <= candidate.path.length_m / 1000 <= km * 1.1
        assert candidate.features.repeat_share <= 0.05


def test_loop_candidates_differ_from_each_other(grid, grid_engine):
    candidates = generate(grid_engine, loop_spec(grid, 10, 10, 16.0))
    assert len(candidates) == 3
    cores = [core_edges(c.path.arcs, grid_engine.length, 2000.0) for c in candidates]
    for i in range(len(cores)):
        for j in range(i + 1, len(cores)):
            assert overlap(cores[i], cores[j], grid_engine.length) <= 0.4


def test_candidates_come_best_first(grid, grid_engine):
    scores = [c.score for c in generate(grid_engine, loop_spec(grid, 10, 10, 16.0))]
    assert scores == sorted(scores, reverse=True)


def test_generation_is_deterministic(grid, grid_engine):
    first = generate(grid_engine, loop_spec(grid, 10, 10, 16.0))
    second = generate(grid_engine, loop_spec(grid, 10, 10, 16.0))
    assert [c.path.arcs.tolist() for c in first] == [c.path.arcs.tolist() for c in second]


def test_loop_from_the_corner_of_the_map_still_works(grid, grid_engine):
    # Most directions lead off the map; the generator must find the ones that do not.
    candidates = generate(grid_engine, loop_spec(grid, 0, 0, 12.0))
    assert len(candidates) >= 1
    assert_sound(grid_engine, candidates[0].path.arcs, node(0, 0), node(0, 0))


def test_requested_count_is_respected(grid, grid_engine):
    assert len(generate(grid_engine, loop_spec(grid, 10, 10, 16.0, n=1))) == 1


def test_impossible_loop_gives_no_candidates(micro):
    engine = Engine(micro)
    spec = RouteSpec(start=Point(lon=60.0, lat=56.8), shape="loop", length_km=LengthKm.around(50))
    assert generate(engine, spec) == []


def p2p_spec(grid, length: LengthKm | None) -> RouteSpec:
    return RouteSpec(start=point(grid, 2, 10), end=point(grid, 18, 10), shape="p2p", length_km=length)


def test_p2p_without_length_starts_with_the_direct_route(grid, grid_engine):
    candidates = generate(grid_engine, p2p_spec(grid, None))
    assert len(candidates) >= 2
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(2, 10), node(18, 10))
    assert min(c.path.length_m for c in candidates) == pytest.approx(8000, rel=0.01)


def test_p2p_with_a_longer_target_takes_a_detour_of_that_length(grid, grid_engine):
    candidates = generate(grid_engine, p2p_spec(grid, LengthKm.around(14)))
    assert len(candidates) >= 2
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(2, 10), node(18, 10))
        assert 12.6 <= candidate.path.length_m / 1000 <= 15.4
        assert candidate.note is None


def test_p2p_target_just_above_the_direct_length_uses_plain_alternatives(grid, grid_engine):
    candidates = generate(grid_engine, p2p_spec(grid, LengthKm.around(8.5)))
    assert len(candidates) >= 1
    for candidate in candidates:
        assert 7.65 <= candidate.path.length_m / 1000 <= 9.35


def test_p2p_target_shorter_than_possible_returns_the_shortest_with_a_note(grid, grid_engine):
    candidates = generate(grid_engine, p2p_spec(grid, LengthKm.around(5)))
    assert len(candidates) >= 1
    assert candidates[0].path.length_m == pytest.approx(8000, rel=0.01)
    assert "8" in candidates[0].note
    assert "короче" in candidates[0].note


def test_p2p_between_disconnected_points_gives_no_candidates(micro):
    engine = Engine(micro)
    # Out of node 7 there is only a one-way street pointing at it.
    spec = RouteSpec(start=Point(lon=60.004, lat=56.800), end=Point(lon=60.000, lat=56.800), shape="p2p")
    assert generate(engine, spec) == []


@pytest.fixture(scope="module")
def lake_engine(grid):
    """The grid with a lake by its north-east corner: nodes (18..20, 18..20) are at the water."""
    engine = Engine(grid)
    shore = np.array([[node(i, j), 0] for i in range(18, 21) for j in range(18, 21)], dtype=np.uint32)
    engine.set_water(shore, [{"name": "Дальнее", "kind": "lake", "area_m2": 50_000}])
    return engine


def test_loop_to_water_reaches_the_lake(grid, lake_engine):
    spec = loop_spec(grid, 10, 10, 24.0).model_copy(update={"via_category": "water"})
    candidates = generate(lake_engine, spec)
    assert len(candidates) >= 1
    for candidate in candidates:
        assert_sound(lake_engine, candidate.path.arcs, node(10, 10), node(10, 10))
        assert lake_engine.at_water[lake_engine.arc_head[candidate.path.arcs]].any()
        assert candidate.via_label == "озеро Дальнее"
        assert 21.6 <= candidate.path.length_m / 1000 <= 26.4


def test_loop_to_water_out_of_reach_gives_nothing(grid, lake_engine):
    # A 6 km loop from the far corner cannot get to a lake 13 km away.
    spec = loop_spec(grid, 0, 0, 6.0).model_copy(update={"via_category": "water"})
    assert generate(lake_engine, spec) == []


def test_water_asked_for_but_unknown_gives_nothing(grid, grid_engine):
    spec = loop_spec(grid, 10, 10, 24.0).model_copy(update={"via_category": "water"})
    assert generate(grid_engine, spec) == []


def test_p2p_via_water_detours_to_the_lake(grid, lake_engine):
    # Through the lake the trip is 16 to 20 km, against 8 km direct.
    spec = p2p_spec(grid, LengthKm.around(18)).model_copy(update={"via_category": "water"})
    candidates = generate(lake_engine, spec)
    assert len(candidates) >= 1
    for candidate in candidates:
        assert_sound(lake_engine, candidate.path.arcs, node(2, 10), node(18, 10))
        assert lake_engine.at_water[lake_engine.arc_head[candidate.path.arcs]].any()


# --- named via places, avoided areas, headings -------------------------------


def nodes_on(engine: Engine, candidate) -> set[int]:
    arcs = candidate.path.arcs
    return {int(engine.arc_tail[arcs[0]]), *engine.arc_head[arcs].tolist()}


def named(grid, i: int, j: int, label: str) -> Point:
    return point(grid, i, j).model_copy(update={"label": label})


def test_loop_through_a_named_place_passes_it_and_fits_the_length(grid, grid_engine):
    spec = loop_spec(grid, 10, 10, 16).model_copy(update={"via": named(grid, 14, 12, "Сысерть")})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 2
    for candidate in candidates:
        assert node(14, 12) in nodes_on(grid_engine, candidate)
        assert 14.4 <= candidate.path.length_m / 1000 <= 17.6
        assert candidate.via_label == "Сысерть" and candidate.note is None
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))


def test_place_too_far_for_the_wanted_loop_is_still_reached_with_a_note(grid, grid_engine):
    spec = loop_spec(grid, 10, 10, 6).model_copy(update={"via": named(grid, 19, 10, "Дальнее")})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert node(19, 10) in nodes_on(grid_engine, candidate)
        # 4.5 km out and back by the next street over, so that no road is ridden twice.
        assert candidate.path.length_m / 1000 == pytest.approx(10.0, abs=0.1)
        assert "Дальнее" in candidate.note and "10 км" in candidate.note


def test_point_to_point_through_a_place_off_the_direct_line(grid, grid_engine):
    spec = p2p_spec(grid, None).model_copy(update={"via": named(grid, 10, 16, "Гора")})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert node(10, 16) in nodes_on(grid_engine, candidate)
        assert_sound(grid_engine, candidate.path.arcs, node(2, 10), node(18, 10))
    assert candidates[0].path.length_m / 1000 >= 14.0  # 8 km east plus 3 km north and back


def test_point_to_point_through_a_place_with_a_length(grid, grid_engine):
    spec = p2p_spec(grid, LengthKm.around(20)).model_copy(update={"via": named(grid, 10, 16, "Гора")})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert node(10, 16) in nodes_on(grid_engine, candidate)
        assert 18.0 <= candidate.path.length_m / 1000 <= 22.0


def test_avoided_area_is_ridden_around(grid, grid_engine):
    centre = point(grid, 10, 10)
    spec = p2p_spec(grid, None).model_copy(
        update={"avoid": [AvoidArea(point=centre, radius_m=1200)], "n_candidates": 1}
    )
    (candidate,) = generate(grid_engine, spec)

    lonlat = np.asarray(grid.node_lonlat)[sorted(nodes_on(grid_engine, candidate))]
    east_km = (lonlat[:, 0] - centre.lon) * 111.195 * np.cos(np.radians(centre.lat))
    north_km = (lonlat[:, 1] - centre.lat) * 111.195
    assert np.hypot(east_km, north_km).min() > 1.2
    assert candidate.path.length_m / 1000 > 8.0


def test_heading_pulls_every_loop_one_way(grid, grid_engine):
    spec = loop_spec(grid, 10, 10, 8).model_copy(update={"bearing_deg": 90.0})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert candidate.path.coords[:, 0].mean() > spec.start.lon
        assert candidate.path.coords[:, 0].min() >= spec.start.lon - 0.009  # at most a block westwards


def test_a_route_passes_a_named_place_or_a_kind_of_place_but_not_both(grid):
    with pytest.raises(ValueError, match="either"):
        RouteSpec(
            start=point(grid, 10, 10),
            shape="loop",
            length_km=LengthKm.around(10),
            via=point(grid, 12, 12),
            via_category="water",
        )


def with_peninsula(grid):
    """The grid plus a dead-end road from node (14, 12): a fork A with two stubs B and C beyond it."""
    from dataclasses import replace

    base = np.asarray(grid.node_lonlat)
    corner = base[node(14, 12)]
    a, b, c = corner + [0.002, 0.001], corner + [0.004, 0.001], corner + [0.002, 0.003]
    first = grid.n_nodes
    ends = [(node(14, 12), first), (first, first + 1), (first, first + 2)]
    lonlat = np.vstack([base, a, b, c])
    extra = len(ends)

    def more(values: np.ndarray, added) -> np.ndarray:
        return np.concatenate([np.asarray(values), np.asarray(added, dtype=np.asarray(values).dtype)])

    geometry = np.round(np.concatenate([lonlat[[u, v]] for u, v in ends]) * 1e7)
    graph = replace(
        grid,
        node_lonlat=lonlat,
        node_osm_id=more(grid.node_osm_id, [9001, 9002, 9003]),
        edge_u=more(grid.edge_u, [u for u, _ in ends]),
        edge_v=more(grid.edge_v, [v for _, v in ends]),
        edge_len_m=more(grid.edge_len_m, [160.0, 125.0, 225.0]),
        edge_way_id=more(grid.edge_way_id, [9001, 9002, 9003]),
        edge_way_pos=np.vstack([grid.edge_way_pos, np.tile(grid.edge_way_pos[:1], (extra, 1))]),
        edge_highway=more(grid.edge_highway, [grid.edge_highway[0]] * extra),
        edge_surface=more(grid.edge_surface, [grid.edge_surface[0]] * extra),
        edge_flags=more(grid.edge_flags, [grid.edge_flags[0]] * extra),
        geom_off=more(grid.geom_off, [int(grid.geom_off[-1]) + 2 * (i + 1) for i in range(extra)]),
        geom_xy=np.vstack([grid.geom_xy, geometry.astype(grid.geom_xy.dtype)]),
    )
    graph.validate()
    return graph, first


def test_a_place_on_a_dead_end_road_is_ridden_to_and_back_from(grid):
    graph, fork = with_peninsula(grid)
    engine = Engine(graph)
    lon, lat = graph.node_lonlat[fork].tolist()
    spec = loop_spec(grid, 10, 10, 16).model_copy(update={"via": Point(lon=lon, lat=lat, label="Мыс")})
    candidates = generate(engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert fork in nodes_on(engine, candidate) and candidate.via_label == "Мыс"
        assert 14.4 <= candidate.path.length_m / 1000 <= 17.6
        assert candidate.features.repeat_share < 0.05  # only the 160 m to the fork and back
