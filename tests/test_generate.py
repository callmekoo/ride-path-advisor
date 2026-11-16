"""Route generation on a synthetic 10 km x 10 km street grid (500 m blocks)."""

import re

import numpy as np
import pytest

from helpers import grid_with_hole, rect_outline, with_walked_cut, without_edges

from rpa.routing.cost import Prefs
from rpa.routing.diversity import core_edges, overlap
from rpa.routing.engine import Engine
from rpa.routing.generate import APPENDIX_SHARE, MIN_APPENDIX_M, _passes_in_order, generate
from rpa.routing.ring import encloses
from rpa.routing.spec import Around, AvoidArea, LengthKm, Point, RouteSpec


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


@pytest.mark.parametrize("km", [8.0, 16.0])
def test_loops_have_no_small_closed_sub_walks(grid, grid_engine, km):
    limit = max(MIN_APPENDIX_M, APPENDIX_SHARE * km * 1000)
    for candidate in generate(grid_engine, loop_spec(grid, 10, 10, km)):
        arcs = candidate.path.arcs
        last_at: dict[int, float] = {int(grid_engine.arc_tail[arcs[0]]): 0.0}
        ridden = 0.0
        for i, arc in enumerate(arcs.tolist()):
            ridden += float(grid_engine.length[arc >> 1])
            head = int(grid_engine.arc_head[arc])
            if head in last_at and i < len(arcs) - 1:
                assert ridden - last_at[head] > limit
            last_at[head] = ridden


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
    spec = loop_spec(grid, 10, 10, 16).model_copy(update={"via": [named(grid, 14, 12, "Сысерть")]})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 2
    for candidate in candidates:
        assert node(14, 12) in nodes_on(grid_engine, candidate)
        assert 14.4 <= candidate.path.length_m / 1000 <= 17.6
        assert candidate.via_label == "через: Сысерть" and candidate.note is None
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))


def test_place_too_far_for_the_wanted_loop_is_still_reached_with_a_note(grid, grid_engine):
    spec = loop_spec(grid, 10, 10, 6).model_copy(update={"via": [named(grid, 19, 10, "Дальнее")]})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    # 4.5 km out and back by the next street over, so that no road is ridden twice.
    assert min(c.path.length_m for c in candidates) / 1000 == pytest.approx(10.0, abs=0.1)
    for candidate in candidates:
        assert node(19, 10) in nodes_on(grid_engine, candidate)
        assert "Дальнее" in candidate.note and "10 км" in candidate.note


def test_point_to_point_through_a_place_off_the_direct_line(grid, grid_engine):
    spec = p2p_spec(grid, None).model_copy(update={"via": [named(grid, 10, 16, "Гора")]})
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert node(10, 16) in nodes_on(grid_engine, candidate)
        assert_sound(grid_engine, candidate.path.arcs, node(2, 10), node(18, 10))
    assert candidates[0].path.length_m / 1000 >= 13.9  # 8 km east plus 3 km north and back, grid edges a bit short


def test_point_to_point_through_a_place_with_a_length(grid, grid_engine):
    spec = p2p_spec(grid, LengthKm.around(20)).model_copy(update={"via": [named(grid, 10, 16, "Гора")]})
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
            via=[point(grid, 12, 12)],
            via_category="water",
        )


SIZES_KM = (6.0, 10.0, 16.0)


@pytest.fixture(scope="module")
def sized_engine(grid):
    engine = Engine(grid)
    engine.set_loop_sizes(SIZES_KM)
    return engine


def assert_one_per_size(candidates, sizes) -> None:
    """One candidate per size, short to long, each within 15 % of its size."""
    assert len(candidates) == len(sizes)
    for candidate, size in zip(candidates, sizes, strict=True):
        assert size * 0.85 <= candidate.path.length_m / 1000 <= size * 1.15
        assert candidate.note is None


def test_a_loop_without_a_length_or_a_place_comes_in_three_sizes(grid, sized_engine):
    candidates = generate(sized_engine, RouteSpec(start=point(grid, 10, 10), shape="loop"))
    assert_one_per_size(candidates, SIZES_KM)
    for candidate in candidates:
        assert_sound(sized_engine, candidate.path.arcs, node(10, 10), node(10, 10))


def test_fewer_wanted_loops_without_a_length_are_the_shortest_sizes(grid, sized_engine):
    candidates = generate(sized_engine, RouteSpec(start=point(grid, 10, 10), shape="loop", n_candidates=2))
    assert_one_per_size(candidates, SIZES_KM[:2])


# Every size of these reaches the lake from (14, 14); smaller loops from there fall short of it.
WATER_SIZES_KM = (8.0, 12.0, 16.0)
WATER_START = (14, 14)


def test_a_loop_to_water_without_a_length_comes_in_three_sizes(grid):
    engine = Engine(grid)
    shore = np.array([[node(i, j), 0] for i in range(18, 21) for j in range(18, 21)], dtype=np.uint32)
    engine.set_water(shore, [{"name": "Дальнее", "kind": "lake", "area_m2": 50_000}])
    engine.set_loop_sizes(WATER_SIZES_KM)
    spec = RouteSpec(start=point(grid, WATER_START[0], WATER_START[1]), shape="loop", via_category="water")
    candidates = generate(engine, spec)
    assert_one_per_size(candidates, WATER_SIZES_KM)
    for candidate in candidates:
        assert engine.at_water[engine.arc_head[candidate.path.arcs]].any()
        assert candidate.via_label == "озеро Дальнее"


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
    spec = loop_spec(grid, 10, 10, 16).model_copy(update={"via": [Point(lon=lon, lat=lat, label="Мыс")]})
    candidates = generate(engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert fork in nodes_on(engine, candidate) and candidate.via_label == "через: Мыс"
        assert 14.4 <= candidate.path.length_m / 1000 <= 17.6
        assert candidate.features.repeat_share < 0.05  # only the 160 m to the fork and back


def test_p2p_without_length_prefers_the_cheapest_route_over_long_detours(grid):
    # Every crossing of the cut at i = 10 is walked except two rows at the southern edge.
    engine = Engine(with_walked_cut(grid, 10))
    spec = RouteSpec(start=point(grid, 10, 10), end=point(grid, 11, 10), shape="p2p", n_candidates=3)
    candidates = generate(engine, spec)

    assert candidates
    assert candidates[0].features.walk_m > 0
    assert candidates[0].path.length_m == min(c.path.length_m for c in candidates)


# --- a chain of places -------------------------------------------------------


def first_visits(engine: Engine, candidate, places: list[int]) -> list[int]:
    """Where along the route each of `places` is first reached (the start counts as 0)."""
    arcs = candidate.path.arcs
    order = [int(engine.arc_tail[arcs[0]]), *engine.arc_head[arcs].tolist()]
    return [order.index(place) for place in places]


SYSERT, GORA = (14, 12), (8, 15)


def chain_spec(grid, *places, km: float | None = None, shape: str = "loop") -> RouteSpec:
    via = [named(grid, i, j, label) if label else point(grid, i, j) for i, j, label in places]
    if shape == "loop":
        spec = RouteSpec(start=point(grid, 10, 10), shape="loop", via=via)
    else:
        spec = RouteSpec(start=point(grid, 2, 10), end=point(grid, 18, 10), shape="p2p", via=via)
    return spec.model_copy(update={"length_km": LengthKm.around(km) if km else None})


def test_passes_in_order_reads_the_start_and_the_order(grid_engine):
    from helpers import arcs_via

    arcs = arcs_via(grid_engine, node(2, 2), node(6, 2), node(6, 6))
    assert _passes_in_order(grid_engine, arcs, (node(2, 2), node(6, 2), node(6, 6)))
    assert _passes_in_order(grid_engine, arcs, (node(4, 2), node(6, 4)))
    assert not _passes_in_order(grid_engine, arcs, (node(6, 4), node(4, 2)))
    assert not _passes_in_order(grid_engine, arcs, (node(4, 2), node(9, 9)))


def test_point_to_point_passes_two_places_in_the_given_order(grid, grid_engine):
    spec = chain_spec(grid, (6, 14, "А"), (14, 6, "Б"), shape="p2p")
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(2, 10), node(18, 10))
        a, b = first_visits(grid_engine, candidate, [node(6, 14), node(14, 6)])
        assert a < b
        assert candidate.via_label == "через: А, Б"


def test_the_reverse_order_gives_another_route(grid, grid_engine):
    forward = generate(grid_engine, chain_spec(grid, (6, 14, "А"), (14, 6, "Б"), shape="p2p"))
    backward = generate(grid_engine, chain_spec(grid, (14, 6, "Б"), (6, 14, "А"), shape="p2p"))

    assert backward and backward[0].path.arcs.tolist() != forward[0].path.arcs.tolist()
    for candidate in backward:
        b, a = first_visits(grid_engine, candidate, [node(14, 6), node(6, 14)])
        assert b < a
        assert candidate.via_label == "через: Б, А"


def test_a_loop_through_two_places_needs_no_length(grid, grid_engine):
    spec = chain_spec(grid, (*SYSERT, "Сысерть"), (*GORA, "Гора"))
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 2
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))
        first, second = first_visits(grid_engine, candidate, [node(*SYSERT), node(*GORA)])
        assert first < second
        assert candidate.via_label == "через: Сысерть, Гора" and candidate.note is None


def test_only_labelled_places_name_the_route(grid, grid_engine):
    spec = chain_spec(grid, (*SYSERT, None), (*GORA, "Гора"))
    assert {c.via_label for c in generate(grid_engine, spec)} == {"через: Гора"}


def test_a_loop_through_two_places_is_lengthened_to_the_wanted_length(grid, grid_engine):
    # Straight through both places the loop is about 11 km.
    spec = chain_spec(grid, (*SYSERT, "Сысерть"), (*GORA, "Гора"), km=20)
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 2
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))
        first, second = first_visits(grid_engine, candidate, [node(*SYSERT), node(*GORA)])
        assert first < second
        assert 18.0 <= candidate.path.length_m / 1000 <= 22.0
        assert candidate.note is None


def test_two_places_too_far_for_the_wanted_length_get_a_note(grid, grid_engine):
    spec = chain_spec(grid, (*SYSERT, "Сысерть"), (*GORA, "Гора"), km=6)
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert candidate.note.startswith("через «Сысерть», «Гора» короче ")
        assert candidate.note.endswith(" км не получится")
        first, second = first_visits(grid_engine, candidate, [node(*SYSERT), node(*GORA)])
        assert first < second


def test_unnamed_places_too_far_are_called_these_places(grid, grid_engine):
    spec = chain_spec(grid, (*SYSERT, None), (*GORA, None), km=6)
    candidates = generate(grid_engine, spec)
    assert candidates and candidates[0].note.startswith("через эти места короче ")
    assert candidates[0].via_label is None


def test_a_place_at_the_start_is_dropped_from_the_chain(grid, grid_engine):
    spec = chain_spec(grid, (2, 10, "Дом"), (10, 16, "Гора"), shape="p2p")
    candidates = generate(grid_engine, spec)
    alone = generate(grid_engine, chain_spec(grid, (10, 16, "Гора"), shape="p2p"))

    assert candidates
    assert [c.path.arcs.tolist() for c in candidates] == [c.path.arcs.tolist() for c in alone]
    assert candidates[0].via_label == "через: Дом, Гора"


def test_a_loop_may_pass_its_start_between_two_places(grid, grid_engine):
    # A figure of eight: out to the first place, back through the start, out to the second.
    spec = chain_spec(grid, (*SYSERT, "Сысерть"), (10, 10, "Дом"), (6, 7, "Лес"))
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        assert_sound(grid_engine, candidate.path.arcs, node(10, 10), node(10, 10))
        arcs = candidate.path.arcs
        heads = grid_engine.arc_head[arcs].tolist()
        at_place = heads.index(node(*SYSERT))
        home = heads.index(node(10, 10), at_place)
        assert node(6, 7) in heads[home:]


def test_a_far_place_gives_several_routes(grid, grid_engine):
    spec = chain_spec(grid, (19, 19, "Дальнее"))
    candidates = generate(grid_engine, spec)

    assert len(candidates) >= 2
    for candidate in candidates:
        assert node(19, 19) in nodes_on(grid_engine, candidate)


def test_a_place_on_a_dead_end_road_in_a_chain_is_ridden_to_and_back_from(grid):
    graph, fork = with_peninsula(grid)
    engine = Engine(graph)
    lon, lat = graph.node_lonlat[fork].tolist()
    via = [Point(lon=lon, lat=lat, label="Мыс"), named(grid, *GORA, "Гора")]
    spec = RouteSpec(start=point(grid, 10, 10), shape="loop", via=via)
    candidates = generate(engine, spec)

    assert len(candidates) >= 1
    for candidate in candidates:
        cape, gora = first_visits(engine, candidate, [fork, node(*GORA)])
        assert cape < gora
        assert candidate.via_label == "через: Мыс, Гора"


def test_size_tolerances_keep_neighbouring_bands_apart():
    from rpa.routing.generate import LOOP_SIZE_TOLERANCE, _size_tolerances

    assert _size_tolerances((30.0, 50.0, 80.0)) == [LOOP_SIZE_TOLERANCE] * 3
    sizes = (20.0, 25.0, 30.0)
    bands = [(km * (1 - t), km * (1 + t)) for km, t in zip(sizes, _size_tolerances(sizes), strict=True)]
    for (_, high), (low, _) in zip(bands, bands[1:], strict=False):
        assert high <= low + 1e-9
    assert all(t <= LOOP_SIZE_TOLERANCE for t in _size_tolerances(sizes))


def test_close_sizes_still_come_back_short_to_long(grid):
    engine = Engine(grid)
    engine.set_loop_sizes((6.0, 7.2, 8.6))

    lengths = [c.path.length_m for c in generate(engine, RouteSpec(start=point(grid, 10, 10), shape="loop"))]

    assert lengths == sorted(lengths)


def test_the_chain_title_drops_a_trailing_kind_and_skips_unlabelled_points(grid):
    from rpa.routing.generate import chain_title

    via = [
        named(grid, 4, 4, "Двуреченск (село)"),
        point(grid, 5, 5),
        named(grid, 6, 6, "Маминское (село) (бывш.)"),
        named(grid, 7, 7, "(село) Старое"),
        named(grid, 8, 8, "Ревда (город), 4 км на восток"),
    ]
    assert chain_title(via) == "через: Двуреченск, Маминское (село), (село) Старое, Ревда (город), 4 км на восток"
    assert chain_title([point(grid, 5, 5)]) is None and chain_title([]) is None


# --- around an object --------------------------------------------------------

# A lake filling the blocks between the streets i = 8..12, j = 8..12: the streets around it make an 8 km ring.
HOLE = (8, 12, 8, 12)


@pytest.fixture(scope="module")
def hole_engine(grid):
    return Engine(grid_with_hole(grid, *HOLE))


def around_spec(
    grid, i: int, j: int, box=HOLE, label: str | None = "Тестовое (озеро)", km: float | None = None, n: int = 3
) -> RouteSpec:
    outline = rect_outline(grid, *box)
    lon, lat = np.mean(outline, axis=0).tolist()
    around = Around(point=Point(lon=lon, lat=lat, label=label), radius_m=1000.0, outline=outline)
    length = LengthKm.around(km) if km is not None else None
    return RouteSpec(start=point(grid, i, j), shape="loop", length_km=length, around=around, n_candidates=n)


def assert_around(engine: Engine, candidates, spec: RouteSpec, i: int, j: int) -> None:
    """At least one route, each a sound loop from (i, j) that goes once around the object."""
    assert len(candidates) >= 1
    for candidate in candidates:
        assert_sound(engine, candidate.path.arcs, node(i, j), node(i, j))
        assert encloses(candidate.path.coords, spec.around)
        assert candidate.core_arcs is not None


def test_a_loop_around_a_lake_from_its_shore_hugs_it(grid, hole_engine):
    spec = around_spec(grid, 8, 10)
    candidates = generate(hole_engine, spec)
    assert_around(hole_engine, candidates, spec, 8, 10)
    assert candidates[0].path.length_m == pytest.approx(8000, rel=0.01)
    assert candidates[0].via_label == "вокруг: Тестовое"
    assert candidates[0].note is None


def test_a_loop_around_a_lake_from_below_it_rides_there_and_back(grid, hole_engine):
    spec = around_spec(grid, 10, 5)
    candidates = generate(hole_engine, spec)
    assert_around(hole_engine, candidates, spec, 10, 5)
    # 1.5 km there, 8 km around, 1.5 km back, give or take a block.
    assert candidates[0].path.length_m / 1000 <= 12.0
    core_m = float(hole_engine.length[candidates[0].core_arcs >> 1].sum())
    assert core_m < candidates[0].path.length_m - 2000


def test_a_loop_around_a_lake_from_the_far_corner_of_the_map(grid, hole_engine):
    spec = around_spec(grid, 0, 0)
    assert_around(hole_engine, generate(hole_engine, spec), spec, 0, 0)


def test_a_loop_around_a_long_narrow_lake(grid):
    box = (9, 11, 4, 16)  # 1 km by 6 km
    engine = Engine(grid_with_hole(grid, *box))
    spec = around_spec(grid, 9, 10, box=box)
    candidates = generate(engine, spec)
    assert_around(engine, candidates, spec, 9, 10)
    assert candidates[0].path.length_m == pytest.approx(14000, rel=0.05)


def test_a_loop_around_an_object_without_an_outline_circles_its_point(grid, hole_engine):
    lon, lat = grid.node_lonlat[node(10, 10)].tolist()
    around = Around(point=Point(lon=lon, lat=lat, label="Гора"), radius_m=750.0)
    spec = RouteSpec(start=point(grid, 8, 10), shape="loop", around=around)
    candidates = generate(hole_engine, spec)
    assert_around(hole_engine, candidates, spec, 8, 10)
    assert candidates[0].via_label == "вокруг: Гора"


def test_an_unnamed_object_is_called_an_object(grid, hole_engine):
    candidates = generate(hole_engine, around_spec(grid, 8, 10, label=None))
    assert candidates and all(c.via_label == "вокруг: объект" for c in candidates)


def test_an_object_with_no_road_on_one_side_cannot_be_ridden_around(grid):
    box = (15, 22, 6, 14)  # reaches past the eastern edge of the map
    engine = Engine(grid_with_hole(grid, *box))
    assert generate(engine, around_spec(grid, 15, 10, box=box)) == []


def test_a_loop_around_an_object_is_deterministic(grid, hole_engine):
    first = generate(hole_engine, around_spec(grid, 10, 5))
    second = generate(hole_engine, around_spec(grid, 10, 5))
    assert [c.path.arcs.tolist() for c in first] == [c.path.arcs.tolist() for c in second]


def test_a_loop_around_a_lake_is_widened_to_the_wanted_length(grid, hole_engine):
    spec = around_spec(grid, 8, 10, km=16.0)  # twice the ride along the shore
    candidates = generate(hole_engine, spec)
    assert_around(hole_engine, candidates, spec, 8, 10)
    for candidate in candidates:
        assert 14.4 <= candidate.path.length_m / 1000 <= 17.6
        assert candidate.note is None


def test_a_loop_around_a_lake_shorter_than_the_shore_gets_a_note(grid, hole_engine):
    spec = around_spec(grid, 8, 10, km=4.0)
    candidates = generate(hole_engine, spec)
    assert_around(hole_engine, candidates, spec, 8, 10)
    assert candidates[0].path.length_m == pytest.approx(8000, rel=0.01)
    assert candidates[0].note == "вокруг «Тестовое (озеро)» короче 8 км не получится"


def test_a_loop_around_a_small_object_stops_widening_at_the_ceiling(grid):
    from rpa.routing.generate import _around_ceiling_m

    engine = Engine(grid_with_hole(grid, 9, 11, 9, 11))
    lon, lat = grid.node_lonlat[node(10, 10)].tolist()
    around = Around(point=Point(lon=lon, lat=lat), radius_m=300.0)
    spec = RouteSpec(start=point(grid, 9, 10), shape="loop", length_km=LengthKm.around(40), around=around)
    candidates = generate(engine, spec)
    assert_around(engine, candidates, spec, 9, 10)
    # A ring 3 km out at most: some 26 km on the grid's streets.
    assert _around_ceiling_m(around) < 3000
    longest = max(c.path.length_m for c in candidates) / 1000
    assert 20 < longest < 30
    # The note names the longest ride found, which need not be the best one offered.
    note = re.fullmatch(r"вокруг этого объекта длиннее (\d+) км не получится", candidates[0].note)
    assert note and round(longest) <= int(note[1]) < 30


def corridor_with_hole(grid):
    """A pond at node (15, 10) whose only way in from the west is the 7 km street along row j = 10."""
    graph = grid_with_hole(grid, 14, 16, 9, 11)
    u, v = np.asarray(graph.edge_u), np.asarray(graph.edge_v)
    across = (v - u == 1) & (u % 21 < 14) & (u // 21 != 10)
    return without_edges(graph, across)


def test_a_far_start_with_one_road_there_gets_the_least_repeating_ride_with_a_note(grid):
    engine = Engine(corridor_with_hole(grid))
    spec = around_spec(grid, 0, 10, box=(14, 16, 9, 11))
    candidates = generate(engine, spec)
    assert len(candidates) == 1
    assert_around(engine, candidates, spec, 0, 10)
    assert candidates[0].features.repeat_share > 0.3
    assert candidates[0].note == "подъезд и возврат по одной дороге"
