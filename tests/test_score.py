import dataclasses

import numpy as np
import pytest

from helpers import arcs_via, node, square, with_edge

from rpa.osm.tags import DISMOUNT, Highway, Surface
from rpa.routing.cost import Prefs
from rpa.routing.diversity import overlap, select_diverse
from rpa.routing.engine import Engine
from rpa.routing.path import build_path
from rpa.routing.score import (
    CROSSING_PENALTY,
    MAX_COUNTED_CROSSINGS,
    EXTRA_COST_PENALTY,
    MAX_TURN_PENALTY,
    Candidate,
    Features,
    features,
    rejection,
    score,
)
from rpa.routing.spec import LengthKm, Point, RouteSpec

START = Point(lon=60.0, lat=56.8)


def test_features_of_a_clean_square_loop(grid_engine):
    w = grid_engine.weights(Prefs())
    f = features(grid_engine, w, square(grid_engine, 2, 2, 4), LengthKm.around(8))
    assert f.length_m == pytest.approx(8000, rel=0.01)
    assert f.length_error == pytest.approx(0.0, abs=0.01)
    assert f.repeat_share == 0.0
    # Residential street with unknown surface at the default avoid_major.
    assert f.cost_ratio == pytest.approx((1 + 1.15 * 0.1) * 1.05, rel=1e-4)
    # Three corners are seen between consecutive arcs; the one at the start is not.
    assert f.turns_per_10km == pytest.approx(3 / 0.8, rel=0.01)
    assert f.major_share == 0.0
    assert f.unpaved_share == 0.0


def test_out_and_back_is_half_repeats(grid_engine):
    w = grid_engine.weights(Prefs())
    out = arcs_via(grid_engine, node(2, 2), node(10, 2))
    back = (out ^ 1)[::-1]
    f = features(grid_engine, w, np.concatenate([out, back]), None)
    assert f.repeat_share == pytest.approx(0.5)


def test_length_error_is_relative_to_the_target(grid_engine):
    w = grid_engine.weights(Prefs())
    f = features(grid_engine, w, square(grid_engine, 2, 2, 4), LengthKm.around(10))
    assert f.length_error == pytest.approx(0.2, abs=0.01)


def test_shares_by_road_class_and_surface(grid):
    highway = grid.edge_highway.copy()
    surface = grid.edge_surface.copy()
    highway[:] = Highway.PRIMARY
    surface[:] = Surface.UNPAVED
    engine = Engine(dataclasses.replace(grid, edge_highway=highway, edge_surface=surface))
    f = features(engine, engine.weights(Prefs()), square(engine, 2, 2, 4), None)
    assert f.major_share == pytest.approx(1.0)
    assert f.unpaved_share == pytest.approx(1.0)


def test_walk_m_is_the_length_of_the_walked_edges(grid):
    plain = Engine(grid)
    arcs = arcs_via(plain, node(2, 2), node(6, 2))
    assert features(plain, plain.weights(Prefs()), arcs, None).walk_m == 0.0

    edge = int(arcs[1] >> 1)
    engine = Engine(with_edge(grid, edge, edge_flags=DISMOUNT))
    f = features(engine, engine.weights(Prefs()), arcs, None)
    assert f.walk_m == pytest.approx(float(grid.edge_len_m[edge]))

    there_and_back = np.concatenate([arcs, (arcs ^ 1)[::-1]])
    assert features(engine, engine.weights(Prefs()), there_and_back, None).walk_m == pytest.approx(
        2 * float(grid.edge_len_m[edge])
    )


def test_empty_route_has_neutral_features(grid_engine):
    f = features(grid_engine, grid_engine.weights(Prefs()), np.empty(0, dtype=np.uint32), LengthKm.around(8))
    assert f.length_m == 0.0
    assert f.length_error == 1.0


def test_score_prefers_fewer_repeats_and_better_length(grid_engine):
    w = grid_engine.weights(Prefs())
    length = LengthKm.around(8)
    clean = features(grid_engine, w, square(grid_engine, 2, 2, 4), length)
    repeated = dataclasses.replace(clean, repeat_share=0.2)
    off_length = dataclasses.replace(clean, length_m=8700.0)
    assert score(clean, length) > score(repeated, length)
    assert score(clean, length) > score(off_length, length)
    assert 0 <= score(clean, length) <= 100


def loop_spec(km: float, **prefs) -> RouteSpec:
    return RouteSpec(start=START, shape="loop", length_km=LengthKm.around(km), prefs=Prefs(**prefs))


def test_rejection_reasons(grid_engine):
    w = grid_engine.weights(Prefs())
    f = features(grid_engine, w, square(grid_engine, 2, 2, 4), LengthKm.around(8))
    assert rejection(f, loop_spec(8)) is None
    assert "длина" in rejection(f, loop_spec(12))
    assert "повтор" in rejection(dataclasses.replace(f, repeat_share=0.5), loop_spec(8))
    assert "грунт" in rejection(dataclasses.replace(f, unpaved_share=0.2), loop_spec(8, surface="paved"))
    assert rejection(dataclasses.replace(f, unpaved_share=0.2), loop_spec(8)) is None
    assert "грунт" in rejection(dataclasses.replace(f, unpaved_share=0.05), loop_spec(8, surface="paved"))
    assert rejection(dataclasses.replace(f, unpaved_share=0.02), loop_spec(8, surface="paved")) is None


def candidate(engine: Engine, arcs: np.ndarray, value: float) -> Candidate:
    f = features(engine, engine.weights(Prefs()), arcs, None)
    return Candidate(path=build_path(engine.g, arcs), features=f, score=value, note=None)


def test_overlap_is_length_weighted_jaccard(grid_engine):
    a = np.unique(square(grid_engine, 2, 2, 4) >> 1)
    b = np.unique(square(grid_engine, 2, 2, 4) >> 1)
    far = np.unique(square(grid_engine, 12, 12, 4) >> 1)
    assert overlap(a, b, grid_engine.length) == pytest.approx(1.0)
    assert overlap(a, far, grid_engine.length) == 0.0
    # Squares side by side share one side of four: 4 edges of 28 in the union.
    beside = np.unique(square(grid_engine, 6, 2, 4) >> 1)
    assert overlap(a, beside, grid_engine.length) == pytest.approx(4 / 28, abs=0.01)


def test_select_diverse_drops_near_duplicates_and_keeps_the_best(grid_engine):
    best = candidate(grid_engine, square(grid_engine, 2, 2, 8), 90.0)
    twin = candidate(grid_engine, square(grid_engine, 2, 2, 8), 85.0)
    other = candidate(grid_engine, square(grid_engine, 10, 10, 8), 70.0)
    chosen = select_diverse([other, twin, best], 3, grid_engine.length)
    assert chosen == [best, other]


def test_select_diverse_respects_the_requested_count(grid_engine):
    a = candidate(grid_engine, square(grid_engine, 2, 2, 8), 90.0)
    b = candidate(grid_engine, square(grid_engine, 10, 10, 8), 70.0)
    assert select_diverse([a, b], 1, grid_engine.length) == [a]


def test_select_diverse_keeps_an_earlier_choice_first_and_adds_to_it(grid_engine):
    def ring(north: int) -> np.ndarray:
        return arcs_via(grid_engine, node(2, 2), node(10, 2), node(10, north), node(2, north), node(2, 2))

    kept = candidate(grid_engine, ring(6), 70.0)
    better = candidate(grid_engine, ring(7), 90.0)  # shares 16 of 34 edges with `kept`
    twin = candidate(grid_engine, ring(6), 80.0)
    chosen = select_diverse([twin, better, kept], 3, grid_engine.length, 0.6, skip_m=0.0, chosen=[kept])
    assert chosen == [kept, better]


def test_shared_start_and_finish_do_not_count_as_overlap(grid_engine):
    # Both loops leave and return along the same 2.5 km street, then go separate ways.
    stem_out = arcs_via(grid_engine, node(0, 10), node(5, 10))
    stem_back = (stem_out ^ 1)[::-1]
    north = np.concatenate([stem_out, square(grid_engine, 5, 10, 8), stem_back])
    south = np.concatenate([stem_out, (square(grid_engine, 5, 2, 8)[::-1] ^ 1), stem_back])
    a, b = candidate(grid_engine, north, 90.0), candidate(grid_engine, south, 80.0)
    assert select_diverse([a, b], 2, grid_engine.length, skip_m=2500.0) == [a, b]


def test_cost_m_is_the_sum_of_arc_weights(grid_engine):
    w = grid_engine.weights(Prefs())
    arcs = square(grid_engine, 2, 2, 4)
    f = features(grid_engine, w, arcs, None)
    assert f.cost_m == pytest.approx(float(w[arcs].astype(np.float64).sum()), rel=1e-6)


def _plain(length_m: float, cost_ratio: float, **kwargs) -> Features:
    return Features(length_m, 0.0, 0.0, cost_ratio, 0.0, 0.0, 0.0, **kwargs)


def test_without_a_length_dearer_routes_lose_by_their_extra_cost():
    cheap, dear = _plain(5000, 1.0), _plain(5000, 2.0)  # the dear one costs twice as much
    assert score(cheap, None, cheapest_cost_m=cheap.cost_m) == score(cheap, None)
    assert score(dear, None, cheapest_cost_m=cheap.cost_m) == pytest.approx(
        score(dear, None) - EXTRA_COST_PENALTY / 2
    )


def test_cheapest_cost_is_ignored_when_a_length_was_asked_for():
    f = _plain(5000, 2.0)
    assert score(f, LengthKm.around(5), cheapest_cost_m=1000.0) == score(f, LengthKm.around(5))


def test_zero_length_route_gets_no_extra_cost_deduction():
    f = _plain(0.0, 1.0)
    assert score(f, None, cheapest_cost_m=1000.0) == score(f, None)


def test_turn_penalty_is_capped():
    twisty = Features(500.0, 0.0, 0.0, 1.0, 10_000.0, 0.0, 0.0)
    assert score(twisty, None) == pytest.approx(100.0 - MAX_TURN_PENALTY)


def crossings_of(engine, arcs):
    return features(engine, engine.weights(Prefs()), arcs, None).crossings


def test_a_clean_square_has_no_crossings(grid_engine):
    assert crossings_of(grid_engine, square(grid_engine, 2, 2, 4)) == 0


def test_a_figure_eight_crosses_itself_once(grid_engine):
    # Two squares sharing the corner (4, 4), ridden from a start away from that corner.
    eight = np.concatenate([
        arcs_via(grid_engine, node(4, 4), node(8, 4), node(8, 8), node(4, 8), node(4, 4)),
        arcs_via(grid_engine, node(4, 4), node(0, 4), node(0, 0), node(4, 0), node(4, 4)),
    ])
    shifted = np.concatenate([eight[2:], eight[:2]])
    assert crossings_of(grid_engine, shifted) == 1


def test_a_shared_stretch_counts_once(grid_engine):
    # Out along y=2 and x=6, then back down x=6 and on along y=2: the stretch is shared, met once.
    route = arcs_via(
        grid_engine, node(2, 2), node(6, 2), node(6, 6), node(8, 6), node(8, 4), node(6, 4), node(6, 2)
    )
    route = np.concatenate([route, arcs_via(grid_engine, node(6, 2), node(2, 2))])
    assert crossings_of(grid_engine, route) == 1


def test_out_and_back_crosses_once(grid_engine):
    out = arcs_via(grid_engine, node(2, 2), node(5, 2))
    back = arcs_via(grid_engine, node(5, 2), node(2, 2))
    assert crossings_of(grid_engine, np.concatenate([out, back])) == 1


def test_crossings_lower_the_score_up_to_a_cap():
    base = Features(8000.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    clean = score(base, None)
    assert score(dataclasses.replace(base, crossings=1), None) == pytest.approx(clean - CROSSING_PENALTY)
    assert score(dataclasses.replace(base, crossings=10), None) == pytest.approx(
        clean - CROSSING_PENALTY * MAX_COUNTED_CROSSINGS
    )
