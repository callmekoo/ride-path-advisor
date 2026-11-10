import dataclasses

import numpy as np
import pytest

from rpa.osm.tags import Highway, Surface
from rpa.routing.cost import Prefs
from rpa.routing.diversity import overlap, select_diverse
from rpa.routing.engine import Engine
from rpa.routing.path import build_path
from rpa.routing.score import Candidate, features, rejection, score
from rpa.routing.spec import LengthKm, Point, RouteSpec

START = Point(lon=60.0, lat=56.8)


def node(i: int, j: int) -> int:
    return j * 21 + i


def arcs_via(engine: Engine, *nodes: int) -> np.ndarray:
    """Arcs along straight grid lines through the given corner nodes."""
    result = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        step = 1 if abs(b - a) < 21 else 21
        step = step if b > a else -step
        for tail in range(a, b, step):
            match = np.flatnonzero((engine.arc_tail == tail) & (engine.arc_head == tail + step))
            result.append(int(match[0]))
    return np.array(result, dtype=np.uint32)


def square(engine: Engine, i: int, j: int, size: int) -> np.ndarray:
    return arcs_via(
        engine, node(i, j), node(i + size, j), node(i + size, j + size), node(i, j + size), node(i, j)
    )


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


def test_shared_start_and_finish_do_not_count_as_overlap(grid_engine):
    # Both loops leave and return along the same 2.5 km street, then go separate ways.
    stem_out = arcs_via(grid_engine, node(0, 10), node(5, 10))
    stem_back = (stem_out ^ 1)[::-1]
    north = np.concatenate([stem_out, square(grid_engine, 5, 10, 8), stem_back])
    south = np.concatenate([stem_out, (square(grid_engine, 5, 2, 8)[::-1] ^ 1), stem_back])
    a, b = candidate(grid_engine, north, 90.0), candidate(grid_engine, south, 80.0)
    assert select_diverse([a, b], 2, grid_engine.length, skip_m=2500.0) == [a, b]
