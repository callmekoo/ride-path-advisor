import numpy as np
import pytest
from helpers import edge_index, with_edge

from rpa.osm.tags import CYCLE_INFRA, DISMOUNT, FORD, Highway, Surface
from rpa.routing.cost import (
    DISMOUNT_FACTOR,
    FORD_FACTOR,
    MAX_MULTIPLIER,
    WALKED_DISMOUNT_FACTOR,
    Prefs,
    arc_weights,
)


def multiplier(g, e: int, prefs: Prefs) -> float:
    return float(arc_weights(g, prefs)[2 * e]) / float(g.edge_len_m[e])


@pytest.mark.parametrize("surface", ["any", "paved", "prefer_unpaved"])
@pytest.mark.parametrize("avoid_major", [0.0, 0.5, 1.0])
def test_finite_weights_never_undercut_edge_length(micro, surface, avoid_major):
    # The kernel's A* heuristic relies on this.
    w = arc_weights(micro, Prefs(avoid_major=avoid_major, surface=surface))
    assert w.dtype == np.float32
    assert w.shape == (2 * micro.n_edges,)
    finite = np.isfinite(w)
    assert (w[finite] >= np.repeat(micro.edge_len_m, 2)[finite]).all()


def test_oneway_forbids_the_reverse_arc(micro):
    e = edge_index(micro, 4, 7)
    w = arc_weights(micro, Prefs())
    assert np.isfinite(w[2 * e])
    assert np.isinf(w[2 * e + 1])


def test_plain_cycleway_costs_exactly_its_length(micro):
    g = with_edge(micro, edge_index(micro, 1, 3), edge_highway=Highway.CYCLEWAY, edge_surface=Surface.PAVED)
    assert multiplier(g, edge_index(micro, 1, 3), Prefs(avoid_major=1.0)) == pytest.approx(1.0)


def test_avoid_major_raises_the_cost_of_big_roads_only(micro):
    e = edge_index(micro, 1, 3)
    primary = with_edge(micro, e, edge_highway=Highway.PRIMARY, edge_surface=Surface.PAVED)
    cycleway = with_edge(micro, e, edge_highway=Highway.CYCLEWAY, edge_surface=Surface.PAVED)
    relaxed, strict = Prefs(avoid_major=0.0), Prefs(avoid_major=1.0)
    assert multiplier(primary, e, relaxed) == pytest.approx(1 + 0.3 * 1.2)
    assert multiplier(primary, e, strict) == pytest.approx(1 + 2.0 * 1.2)
    assert multiplier(cycleway, e, relaxed) == multiplier(cycleway, e, strict)


def test_cycle_infrastructure_halves_the_road_class_penalty(micro):
    e = edge_index(micro, 1, 3)
    g = with_edge(micro, e, edge_highway=Highway.PRIMARY, edge_surface=Surface.PAVED, edge_flags=CYCLE_INFRA)
    assert multiplier(g, e, Prefs(avoid_major=1.0)) == pytest.approx(1 + 2.0 * 0.6)


def test_paved_mode_ranks_surfaces_and_forbids_bad_ones(micro):
    e = edge_index(micro, 1, 3)
    prefs = Prefs(avoid_major=0.0, surface="paved")
    by_surface = [
        multiplier(with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=s), e, prefs)
        for s in Surface
    ]
    assert by_surface[: Surface.BAD] == pytest.approx([1.0, 1.1, 2.0, 5.0, 12.0])
    assert np.isinf(by_surface[Surface.BAD])


def test_bad_surface_stays_rideable_outside_paved_mode(micro):
    e = edge_index(micro, 1, 3)
    g = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.BAD)
    assert multiplier(g, e, Prefs(surface="any")) == pytest.approx(1.8)
    assert multiplier(g, e, Prefs(surface="prefer_unpaved")) == pytest.approx(1.5)


def test_prefer_unpaved_makes_asphalt_dearer_than_dirt(micro):
    e = edge_index(micro, 1, 3)
    prefs = Prefs(surface="prefer_unpaved")
    paved = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.PAVED)
    dirt = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.UNPAVED)
    assert multiplier(paved, e, prefs) > multiplier(dirt, e, prefs)


def test_soft_restriction_and_ford_multiply(micro):
    footway = edge_index(micro, 12, 14)  # footway: soft-restricted, unknown surface
    prefs = Prefs(avoid_major=0.5)
    expected = (1 + 1.15 * 0.3) * 1.05 * 3.0
    assert multiplier(micro, footway, prefs) == pytest.approx(expected, rel=1e-5)
    forded = with_edge(micro, footway, edge_flags=FORD)
    assert multiplier(forded, footway, prefs) == pytest.approx(expected * 1.5, rel=1e-5)


def test_multiplier_is_capped(micro):
    e = edge_index(micro, 12, 14)  # soft-restricted
    g = with_edge(micro, e, edge_highway=Highway.TRUNK, edge_surface=Surface.UNPAVED)
    # (1 + 2 * 3.0) * 12 * 3 = 252 without the cap.
    assert multiplier(g, e, Prefs(avoid_major=1.0, surface="paved")) == pytest.approx(50.0)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_avoid_major_out_of_range_is_rejected(value):
    with pytest.raises(ValueError):
        Prefs(avoid_major=value)


def test_unknown_surface_mode_is_rejected():
    with pytest.raises(ValueError):
        Prefs(surface="gravel")


def test_familiarity_makes_unridden_roads_dearer(micro):
    e = edge_index(micro, 1, 3)
    plain = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.PAVED)
    rides = np.zeros(micro.n_edges, dtype=np.uint16)
    prefs = Prefs(avoid_major=0.0, familiarity=1.0)

    def at(count: int) -> float:
        rides[e] = count
        return float(arc_weights(plain, prefs, rides)[2 * e]) / float(plain.edge_len_m[e])

    assert at(0) == pytest.approx(2.5)
    assert at(30) == pytest.approx(1.0)
    assert at(1000) == pytest.approx(1.0)
    assert at(0) > at(1) > at(5) > at(30)


def test_familiarity_zero_or_no_history_changes_nothing(micro):
    rides = np.zeros(micro.n_edges, dtype=np.uint16)
    base = arc_weights(micro, Prefs(familiarity=1.0))
    np.testing.assert_array_equal(arc_weights(micro, Prefs(familiarity=0.0), rides), base)


def test_having_ridden_a_restricted_way_twice_lifts_the_restriction(micro):
    footway = edge_index(micro, 12, 14)
    rides = np.zeros(micro.n_edges, dtype=np.uint16)
    prefs = Prefs(familiarity=0.0)
    restricted = float(arc_weights(micro, prefs, rides)[2 * footway])
    rides[footway] = 1
    assert float(arc_weights(micro, prefs, rides)[2 * footway]) == restricted
    rides[footway] = 2
    assert float(arc_weights(micro, prefs, rides)[2 * footway]) == pytest.approx(restricted / 3.0)


def test_forest_preference_makes_open_roads_dearer(micro):
    e = edge_index(micro, 1, 3)
    plain = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.PAVED)
    forest = np.zeros(micro.n_edges, dtype=np.float32)
    prefs = Prefs(avoid_major=0.0, forest=1.0)

    def at(share: float) -> float:
        forest[e] = share
        return float(arc_weights(plain, prefs, forest=forest)[2 * e]) / float(plain.edge_len_m[e])

    assert at(0.0) == pytest.approx(2.0)
    assert at(0.5) == pytest.approx(1.5)
    assert at(1.0) == pytest.approx(1.0)
    # Without land cover data the preference has no effect.
    np.testing.assert_array_equal(arc_weights(plain, prefs), arc_weights(plain, Prefs(avoid_major=0.0, forest=0.0)))


def test_hill_preferences_depend_on_the_direction_of_travel(micro):
    e = edge_index(micro, 1, 3)
    plain = with_edge(micro, e, edge_highway=Highway.UNCLASSIFIED, edge_surface=Surface.PAVED)
    length = float(plain.edge_len_m[e])
    ascent = np.zeros((micro.n_edges, 2), dtype=np.float32)
    ascent[e] = [0.05 * length, 0.0]  # a 5 % climb riding u -> v, a descent the other way

    def multipliers(hills: str) -> tuple[float, float]:
        w = arc_weights(plain, Prefs(avoid_major=0.0, hills=hills), ascent=ascent)
        return float(w[2 * e]) / length, float(w[2 * e + 1]) / length

    assert multipliers("any") == pytest.approx((1.0, 1.0))
    assert multipliers("avoid") == pytest.approx((1.5, 1.0))
    assert multipliers("seek") == pytest.approx((1.0, 1.5))


def dismount_edge(g):
    e = edge_index(g, 1, 3)
    return e, with_edge(g, e, edge_highway=Highway.CYCLEWAY, edge_surface=Surface.PAVED, edge_flags=DISMOUNT)


def test_dismount_edge_has_its_own_restriction_factor(micro):
    e, g = dismount_edge(micro)
    assert multiplier(g, e, Prefs(avoid_major=0.0, familiarity=0.0, forest=0.0)) == pytest.approx(DISMOUNT_FACTOR)


def test_walked_dismount_edge_is_cheaper_after_two_rides(micro):
    e, g = dismount_edge(micro)
    prefs = Prefs(avoid_major=0.0, familiarity=0.0, forest=0.0)
    rides = np.zeros(g.n_edges, dtype=np.uint16)
    rides[e] = 1
    assert float(arc_weights(g, prefs, rides)[2 * e]) / float(g.edge_len_m[e]) == pytest.approx(DISMOUNT_FACTOR)
    rides[e] = 2
    assert float(arc_weights(g, prefs, rides)[2 * e]) / float(g.edge_len_m[e]) == pytest.approx(WALKED_DISMOUNT_FACTOR)


def test_dismount_and_ford_multiply(micro):
    e, g = dismount_edge(micro)
    g = with_edge(g, e, edge_flags=FORD)
    prefs = Prefs(avoid_major=0.0, familiarity=0.0, forest=0.0)
    assert multiplier(g, e, prefs) == pytest.approx(DISMOUNT_FACTOR * FORD_FACTOR)


@pytest.mark.parametrize("surface", ["any", "paved", "prefer_unpaved"])
@pytest.mark.parametrize("surf_class", list(Surface))
def test_dismount_weight_is_finite_and_not_below_length(micro, surface, surf_class):
    e, g = dismount_edge(micro)
    g = with_edge(g, e, edge_surface=surf_class)
    w = arc_weights(g, Prefs(surface=surface))
    length = float(g.edge_len_m[e])
    if surface == "paved" and surf_class == Surface.BAD:
        assert np.isinf(w[2 * e]) and np.isinf(w[2 * e + 1])
    else:
        assert np.isfinite(w[2 * e]) and np.isfinite(w[2 * e + 1])
        assert w[2 * e] >= length and w[2 * e + 1] >= length


def test_dismount_multiplier_is_capped(micro):
    e, g = dismount_edge(micro)
    g = with_edge(g, e, edge_highway=Highway.TRUNK, edge_surface=Surface.UNPAVED, edge_flags=FORD)
    prefs = Prefs(avoid_major=1.0, surface="paved", familiarity=1.0, forest=1.0, hills="avoid")
    rides = np.zeros(g.n_edges, dtype=np.uint16)
    forest = np.zeros(g.n_edges, dtype=np.float32)
    ascent = np.full((g.n_edges, 2), 100.0, dtype=np.float32)
    w = arc_weights(g, prefs, rides, forest, ascent)
    assert float(w[2 * e]) / float(g.edge_len_m[e]) == pytest.approx(MAX_MULTIPLIER)
