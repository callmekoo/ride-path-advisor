from itertools import pairwise

import numpy as np
from helpers import arcs_via, node_index, square

from rpa.routing.cost import Prefs
from rpa.routing.engine import Engine
from rpa.routing.trim import cut_small_loops, tidy, trim_spurs
from rpa.routing.walk import route_legs, route_through


def node(i: int, j: int) -> int:
    return j * 21 + i


def arcs(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint32)


def test_trim_removes_an_out_and_back_spur():
    assert trim_spurs(arcs(0, 2, 3, 4)).tolist() == [0, 4]


def test_trim_unwinds_nested_spurs():
    # 2 4 | 5 3 retraces two edges back to where arc 0 ended.
    assert trim_spurs(arcs(0, 2, 4, 5, 3, 6)).tolist() == [0, 6]


def test_trim_leaves_a_clean_route_alone():
    assert trim_spurs(arcs(0, 2, 4, 6)).tolist() == [0, 2, 4, 6]


def test_trim_of_a_pure_out_and_back_is_empty():
    result = trim_spurs(arcs(0, 2, 3, 1))
    assert result.tolist() == []
    assert result.dtype == np.uint32


def cut(engine, route, max_m, keep=frozenset()):
    return cut_small_loops(route, engine.arc_tail, engine.arc_head, engine.length, max_m, keep)


def assert_connected(engine, route):
    assert (engine.arc_head[route[:-1]] == engine.arc_tail[route[1:]]).all()


def block_loop(engine):
    """East along y=5, a 2 km loop round one cell back to (4, 5), then north."""
    return np.concatenate([
        arcs_via(engine, node(2, 5), node(4, 5)),
        arcs_via(engine, node(4, 5), node(5, 5), node(5, 4), node(4, 4), node(4, 5)),
        arcs_via(engine, node(4, 5), node(4, 8)),
    ])


def straight(engine):
    return arcs_via(engine, node(2, 5), node(4, 5), node(4, 8))


def test_a_small_loop_is_cut_out_of_the_route(grid_engine):
    result = cut(grid_engine, block_loop(grid_engine), 3000.0)
    assert result.tolist() == straight(grid_engine).tolist()


def test_a_loop_longer_than_the_limit_stays(grid_engine):
    route = block_loop(grid_engine)
    assert cut(grid_engine, route, 1000.0).tolist() == route.tolist()


def test_a_loop_through_a_kept_node_stays(grid_engine):
    route = block_loop(grid_engine)
    assert cut(grid_engine, route, 3000.0, frozenset({node(5, 4)})).tolist() == route.tolist()


def test_the_return_to_the_start_is_not_a_loop(grid_engine):
    ring = square(grid_engine, 2, 2, 4)
    for max_m in (1000.0, 7999.0):
        assert cut(grid_engine, ring, max_m).tolist() == ring.tolist()


def test_nested_loops_are_both_cut(grid_engine):
    # A 3 km loop round a block with a 2 km loop hanging off its far corner (6, 6).
    route = np.concatenate([
        arcs_via(grid_engine, node(2, 5), node(4, 5)),
        arcs_via(grid_engine, node(4, 5), node(4, 6), node(6, 6)),
        arcs_via(grid_engine, node(6, 6), node(6, 7), node(7, 7), node(7, 6), node(6, 6)),
        arcs_via(grid_engine, node(6, 6), node(6, 5), node(4, 5)),
        arcs_via(grid_engine, node(4, 5), node(4, 8)),
    ])
    assert_connected(grid_engine, route)
    assert cut(grid_engine, route, 20000.0).tolist() == straight(grid_engine).tolist()


def test_cutting_forgets_the_nodes_it_removed(grid_engine):
    # The loop's nodes are cut away; riding through them later must not count as a return.
    route = np.concatenate([
        block_loop(grid_engine),
        arcs_via(grid_engine, node(4, 8), node(5, 8), node(5, 4)),
    ])
    result = cut(grid_engine, route, 2000.0)
    assert_connected(grid_engine, result)
    assert result.tolist() == np.concatenate([
        straight(grid_engine), arcs_via(grid_engine, node(4, 8), node(5, 8), node(5, 4))
    ]).tolist()


def test_tidy_also_removes_the_out_and_back_a_cut_leaves(grid_engine):
    # East, a loop round a cell, and straight back west: cutting the loop leaves an out-and-back.
    route = np.concatenate([
        arcs_via(grid_engine, node(2, 5), node(4, 5)),
        arcs_via(grid_engine, node(4, 5), node(5, 5), node(5, 4), node(4, 4), node(4, 5)),
        arcs_via(grid_engine, node(4, 5), node(2, 5)),
    ])
    assert tidy(route, grid_engine.arc_tail, grid_engine.arc_head, grid_engine.length, 3000.0).tolist() == []
    assert cut(grid_engine, route, 3000.0).tolist() != []


def test_a_cut_route_is_still_connected(grid_engine):
    result = cut(grid_engine, block_loop(grid_engine), 3000.0)
    assert_connected(grid_engine, result)


def test_walk_visits_the_nodes_in_order(grid_engine):
    w = grid_engine.weights(Prefs())
    stops = [node(2, 2), node(10, 2), node(10, 10)]
    walk = route_through(grid_engine, w, stops)
    assert grid_engine.arc_tail[walk[0]] == stops[0]
    assert grid_engine.arc_head[walk[-1]] == stops[-1]
    assert (grid_engine.arc_head[walk[:-1]] == grid_engine.arc_tail[walk[1:]]).all()
    assert stops[1] in grid_engine.arc_head[walk].tolist()


def test_walk_comes_back_by_different_streets(grid_engine):
    w = grid_engine.weights(Prefs())
    walk = route_through(grid_engine, w, [node(5, 5), node(15, 5), node(5, 5)])
    edges = walk >> 1
    assert len(np.unique(edges)) == len(edges)
    assert grid_engine.arc_tail[walk[0]] == grid_engine.arc_head[walk[-1]] == node(5, 5)


def test_walk_does_not_turn_back_at_a_via_node(grid_engine):
    w = grid_engine.weights(Prefs())
    walk = route_through(grid_engine, w, [node(5, 5), node(15, 5), node(5, 5)])
    assert not (walk[1:] == (walk[:-1] ^ 1)).any()


def test_walk_does_not_modify_the_callers_weights(grid_engine):
    w = grid_engine.weights(Prefs())
    before = w.copy()
    route_through(grid_engine, w, [node(5, 5), node(15, 5), node(5, 5)])
    np.testing.assert_array_equal(w, before)


def test_walk_may_reuse_a_dead_end_street(micro):
    # Node 14 hangs off the map by a single footway: the only way back is the way in.
    engine = Engine(micro)
    w = engine.weights(Prefs())
    start, tip = node_index(micro, 6), node_index(micro, 14)
    walk = route_through(engine, w, [tip, start, tip])
    assert walk is not None
    assert engine.arc_tail[walk[0]] == engine.arc_head[walk[-1]] == tip


def test_walk_fails_when_a_leg_is_impossible(micro):
    engine = Engine(micro)
    w = engine.weights(Prefs())
    # Node 7 can only be left against a one-way street.
    assert route_through(engine, w, [node_index(micro, 4), node_index(micro, 7), node_index(micro, 4)]) is None


def test_walk_tolerates_a_repeated_stop(grid_engine):
    w = grid_engine.weights(Prefs())
    walk = route_through(grid_engine, w, [node(5, 5), node(5, 5), node(8, 5)])
    assert grid_engine.arc_tail[walk[0]] == node(5, 5)
    assert grid_engine.arc_head[walk[-1]] == node(8, 5)


def test_legs_are_the_walk_split_at_the_stops(grid_engine):
    w = grid_engine.weights(Prefs())
    stops = [node(5, 5), node(15, 5), node(15, 12), node(5, 5)]
    legs = route_legs(grid_engine, w, stops)
    assert len(legs) == 3
    for leg, (a, b) in zip(legs, pairwise(stops), strict=True):
        assert grid_engine.arc_tail[leg[0]] == a and grid_engine.arc_head[leg[-1]] == b
    assert np.concatenate(legs).tolist() == route_through(grid_engine, w, stops).tolist()


def test_legs_are_none_when_a_leg_is_impossible(micro):
    engine = Engine(micro)
    w = engine.weights(Prefs())
    assert route_legs(engine, w, [node_index(micro, 4), node_index(micro, 7), node_index(micro, 4)]) is None
