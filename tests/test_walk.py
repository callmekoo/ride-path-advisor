import numpy as np
from helpers import node_index

from rpa.routing.cost import Prefs
from rpa.routing.engine import Engine
from rpa.routing.trim import trim_spurs
from rpa.routing.walk import route_through


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
