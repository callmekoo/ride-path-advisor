"""Property tests for the Rust search kernel, checked against scipy's Dijkstra."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import rpa_kernel

NO_ARC = 0xFFFFFFFF
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


class Case:
    """A random graph plus the scipy all-pairs reference."""

    def __init__(self, lonlat, edge_u, edge_v, w, aux):
        self.n = len(lonlat)
        self.w = w
        self.aux = aux
        # Arc a = 2*e + d: d=0 runs u->v, d=1 runs v->u.
        self.tail = np.stack([edge_u, edge_v], axis=1).ravel()
        self.head = np.stack([edge_v, edge_u], axis=1).ravel()
        self.graph = rpa_kernel.Graph(edge_u, edge_v, lonlat)

        dense = np.full((self.n, self.n), np.inf)
        np.minimum.at(dense, (self.tail, self.head), w.astype(np.float64))
        rows, cols = np.nonzero(np.isfinite(dense))
        matrix = csr_matrix((dense[rows, cols], (rows, cols)), shape=(self.n, self.n))
        self.ref = dijkstra(matrix, directed=True)


@st.composite
def cases(draw) -> Case:
    n = draw(st.integers(2, 30))
    cells = draw(
        st.lists(st.tuples(st.integers(0, 20), st.integers(0, 20)), min_size=n, max_size=n, unique=True)
    )
    lonlat = np.array([[60.0 + 0.01 * x, 56.8 + 0.01 * y] for x, y in cells])
    m = draw(st.integers(1, 80))
    pair = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda p: p[0] != p[1])
    pairs = np.array(draw(st.lists(pair, min_size=m, max_size=m)), dtype=np.uint32)
    edge_u, edge_v = pairs[:, 0].copy(), pairs[:, 1].copy()
    stretch = np.array(draw(st.lists(st.floats(1.0, 1.5), min_size=m, max_size=m)))
    length = haversine_m(lonlat[edge_u], lonlat[edge_v]) * stretch
    mult = np.array(
        draw(st.lists(st.sampled_from([1.0, 1.5, 3.0, 5.0, np.inf]), min_size=2 * m, max_size=2 * m))
    )
    w = (np.repeat(length, 2) * mult).astype(np.float32)
    return Case(lonlat, edge_u, edge_v, w, length.astype(np.float32))


def line_graph():
    lonlat = np.array([[60.0, 56.8], [60.01, 56.8], [60.02, 56.8], [60.03, 56.8]])
    edge_u = np.array([0, 1, 2], dtype=np.uint32)
    edge_v = np.array([1, 2, 3], dtype=np.uint32)
    length = haversine_m(lonlat[edge_u], lonlat[edge_v]).astype(np.float32)
    return rpa_kernel.Graph(edge_u, edge_v, lonlat), length


def test_graph_reports_sizes():
    graph, _ = line_graph()
    assert (graph.n_nodes, graph.n_edges) == (4, 3)


def test_route_on_line_returns_arcs_in_order():
    graph, length = line_graph()
    arcs, cost = graph.route(0, 3, np.repeat(length, 2))
    assert arcs.tolist() == [0, 2, 4]
    assert cost == pytest.approx(float(length.sum()), rel=1e-6)


def test_route_backwards_uses_reverse_arcs():
    graph, length = line_graph()
    arcs, _ = graph.route(3, 0, np.repeat(length, 2))
    assert arcs.tolist() == [5, 3, 1]


def test_route_to_self_is_empty():
    graph, length = line_graph()
    arcs, cost = graph.route(2, 2, np.repeat(length, 2))
    assert arcs.tolist() == []
    assert cost == 0.0


def test_forbidden_arc_blocks_one_direction_only():
    graph, length = line_graph()
    w = np.repeat(length, 2)
    w[2] = np.inf  # edge 1, forward
    assert graph.route(0, 3, w) is None
    assert graph.route(3, 0, w) is not None


def test_wrong_weight_length_is_rejected():
    graph, length = line_graph()
    with pytest.raises(ValueError):
        graph.route(0, 3, length)
    with pytest.raises(ValueError):
        graph.sssp(0, length, length)
    with pytest.raises(ValueError):
        graph.sssp(0, np.repeat(length, 2), np.repeat(length, 2))


def test_node_out_of_range_is_rejected():
    graph, length = line_graph()
    with pytest.raises(ValueError):
        graph.route(0, 4, np.repeat(length, 2))
    with pytest.raises(ValueError):
        graph.sssp(9, np.repeat(length, 2), length)


def test_mismatched_edge_arrays_are_rejected():
    lonlat = np.array([[60.0, 56.8], [60.01, 56.8]])
    with pytest.raises(ValueError):
        rpa_kernel.Graph(np.array([0], dtype=np.uint32), np.array([1, 0], dtype=np.uint32), lonlat)
    with pytest.raises(ValueError):
        rpa_kernel.Graph(np.array([0], dtype=np.uint32), np.array([2], dtype=np.uint32), lonlat)


@settings(max_examples=200, deadline=None)
@given(cases(), st.data())
def test_route_matches_scipy(case: Case, data):
    src = data.draw(st.integers(0, case.n - 1))
    dst = data.draw(st.integers(0, case.n - 1))
    result = case.graph.route(src, dst, case.w)
    expected = case.ref[src, dst]
    if np.isinf(expected):
        assert result is None
        return
    arcs, cost = result
    assert cost == pytest.approx(expected, rel=1e-4, abs=1e-6)
    if src == dst:
        assert len(arcs) == 0
        return
    assert case.tail[arcs[0]] == src
    assert case.head[arcs[-1]] == dst
    assert (case.head[arcs[:-1]] == case.tail[arcs[1:]]).all()
    assert case.w[arcs].astype(np.float64).sum() == pytest.approx(cost, rel=1e-5)


@settings(max_examples=200, deadline=None)
@given(cases(), st.data())
def test_sssp_forward_matches_scipy(case: Case, data):
    src = data.draw(st.integers(0, case.n - 1))
    cost, aux_sum, pred = case.graph.sssp(src, case.w, case.aux)
    np.testing.assert_allclose(cost, case.ref[src], rtol=1e-4, atol=1e-6)
    assert pred[src] == NO_ARC
    assert aux_sum[src] == 0.0
    for node in range(case.n):
        if node == src:
            continue
        arc = int(pred[node])
        if np.isinf(cost[node]):
            assert arc == NO_ARC
            assert np.isinf(aux_sum[node])
            continue
        prev = case.tail[arc]
        assert case.head[arc] == node
        assert cost[node] == pytest.approx(cost[prev] + case.w[arc], rel=1e-4)
        assert aux_sum[node] == pytest.approx(aux_sum[prev] + case.aux[arc >> 1], rel=1e-4)


@settings(max_examples=200, deadline=None)
@given(cases(), st.data())
def test_sssp_reverse_matches_scipy(case: Case, data):
    src = data.draw(st.integers(0, case.n - 1))
    cost, _, pred = case.graph.sssp(src, case.w, case.aux, reverse=True)
    np.testing.assert_allclose(cost, case.ref[:, src], rtol=1e-4, atol=1e-6)
    for node in range(case.n):
        if node == src or np.isinf(cost[node]):
            continue
        arc = int(pred[node])
        assert case.tail[arc] == node  # the arc leaves `node` towards `src`
        assert cost[node] == pytest.approx(cost[case.head[arc]] + case.w[arc], rel=1e-4)


@settings(max_examples=100, deadline=None)
@given(cases(), st.data())
def test_sssp_max_cost_truncates(case: Case, data):
    src = data.draw(st.integers(0, case.n - 1))
    finite = case.ref[src][np.isfinite(case.ref[src])]
    cutoff = float(np.median(finite))
    cost, _, _ = case.graph.sssp(src, case.w, case.aux, max_cost=cutoff)
    inside = case.ref[src] < cutoff * 0.999
    outside = case.ref[src] > cutoff * 1.001
    np.testing.assert_allclose(cost[inside], case.ref[src][inside], rtol=1e-4, atol=1e-6)
    assert np.isinf(cost[outside]).all()
