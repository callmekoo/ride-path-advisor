import dataclasses
import json

import numpy as np
import pytest

from rpa.graph.store import FORMAT_VERSION, GraphData, load_graph, save_graph


def tiny_graph() -> GraphData:
    """Three nodes in a row: edge 0 has an intermediate geometry point."""
    return GraphData(
        node_lonlat=np.array([[60.0, 56.8], [60.002, 56.8], [60.003, 56.8]]),
        node_osm_id=np.array([101, 103, 104], dtype=np.int64),
        edge_u=np.array([0, 1], dtype=np.uint32),
        edge_v=np.array([1, 2], dtype=np.uint32),
        edge_len_m=np.array([121.8, 60.9], dtype=np.float32),
        edge_way_id=np.array([10, 10], dtype=np.int64),
        edge_way_pos=np.array([[0, 2], [2, 3]], dtype=np.uint16),
        edge_highway=np.array([3, 3], dtype=np.uint8),
        edge_surface=np.array([2, 2], dtype=np.uint8),
        edge_flags=np.array([3, 3], dtype=np.uint16),
        geom_off=np.array([0, 3, 5], dtype=np.uint32),
        geom_xy=np.array(
            [
                [600000000, 568000000],
                [600010000, 568000000],
                [600020000, 568000000],
                [600020000, 568000000],
                [600030000, 568000000],
            ],
            dtype=np.int32,
        ),
    )


def test_sizes():
    g = tiny_graph()
    assert (g.n_nodes, g.n_edges) == (3, 2)


def test_edge_geometry_is_in_degrees():
    g = tiny_graph()
    np.testing.assert_allclose(g.edge_geometry(0), [[60.0, 56.8], [60.001, 56.8], [60.002, 56.8]])
    np.testing.assert_allclose(g.edge_geometry(1), [[60.002, 56.8], [60.003, 56.8]])


def test_round_trip_keeps_values_and_dtypes(tmp_path):
    g = tiny_graph()
    save_graph(g, tmp_path / "graph", {"source": "unit-test"})
    loaded, meta = load_graph(tmp_path / "graph")
    for field in dataclasses.fields(GraphData):
        original, restored = getattr(g, field.name), getattr(loaded, field.name)
        assert restored.dtype == original.dtype, field.name
        np.testing.assert_array_equal(restored, original)
    assert meta["source"] == "unit-test"
    assert meta["format_version"] == FORMAT_VERSION
    assert (meta["n_nodes"], meta["n_edges"]) == (3, 2)


def test_load_without_mmap_gives_plain_arrays(tmp_path):
    save_graph(tiny_graph(), tmp_path / "graph", {})
    loaded, _ = load_graph(tmp_path / "graph", mmap=False)
    assert not isinstance(loaded.edge_u, np.memmap)


def test_save_replaces_existing_graph_and_leaves_no_temp_dir(tmp_path):
    target = tmp_path / "graph"
    save_graph(tiny_graph(), target, {"run": 1})
    save_graph(tiny_graph(), target, {"run": 2})
    _, meta = load_graph(target)
    assert meta["run"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["graph"]


def test_load_rejects_other_format_version(tmp_path):
    target = tmp_path / "graph"
    save_graph(tiny_graph(), target, {})
    meta_path = target / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = FORMAT_VERSION + 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format"):
        load_graph(target)


@pytest.mark.parametrize(
    "broken",
    [
        {"edge_v": np.array([1], dtype=np.uint32)},  # length differs from edge_u
        {"edge_u": np.array([0, 3], dtype=np.uint32)},  # node 3 does not exist
        {"edge_len_m": np.array([121.8, 60.9], dtype=np.float64)},  # wrong dtype
        {"geom_off": np.array([0, 3, 4], dtype=np.uint32)},  # does not cover geom_xy
        {"geom_off": np.array([0, 4, 5], dtype=np.uint32)},  # edge 1 has a single point
        {"node_osm_id": np.array([101, 103], dtype=np.int64)},  # one id short
    ],
)
def test_validate_rejects_inconsistent_arrays(broken):
    g = dataclasses.replace(tiny_graph(), **broken)
    with pytest.raises(ValueError):
        g.validate()


def test_save_refuses_invalid_graph(tmp_path):
    g = dataclasses.replace(tiny_graph(), edge_u=np.array([0, 3], dtype=np.uint32))
    with pytest.raises(ValueError):
        save_graph(g, tmp_path / "graph", {})
    assert list(tmp_path.iterdir()) == []
