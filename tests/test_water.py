import numpy as np
from helpers import MICRO, node_index

from rpa.enrich.water import load_water, read_water, save_water, water_access


def test_read_water_keeps_lakes_and_rivers_only():
    bodies = read_water(MICRO)
    found = sorted((body.kind, body.name) for body in bodies)
    # The wastewater basin and the stream are left out; the tiny unnamed pond is still read here.
    assert found == [("lake", "Тихое"), ("river", "Быстрая"), ("water", "")]


def test_access_nodes_are_the_graph_nodes_near_water(micro):
    nodes, bodies = water_access(micro, read_water(MICRO))
    by_name = {body["name"]: i for i, body in enumerate(bodies)}
    assert set(by_name) == {"Тихое", "Быстрая"}  # the 15 m pond is too small to matter

    def at(name: str) -> set[int]:
        return {int(micro.node_osm_id[node]) for node, body in nodes if body == by_name[name]}

    # Node 14 is 30 m from the lake shore and node 12 is 91 m; node 6 is 160 m away.
    assert at("Тихое") == {12, 14}
    # The river passes 30 m west of node 1.
    assert at("Быстрая") == {1}


def test_water_bodies_are_described(micro):
    _, bodies = water_access(micro, read_water(MICRO))
    lake = next(body for body in bodies if body["name"] == "Тихое")
    assert lake["kind"] == "lake"
    assert 28_000 < lake["area_m2"] < 33_000  # 183 m x 167 m
    assert 60.0045 < lake["lon"] < 60.0075 and 56.8025 < lake["lat"] < 56.8040


def test_no_water_at_all(micro):
    nodes, bodies = water_access(micro, [])
    assert nodes.shape == (0, 2) and bodies == []


def test_water_round_trip(micro, tmp_path):
    assert load_water(tmp_path) is None
    nodes, bodies = water_access(micro, read_water(MICRO))
    save_water(tmp_path, nodes, bodies)
    loaded_nodes, loaded_bodies = load_water(tmp_path)
    np.testing.assert_array_equal(loaded_nodes, nodes)
    assert loaded_bodies == bodies
    assert node_index(micro, 14) in loaded_nodes[:, 0]
