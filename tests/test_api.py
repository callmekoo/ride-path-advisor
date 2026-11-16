import dataclasses
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import grid_with_hole, rect_outline

from rpa.api.app import create_app
from rpa.api.jobs import TrackImport
from rpa.osm.tags import DISMOUNT
from rpa.routing.engine import DEFAULT_LOOP_SIZES_KM, Engine

GPX_POINT = "{http://www.topografix.com/GPX/1/1}trk/{http://www.topografix.com/GPX/1/1}trkseg/{http://www.topografix.com/GPX/1/1}trkpt"
DEGREES_PER_KM_EAST = 1 / (111.195 * np.cos(np.radians(56.8)))


@pytest.fixture
def client(grid_engine, tmp_path) -> TestClient:
    return TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite"))


def at(grid, i: int, j: int) -> dict:
    lon, lat = grid.node_lonlat[j * 21 + i].tolist()
    return {"lon": lon, "lat": lat}


def loop(grid, km: float = 8.0, **extra) -> dict:
    length = {"target": km, "min": km * 0.9, "max": km * 1.1}
    return {"start": at(grid, 10, 10), "shape": "loop", "length_km": length} | extra


def test_meta_describes_the_graph(client, grid):
    meta = client.get("/api/meta").json()

    assert (meta["nodes"], meta["edges"]) == (441, 840)
    assert meta["road_km"] == pytest.approx(420, rel=0.01)
    west, south, east, north = meta["bbox"]
    assert (west, south) == pytest.approx((60.0, 56.8)) and east > west and north > south
    assert meta["has_rides"] is False and meta["has_water"] is False and meta["has_terrain"] is False
    assert meta["ridden_km"] == 0
    assert meta["has_gazetteer"] is False and meta["has_llm"] is False


def test_loop_request_returns_distinct_closed_candidates(client, grid):
    response = client.post("/api/route", json=loop(grid))
    assert response.status_code == 200
    body = response.json()

    assert body["request_id"] > 0 and body["warnings"] == [] and body["message"] is None
    assert body["spec"]["shape"] == "loop" and body["spec"]["prefs"]["familiarity"] == 0.4
    candidates = body["candidates"]
    assert len(candidates) >= 2
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert 7.2 <= candidate["length_km"] <= 8.8
        line = candidate["geometry"]
        assert line["type"] == "LineString"
        assert line["coordinates"][0] == line["coordinates"][-1] == pytest.approx(list(at(grid, 10, 10).values()))
        assert candidate["title"].startswith("Кольцо")
        assert candidate["features"]["repeat_share"] <= 0.05
        assert "crossings" in candidate["features"]
        assert candidate["km_by_highway"] == {"residential": pytest.approx(candidate["length_km"], abs=0.01)}
        assert candidate["profile"] is None and candidate["ascent_m"] is None


def test_every_request_gets_fresh_ids(client, grid):
    first = client.post("/api/route", json=loop(grid)).json()
    second = client.post("/api/route", json=loop(grid)).json()

    assert second["request_id"] > first["request_id"]
    assert not {c["id"] for c in first["candidates"]} & {c["id"] for c in second["candidates"]}


def test_point_to_point_runs_from_start_to_end(client, grid):
    spec = {"start": at(grid, 2, 10), "end": at(grid, 18, 10), "shape": "p2p", "n_candidates": 1}
    (candidate,) = client.post("/api/route", json=spec).json()["candidates"]

    coordinates = candidate["geometry"]["coordinates"]
    assert coordinates[0] == pytest.approx(list(at(grid, 2, 10).values()))
    assert coordinates[-1] == pytest.approx(list(at(grid, 18, 10).values()))
    assert candidate["length_km"] == pytest.approx(8.0, abs=0.05)


def test_walked_stretches_reach_the_response(grid, tmp_path):
    spec = {"start": at(grid, 2, 10), "end": at(grid, 18, 10), "shape": "p2p", "n_candidates": 1}
    plain = TestClient(create_app(Engine(grid), tmp_path / "plain.sqlite"))
    (candidate,) = plain.post("/api/route", json=spec).json()["candidates"]
    assert candidate["walk_spans"] == [] and candidate["features"]["walk_m"] == 0

    # Walk the whole cross-street cut at i = 10 so that no way round it exists.
    nodes_u, nodes_v = np.asarray(grid.edge_u), np.asarray(grid.edge_v)
    cut = np.flatnonzero((nodes_u % 21 == 10) & (nodes_v % 21 == 11))
    flags = grid.edge_flags.copy()
    flags[cut] |= DISMOUNT
    engine = Engine(dataclasses.replace(grid, edge_flags=flags))
    client = TestClient(create_app(engine, tmp_path / "walk.sqlite"))
    (candidate,) = client.post("/api/route", json=spec).json()["candidates"]

    assert candidate["features"]["walk_m"] > 0
    spans = candidate["walk_spans"]
    assert spans and all(first <= last < len(candidate["geometry"]["coordinates"]) for first, last in spans)


def test_a_start_on_a_walk_only_node_does_not_begin_on_foot(grid, tmp_path):
    at_start = 10 * 21 + 10
    flags = grid.edge_flags.copy()
    flags[(np.asarray(grid.edge_u) == at_start) | (np.asarray(grid.edge_v) == at_start)] |= DISMOUNT
    engine = Engine(dataclasses.replace(grid, edge_flags=flags))
    client = TestClient(create_app(engine, tmp_path / "platform.sqlite"))
    spec = {"start": at(grid, 10, 10), "end": at(grid, 18, 10), "shape": "p2p", "n_candidates": 1}

    (candidate,) = client.post("/api/route", json=spec).json()["candidates"]

    assert candidate["walk_spans"] == [] and candidate["features"]["walk_m"] == 0


def test_malformed_spec_is_rejected(client, grid):
    response = client.post("/api/route", json={"start": at(grid, 10, 10), "shape": "p2p"})
    assert response.status_code == 422


def test_start_outside_the_map_is_refused(client, grid):
    spec = loop(grid)
    spec["start"]["lon"] -= 50 * DEGREES_PER_KM_EAST
    response = client.post("/api/route", json=spec)

    assert response.status_code == 422
    assert "старт" in response.json()["detail"] and "км" in response.json()["detail"]


def test_end_outside_the_map_is_refused(client, grid):
    end = at(grid, 18, 10)
    end["lat"] -= 1.0
    response = client.post("/api/route", json={"start": at(grid, 2, 10), "end": end, "shape": "p2p"})

    assert response.status_code == 422
    assert "финиш" in response.json()["detail"]


def test_an_intermediate_point_is_checked_like_the_ends(client, grid):
    far, near = at(grid, 10, 10), at(grid, 0, 10)  # the grid's edge: no closer road
    far["lat"] -= 1.0
    near["lon"] -= 1.0 * DEGREES_PER_KM_EAST
    spec = {"start": at(grid, 2, 10), "end": at(grid, 18, 10), "shape": "p2p"}

    refused = client.post("/api/route", json={**spec, "via": [far]})
    assert refused.status_code == 422 and "промежуточная точка 1" in refused.json()["detail"]

    far["label"] = "Ключи"
    assert "Ключи" in client.post("/api/route", json={**spec, "via": [far]}).json()["detail"]

    body = client.post("/api/route", json={**spec, "via": [at(grid, 6, 10), near]}).json()
    assert len(body["warnings"]) == 1 and "промежуточная точка 2" in body["warnings"][0]


def test_start_a_little_off_the_roads_gives_a_warning(client, grid):
    start = at(grid, 0, 10)
    start["lon"] -= 1.0 * DEGREES_PER_KM_EAST
    body = client.post("/api/route", json={"start": start, "end": at(grid, 10, 10), "shape": "p2p"}).json()

    assert len(body["candidates"]) >= 1
    assert len(body["warnings"]) == 1 and "старт" in body["warnings"][0]


def test_asking_for_water_the_graph_does_not_know_explains_itself(client, grid):
    body = client.post("/api/route", json=loop(grid, via_category="water")).json()

    assert body["candidates"] == [] and body["request_id"] > 0
    assert "вод" in body["message"]


def test_impossible_request_explains_itself(client, grid):
    body = client.post("/api/route", json=loop(grid, km=500.0)).json()

    assert body["candidates"] == []
    assert body["message"]


def around(grid, i0: int, i1: int, label: str | None) -> dict:
    """An object over the blocks between the streets i0..i1 and j = 6..14."""
    outline = [list(point) for point in rect_outline(grid, i0, i1, 6, 14)]
    lon, lat = np.mean(outline, axis=0).tolist()
    return {"point": {"lon": lon, "lat": lat, "label": label}, "radius_m": 1000, "outline": outline}


@pytest.mark.parametrize(("label", "named"), [("Тестовое (озеро)", "«Тестовое (озеро)»"), (None, "этот объект")])
def test_an_object_with_no_road_on_one_side_explains_itself(grid, tmp_path, label, named):
    # The object reaches past the eastern edge of the map.
    engine = Engine(grid_with_hole(grid, 15, 22, 6, 14))
    client = TestClient(create_app(engine, tmp_path / "rpa.sqlite"))
    spec = {"start": at(grid, 15, 10), "shape": "loop", "around": around(grid, 15, 22, label)}
    body = client.post("/api/route", json=spec).json()

    assert body["candidates"] == []
    assert body["message"] == f"Объехать {named} не получается: с одной стороны нет дороги."


def test_a_ride_around_an_object_is_titled_by_it(grid, tmp_path):
    engine = Engine(grid_with_hole(grid, 8, 12, 6, 14))
    client = TestClient(create_app(engine, tmp_path / "rpa.sqlite"))
    spec = {"start": at(grid, 8, 10), "shape": "loop", "around": around(grid, 8, 12, "Тестовое (озеро)")}
    body = client.post("/api/route", json=spec).json()

    assert body["message"] is None
    assert body["candidates"][0]["title"].endswith(" · вокруг: Тестовое")


def test_candidate_downloads_as_gpx(client, grid):
    candidate = client.post("/api/route", json=loop(grid)).json()["candidates"][0]
    response = client.get(f"/api/candidates/{candidate['id']}.gpx")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/gpx+xml")
    assert f"route-{candidate['id']}.gpx" in response.headers["content-disposition"]
    root = ET.fromstring(response.text)
    assert len(root.findall(GPX_POINT)) == len(candidate["geometry"]["coordinates"])
    assert root.find("{http://www.topografix.com/GPX/1/1}trk/{http://www.topografix.com/GPX/1/1}name").text == (
        candidate["title"]
    )


def test_unknown_candidate_has_no_gpx(client):
    assert client.get("/api/candidates/12345.gpx").status_code == 404


def test_elevation_reaches_the_response_and_the_gpx(grid, tmp_path):
    engine = Engine(grid)
    # Two samples per edge on a plane rising 40 m per km eastwards.
    lon = np.asarray(grid.geom_xy)[:, 0] / 1e7
    height_m = (lon - 60.0) / DEGREES_PER_KM_EAST * 40
    engine.set_terrain(
        forest=np.full(grid.n_edges, 255, dtype=np.uint8),
        prof_off=np.asarray(grid.geom_off),
        prof_ele=np.round(height_m * 10).astype(np.int16),
        ascent=np.zeros((grid.n_edges, 2), dtype=np.float32),
    )
    client = TestClient(create_app(engine, tmp_path / "rpa.sqlite"))
    assert client.get("/api/meta").json()["has_terrain"] is True

    candidate = client.post("/api/route", json=loop(grid)).json()["candidates"][0]
    profile = candidate["profile"]
    assert len(profile["distance_m"]) == len(profile["elevation_m"]) <= 500
    assert profile["distance_m"][0] == 0
    assert profile["distance_m"][-1] == pytest.approx(candidate["length_km"] * 1000, rel=0.01)
    assert profile["elevation_m"][0] == pytest.approx(200, abs=0.2)
    assert candidate["ascent_m"] > 0 and candidate["features"]["forest_share"] == pytest.approx(1.0)

    gpx = ET.fromstring(client.get(f"/api/candidates/{candidate['id']}.gpx").text)
    first = gpx.find(GPX_POINT)
    assert float(first[0].text) == pytest.approx(200, abs=0.2)


def test_places_can_be_created_listed_changed_and_removed(client):
    assert client.get("/api/places").json() == []

    home = client.post("/api/places", json={"name": " дом ", "lon": 60.6, "lat": 56.84})
    assert home.status_code == 201
    assert home.json() == {"id": 1, "name": "дом", "lon": 60.6, "lat": 56.84, "is_default": True}
    work = client.post("/api/places", json={"name": "работа", "lon": 60.5, "lat": 56.8}).json()
    assert work["is_default"] is False

    changed = client.put(f"/api/places/{work['id']}", json={"is_default": True, "lat": 56.81})
    assert changed.json() == {"id": 2, "name": "работа", "lon": 60.5, "lat": 56.81, "is_default": True}
    assert [place["is_default"] for place in client.get("/api/places").json()] == [False, True]

    assert client.delete(f"/api/places/{work['id']}").status_code == 204
    assert [place["name"] for place in client.get("/api/places").json()] == ["дом"]
    assert client.get("/api/places").json()[0]["is_default"] is True


def test_place_errors(client):
    client.post("/api/places", json={"name": "Дом", "lon": 60.6, "lat": 56.84})

    assert client.post("/api/places", json={"name": "дом", "lon": 60.0, "lat": 56.0}).status_code == 409
    assert client.post("/api/places", json={"name": "  ", "lon": 60.0, "lat": 56.0}).status_code == 422
    assert client.post("/api/places", json={"name": "x", "lon": 600.0, "lat": 56.0}).status_code == 422
    assert client.put("/api/places/99", json={"name": "x"}).status_code == 404
    assert client.delete("/api/places/99").status_code == 404


# --- free-text requests --------------------------------------------------------


class FakeLlm:
    """Stands in for the model: answers with prepared structures and remembers what it was asked."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def parse(self, system, user, model):
        self.asked.append(user)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return model.model_validate(answer)


def text_client(grid_engine, tmp_path, *answers) -> tuple[TestClient, FakeLlm]:
    from rpa.osm.gazetteer import Gazetteer, Place, build_gazetteer

    grid = grid_engine.g
    lake = Place("озеро Круглое", "lake", *grid.node_lonlat[16 * 21 + 16].tolist())
    build_gazetteer([lake, Place("Ревда", "town", *grid.node_lonlat[4 * 21 + 4].tolist())], tmp_path / "places.sqlite")
    llm = FakeLlm(*answers)
    client = TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite", Gazetteer(tmp_path / "places.sqlite"), llm))
    return client, llm


def test_text_becomes_a_spec_that_the_router_accepts(grid_engine, grid, tmp_path):
    client, llm = text_client(grid_engine, tmp_path, {"length_km": 8, "avoid_major": 0.8})
    assert client.get("/api/meta").json()["has_llm"] is True
    client.post("/api/places", json={"name": "дом", **at(grid, 10, 10)})

    body = client.post("/api/parse", json={"text": "кольцо 8 км, поменьше больших дорог"}).json()

    assert body["question"] is None and body["warnings"] == []
    assert body["understood"]["length_km"] == 8 and body["understood"]["start"] is None
    assert body["spec"]["start"]["label"] == "дом" and body["spec"]["prefs"]["avoid_major"] == 0.8
    assert llm.asked == ["Сохранённые места: дом\n\nЗапрос: кольцо 8 км, поменьше больших дорог"]
    assert len(client.post("/api/route", json=body["spec"]).json()["candidates"]) >= 2


def test_without_a_start_the_service_asks_and_a_click_answers(grid_engine, grid, tmp_path):
    client, llm = text_client(grid_engine, tmp_path, {"length_km": 8})

    asked = client.post("/api/parse", json={"text": "кольцо 8 км"}).json()
    assert asked["spec"] is None and asked["question"]["about"] == "start"

    answered = client.post("/api/parse", json={"understood": asked["understood"], "points": [at(grid, 10, 10)]}).json()
    assert answered["spec"]["start"]["lon"] == at(grid, 10, 10)["lon"]
    assert len(llm.asked) == 1  # the second call resolved what was already understood


def test_a_follow_up_shows_the_model_the_previous_request_and_what_was_offered(grid_engine, grid, tmp_path):
    client, llm = text_client(
        grid_engine, tmp_path, {"length_km": 16, "via": ["озеро Круглое"]}, {"length_km": 20, "via": ["озеро Круглое"]}
    )
    first = client.post("/api/parse", json={"text": "16 км до Круглого и обратно", "points": [at(grid, 10, 10)]}).json()
    routed = client.post("/api/route", json=first["spec"]).json()
    assert routed["candidates"][0]["via_label"] == "через: озеро Круглое"

    second = client.post("/api/parse", json={"text": "подлиннее", "request_id": routed["request_id"]}).json()

    follow_up = llm.asked[1]
    assert 'Предыдущий запрос: {' in follow_up and '"via":["озеро Круглое"]' in follow_up and '"length_km":16.0' in follow_up
    assert "Предложенные варианты:\n1. Кольцо" in follow_up and "озеро Круглое; лес 0%" in follow_up
    assert follow_up.endswith("Новая реплика: подлиннее") and "60." not in follow_up
    assert second["spec"]["length_km"]["target"] == 20
    assert second["spec"]["start"] == first["spec"]["start"]
    assert second["spec"]["via"] == first["spec"]["via"]
    assert first["spec"]["via"] != []


def test_parse_failures_are_explained(grid_engine, grid, tmp_path):
    from rpa.nlu.client import LlmError

    client, _ = text_client(grid_engine, tmp_path, LlmError("Модель недоступна: таймаут"))
    failed = client.post("/api/parse", json={"text": "кольцо 8 км"})
    assert failed.status_code == 502 and "таймаут" in failed.json()["detail"]
    assert client.post("/api/parse", json={"text": "  "}).status_code == 422
    assert client.post("/api/parse", json={"text": "короче", "request_id": 999}).status_code == 404

    no_llm = TestClient(create_app(grid_engine, tmp_path / "other.sqlite"))
    response = no_llm.post("/api/parse", json={"text": "кольцо 8 км"})
    assert response.status_code == 503 and "LLM_API_KEY" in response.json()["detail"]


def test_clicked_points_alone_make_a_route_without_the_model(client, grid):
    assert client.get("/api/meta").json()["has_llm"] is False
    points = [at(grid, 4, 4), at(grid, 10, 10), at(grid, 16, 16)]

    two = client.post("/api/parse", json={"points": points[::2]})
    assert two.status_code == 200
    spec = two.json()["spec"]
    assert (spec["start"]["lon"], spec["end"]["lon"]) == (points[0]["lon"], points[2]["lon"]) and spec["via"] == []

    spec = client.post("/api/parse", json={"points": points}).json()["spec"]
    assert [(p["lon"], p["lat"]) for p in spec["via"]] == [(points[1]["lon"], points[1]["lat"])]
    assert spec["end"]["lon"] == points[2]["lon"]
    assert client.post("/api/parse", json={"text": " "}).status_code == 422


def test_in_a_loop_the_last_clicked_point_is_passed_not_the_finish(grid_engine, grid, tmp_path):
    client, _ = text_client(grid_engine, tmp_path, {"shape": "loop", "length_km": 10})
    points = [at(grid, 4, 4), at(grid, 10, 10), at(grid, 16, 16)]

    spec = client.post("/api/parse", json={"text": "кольцо", "points": points}).json()["spec"]

    assert spec["shape"] == "loop" and spec["end"] is None and spec["start"]["lon"] == points[0]["lon"]
    assert [p["lon"] for p in spec["via"]] == [points[1]["lon"], points[2]["lon"]]


def test_a_start_asked_about_is_answered_by_sending_the_point_first(grid_engine, grid, tmp_path):
    client, llm = text_client(grid_engine, tmp_path, {"length_km": 8, "end": "озеро Круглое"})
    asked = client.post("/api/parse", json={"text": "8 км до Круглого"}).json()
    assert asked["question"]["about"] == "start"

    answered = client.post("/api/parse", json={"understood": asked["understood"], "points": [at(grid, 4, 4)]}).json()

    assert answered["spec"]["start"]["lon"] == at(grid, 4, 4)["lon"] and answered["spec"]["end"]["label"]
    assert len(llm.asked) == 1


def test_echoed_clicked_labels_do_not_duplicate_the_points_that_are_sent(grid_engine, grid, tmp_path):
    echo = {"length_km": 20, "end": "точка на карте", "via": ["точка 1"]}
    client, _ = text_client(grid_engine, tmp_path, echo)
    first = client.post("/api/parse", json={"points": [at(grid, 4, 4), at(grid, 10, 10), at(grid, 16, 16)]}).json()
    routed = client.post("/api/route", json=first["spec"]).json()

    second = client.post(
        "/api/parse",
        json={"text": "подлиннее", "request_id": routed["request_id"], "points": [at(grid, 4, 4), at(grid, 10, 10), at(grid, 16, 16)]},
    ).json()

    assert second["understood"]["end"] is None and second["understood"]["via"] == []
    assert len(second["spec"]["via"]) == 1 and second["spec"]["end"]["lon"] == at(grid, 16, 16)["lon"]
    assert second["spec"]["length_km"]["target"] == 20


def test_without_new_points_an_echoed_clicked_finish_is_the_previous_finish(grid_engine, grid, tmp_path):
    client, _ = text_client(grid_engine, tmp_path, {"length_km": 20, "end": "точка на карте"})
    first = client.post("/api/parse", json={"points": [at(grid, 4, 4), at(grid, 16, 16)]}).json()
    routed = client.post("/api/route", json=first["spec"]).json()

    second = client.post("/api/parse", json={"text": "подлиннее", "request_id": routed["request_id"]}).json()

    assert second["spec"]["start"]["lon"] == at(grid, 4, 4)["lon"]
    assert second["spec"]["end"]["lon"] == at(grid, 16, 16)["lon"]


def test_too_many_intermediate_points_are_refused_in_words(client, grid):
    client.post("/api/places", json={"name": "дом", **at(grid, 10, 10)})
    nine = [at(grid, i, 4) for i in range(2, 11)]
    named = {"shape": "loop", "start": "дом"}

    refused = client.post("/api/parse", json={"understood": named, "points": nine})
    assert refused.status_code == 422 and refused.json()["detail"] == "Промежуточных точек может быть не больше 8."
    assert client.post("/api/parse", json={"understood": named, "points": nine[:8]}).status_code == 200
    assert client.post("/api/parse", json={"points": [at(grid, i, 4) for i in range(2, 12)]}).status_code == 422


def test_a_named_start_echoed_for_a_clicked_home_keeps_the_points_roles(grid_engine, grid, tmp_path):
    client, _ = text_client(grid_engine, tmp_path, {"length_km": 20, "start": "дом"})
    home = {**at(grid, 10, 10), "label": "дом"}
    points = [home, at(grid, 16, 16)]
    first = client.post("/api/parse", json={"points": points}).json()
    routed = client.post("/api/route", json=first["spec"]).json()

    second = client.post("/api/parse", json={"text": "подлиннее", "request_id": routed["request_id"], "points": points}).json()

    assert second["understood"]["start"] is None
    assert second["spec"]["start"]["lon"] == home["lon"] and second["spec"]["via"] == []
    assert second["spec"]["end"]["lon"] == at(grid, 16, 16)["lon"]


# Saved routes


def test_a_candidate_can_be_saved_opened_exported_and_removed(client, grid):
    answer = client.post("/api/route", json=loop(grid)).json()
    first, second = answer["candidates"][:2]

    saved = client.post("/api/routes", json={"candidate_id": first["id"], "name": " вокруг квартала "})
    assert saved.status_code == 201
    assert saved.json() == {
        "id": saved.json()["id"],
        "name": "вокруг квартала",
        "created": saved.json()["created"],
        "length_km": first["length_km"],
        "ascent_m": None,
    }
    untitled = client.post("/api/routes", json={"candidate_id": second["id"]}).json()
    assert untitled["name"] == second["title"]
    assert client.get("/api/routes").json() == [untitled, saved.json()]

    opened = client.get(f"/api/routes/{untitled['id']}").json()
    assert opened["spec"] == answer["spec"]
    assert opened["candidate"] == {key: value for key, value in second.items() if key != "id"}

    gpx = client.get(f"/api/routes/{untitled['id']}.gpx")
    assert gpx.headers["content-type"].startswith("application/gpx+xml")
    assert len(ET.fromstring(gpx.text).findall(GPX_POINT)) == len(second["geometry"]["coordinates"])

    assert client.delete(f"/api/routes/{untitled['id']}").status_code == 204
    assert client.get("/api/routes").json() == [saved.json()]


def test_saved_route_errors(client):
    assert client.post("/api/routes", json={"candidate_id": 12345}).status_code == 404
    assert client.post("/api/routes", json={"candidate_id": 1, "name": "  "}).status_code == 422
    assert client.get("/api/routes/12345").status_code == 404
    assert client.get("/api/routes/12345.gpx").status_code == 404
    assert client.delete("/api/routes/12345").status_code == 404


# Ridden roads


def test_ridden_roads_come_in_levels(grid, tmp_path):
    rides = np.zeros(grid.n_edges, dtype=np.uint16)
    rides[[0, 1]], rides[2], rides[3], rides[4:] = 1, 4, 60, 7
    client = TestClient(create_app(Engine(grid, rides), tmp_path / "rpa.sqlite"))

    answer = client.get("/api/heatmap")
    roads = answer.json()

    assert answer.headers["content-encoding"] == "gzip"
    assert roads["type"] == "FeatureCollection"
    assert [(f["properties"]["level"], f["properties"]["label"]) for f in roads["features"]] == [
        (0, "1"),
        (1, "2–4"),
        (2, "5–14"),
        (3, "15–49"),
        (4, "50+"),
    ]
    assert [len(f["geometry"]["coordinates"]) for f in roads["features"]] == [2, 1, grid.n_edges - 4, 0, 1]
    assert roads["features"][1]["geometry"]["coordinates"][0] == np.round(grid.edge_geometry(2), 5).tolist()
    lonlat = np.asarray(grid.node_lonlat)
    assert roads["bbox"] == pytest.approx([*lonlat.min(axis=0), *lonlat.max(axis=0)], abs=1e-6)


def test_only_ridden_roads_are_framed(grid, tmp_path):
    rides = np.zeros(grid.n_edges, dtype=np.uint16)
    rides[3] = 2
    client = TestClient(create_app(Engine(grid, rides), tmp_path / "rpa.sqlite"))

    ridden = grid.edge_geometry(3)
    assert client.get("/api/heatmap").json()["bbox"] == pytest.approx([*ridden.min(axis=0), *ridden.max(axis=0)], abs=1e-6)


def test_no_ride_history_means_no_ridden_roads(client):
    assert client.get("/api/heatmap").status_code == 404


# The web page


def test_the_built_page_is_served_next_to_the_api(grid_engine, tmp_path):
    site = tmp_path / "dist"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<!doctype html><title>Маршруты</title>", encoding="utf-8")
    (site / "assets" / "app.js").write_text("console.log(1)")
    client = TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite", static_dir=site))

    assert "Маршруты" in client.get("/").text
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/meta").json()["nodes"] > 0
    assert client.get("/api/nothing").status_code == 404


def test_a_rebuilt_page_is_not_served_from_the_browser_cache(grid_engine, tmp_path):
    site = tmp_path / "dist"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<!doctype html><title>Маршруты</title>", encoding="utf-8")
    (site / "assets" / "index-abc123.js").write_text("console.log(1)")
    client = TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite", static_dir=site))

    # The page names its scripts by content hash, so it alone must be re-checked on every visit.
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/index.html").headers["cache-control"] == "no-cache"
    assert client.get("/assets/index-abc123.js").headers["cache-control"] == "public, max-age=31536000, immutable"
    unchanged = client.get("/", headers={"if-none-match": client.get("/").headers["etag"]})
    assert unchanged.status_code == 304
    assert unchanged.headers["cache-control"] == "no-cache"


def test_without_a_built_page_there_is_only_the_api(grid_engine, tmp_path):
    for static_dir in (None, tmp_path / "missing"):
        client = TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite", static_dir=static_dir))
        assert client.get("/").status_code == 404
        assert client.get("/api/meta").status_code == 200


# New rides without a restart


def save_rides(path, rides) -> None:
    """The way `rpa match` writes them: whole, then swapped in."""
    np.save(path.with_suffix(".tmp.npy"), rides)
    path.with_suffix(".tmp.npy").replace(path)


def test_new_rides_are_picked_up_without_a_restart(grid, tmp_path):
    rides_path = tmp_path / "rides.npy"
    client = TestClient(create_app(Engine(grid), tmp_path / "rpa.sqlite", rides_path=rides_path))
    assert client.get("/api/meta").json()["has_rides"] is False

    rides = np.zeros(grid.n_edges, dtype=np.uint16)
    rides[:3] = 2
    save_rides(rides_path, rides)
    meta = client.get("/api/meta").json()
    assert meta["has_rides"] is True
    assert meta["ridden_km"] == pytest.approx(float(np.asarray(grid.edge_len_m)[:3].sum()) / 1000)
    assert len(client.get("/api/heatmap").json()["features"][1]["geometry"]["coordinates"]) == 3

    rides[3:10] = 2
    save_rides(rides_path, rides)
    assert len(client.get("/api/heatmap").json()["features"][1]["geometry"]["coordinates"]) == 10
    assert client.post("/api/route", json=loop(grid)).json()["candidates"][0]["features"]["ridden_share"] >= 0


def test_rides_for_another_graph_are_ignored(grid, tmp_path):
    rides_path = tmp_path / "rides.npy"
    rides = np.ones(grid.n_edges, dtype=np.uint16)
    client = TestClient(create_app(Engine(grid, rides), tmp_path / "rpa.sqlite", rides_path=rides_path))
    before = client.get("/api/meta").json()["ridden_km"]

    save_rides(rides_path, np.ones(grid.n_edges + 5, dtype=np.uint16))

    assert client.get("/api/meta").json()["ridden_km"] == before
    assert client.post("/api/route", json=loop(grid)).status_code == 200


# Uploading rides


def tracks_client(grid_engine, tmp_path, code: str = "print('готово')") -> tuple[TestClient, TrackImport]:
    tracks = TrackImport(tmp_path / "inbox", tmp_path / "jobs" / "tracks.log", [sys.executable, "-c", code])
    return TestClient(create_app(grid_engine, tmp_path / "rpa.sqlite", tracks=tracks)), tracks


def test_rides_are_uploaded_and_processed(grid_engine, tmp_path):
    client, tracks = tracks_client(grid_engine, tmp_path)
    assert client.get("/api/meta").json()["can_import_tracks"] is True
    assert client.get("/api/tracks").json()["state"] == "idle"

    sent = client.put("/api/tracks/files/Утро в лесу.gpx", content=b"<gpx></gpx>")
    assert (sent.status_code, sent.json()) == (201, {"name": "Утро в лесу.gpx", "size": 11})
    assert (tmp_path / "inbox" / "Утро в лесу.gpx").read_bytes() == b"<gpx></gpx>"

    started = client.post("/api/tracks/import")
    assert started.status_code == 202 and started.json()["state"] in ("running", "done")
    tracks.wait()
    status = client.get("/api/tracks").json()
    assert (status["state"], status["log"]) == ("done", ["готово"])
    assert status["inbox"] == [{"name": "Утро в лесу.gpx", "size": 11}]


def test_upload_refusals_are_explained(grid_engine, tmp_path):
    client, tracks = tracks_client(grid_engine, tmp_path, "import time; time.sleep(30)")
    refused = client.put("/api/tracks/files/notes.txt", content=b"x")
    assert refused.status_code == 400 and ".gpx" in refused.json()["detail"]

    assert client.post("/api/tracks/import").status_code == 202
    try:
        assert client.post("/api/tracks/import").status_code == 409
        assert client.put("/api/tracks/files/a.gpx", content=b"x").status_code == 409
    finally:
        tracks.stop()


def test_without_an_importer_there_is_no_upload(client):
    assert client.get("/api/meta").json()["can_import_tracks"] is False
    assert client.get("/api/tracks").status_code == 404
    assert client.put("/api/tracks/files/a.gpx", content=b"x").status_code == 404
    assert client.post("/api/tracks/import").status_code == 404


def test_new_loop_sizes_are_picked_up_without_a_restart(grid, tmp_path):
    rides_path, sizes_path = tmp_path / "rides.npy", tmp_path / "ride_sizes.json"
    engine = Engine(grid)
    client = TestClient(create_app(engine, tmp_path / "rpa.sqlite", rides_path=rides_path))
    client.get("/api/meta")
    assert engine.loop_sizes_km == DEFAULT_LOOP_SIZES_KM

    sizes_path.write_text('{"km": [25, 40, 60]}')
    client.get("/api/meta")
    assert engine.loop_sizes_km == (25.0, 40.0, 60.0)

    sizes_path.write_text('{"km": [20, 35, 55]}')
    sizes_path.touch()
    client.get("/api/meta")
    assert engine.loop_sizes_km == (20.0, 35.0, 55.0)

    sizes_path.unlink()  # `rpa match` found too few rides and removed the file
    client.get("/api/meta")
    assert engine.loop_sizes_km == DEFAULT_LOOP_SIZES_KM


def test_a_named_finish_of_a_loop_counts_as_a_place_to_pass(grid_engine, grid, tmp_path):
    client, _ = text_client(grid_engine, tmp_path)
    client.post("/api/places", json={"name": "дом", **at(grid, 10, 10)})
    eight = [at(grid, i, 4) for i in range(2, 10)]
    loop_to_town = {"shape": "loop", "start": "дом", "end": "Ревда"}

    refused = client.post("/api/parse", json={"understood": loop_to_town, "points": eight})

    assert refused.status_code == 422 and refused.json()["detail"] == "Промежуточных точек может быть не больше 8."
    assert client.post("/api/parse", json={"understood": loop_to_town, "points": eight[:7]}).status_code == 200
