import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rpa.api.app import create_app
from rpa.routing.engine import Engine

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


def test_malformed_spec_is_rejected(client, grid):
    response = client.post("/api/route", json={"start": at(grid, 10, 10), "shape": "loop"})
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
