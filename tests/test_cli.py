import json

import pytest
from helpers import MICRO
from typer.testing import CliRunner

from rpa.cli import app
from rpa.graph.store import FORMAT_VERSION, load_graph
from rpa.routing.path import RoutePath
from rpa.viz.map import render_map

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """An isolated data directory holding a graph built from the micro map."""
    monkeypatch.setenv("RPA_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["build-graph", str(MICRO)])
    assert result.exit_code == 0, result.output
    return tmp_path / "data"


def test_build_graph_writes_a_versioned_directory_and_points_current_at_it(data_dir):
    graph_root = data_dir / "graph"
    built = [p for p in graph_root.iterdir() if p.name != "current"]
    assert len(built) == 1
    assert built[0].name.endswith(f"-v{FORMAT_VERSION}")
    assert (graph_root / "current").resolve() == built[0].resolve()

    graph, meta = load_graph(graph_root / "current")
    assert graph.n_edges == 7
    assert meta["source_file"] == "micro.opl"
    assert len(meta["source_sha256"]) == 64


def test_build_graph_reports_what_it_built(tmp_path):
    result = runner.invoke(app, ["build-graph", str(MICRO), "--out", str(tmp_path / "g")])
    assert result.exit_code == 0, result.output
    assert "7" in result.output
    assert "residential" in result.output
    assert json.loads((tmp_path / "g" / "meta.json").read_text())["n_nodes"] == 7


def test_rebuilding_the_same_file_is_fine(data_dir):
    result = runner.invoke(app, ["build-graph", str(MICRO)])
    assert result.exit_code == 0, result.output


def test_route_prints_a_summary_and_writes_a_map(data_dir, tmp_path):
    html = tmp_path / "out" / "route.html"
    result = runner.invoke(app, ["route", "--from", "56.800,60.000", "--to", "56.803,60.004", "--html", str(html)])
    assert result.exit_code == 0, result.output
    assert "км" in result.output
    assert "residential" in result.output
    page = html.read_text()
    assert "FeatureCollection" in page
    assert "maplibre" in page
    assert "OpenStreetMap" in page


def test_route_warns_when_a_point_is_far_from_any_road(data_dir):
    result = runner.invoke(app, ["route", "--from", "56.900,60.000", "--to", "56.803,60.004"])
    assert result.exit_code == 0, result.output
    assert "далеко" in result.output


def test_route_fails_clearly_when_there_is_no_way_through(data_dir):
    # Node 7 can only be left against a one-way street.
    result = runner.invoke(app, ["route", "--from", "56.800,60.004", "--to", "56.800,60.000"])
    assert result.exit_code == 1
    assert "не найден" in result.output


def test_route_rejects_malformed_coordinates(data_dir):
    result = runner.invoke(app, ["route", "--from", "somewhere", "--to", "56.803,60.004"])
    assert result.exit_code == 2


def test_route_without_a_graph_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("RPA_DATA_DIR", str(tmp_path / "nothing"))
    result = runner.invoke(app, ["route", "--from", "56.800,60.000", "--to", "56.803,60.004"])
    assert result.exit_code == 1
    assert "build-graph" in result.output


def test_render_map_embeds_routes_safely(tmp_path):
    import numpy as np

    route = RoutePath(
        arcs=np.array([0], dtype=np.uint32),
        coords=np.array([[60.0, 56.8], [60.001, 56.8]]),
        length_m=60.9,
        km_by_highway={"residential": 0.0609},
        km_by_surface={"unknown": 0.0609},
    )
    out = tmp_path / "map.html"
    render_map([route], ["</script><b>x</b>"], out)
    page = out.read_text()
    assert "[60.0, 56.8]" in page
    assert "</script><b>" not in page


def test_serve_without_a_graph_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("RPA_DATA_DIR", str(tmp_path / "nothing"))
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "build-graph" in result.output


def test_serve_starts_the_api_on_the_built_graph(data_dir, monkeypatch):
    import uvicorn

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **options: started.update(app=application, **options))
    monkeypatch.chdir(data_dir)  # away from the developer's .env
    monkeypatch.setenv("RPA_WEB_DIR", str(data_dir / "no-page"))
    result = runner.invoke(app, ["serve", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert (started["host"], started["port"]) == ("127.0.0.1", 8123)
    assert {"/api/route", "/api/meta", "/api/places"} <= {route.path for route in started["app"].routes}
    assert (data_dir / "rpa.sqlite").exists()
    assert "Интерфейс: не собран" in result.output


def test_serve_hands_out_the_built_page(data_dir, tmp_path, monkeypatch):
    import uvicorn

    site = tmp_path / "dist"
    site.mkdir()
    (site / "index.html").write_text("<title>rpa</title>")
    monkeypatch.setenv("RPA_WEB_DIR", str(site))
    monkeypatch.chdir(data_dir)  # away from the developer's .env
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **options: started.update(app=application, **options))
    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert "Интерфейс: http://127.0.0.1:8000/" in result.output
    assert "" in {getattr(route, "path", None) for route in started["app"].routes}


def test_serve_watches_rides_and_takes_uploads(data_dir, monkeypatch):
    import uvicorn

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **options: started.update(app=application))
    monkeypatch.chdir(data_dir)  # away from the developer's .env
    monkeypatch.setenv("RPA_WEB_DIR", str(data_dir / "no-page"))
    assert runner.invoke(app, ["serve"]).exit_code == 0

    from fastapi.testclient import TestClient

    client = TestClient(started["app"])
    assert client.get("/api/meta").json()["can_import_tracks"] is True
    assert client.put("/api/tracks/files/a.gpx", content=b"<gpx/>").status_code == 201
    assert (data_dir / "inbox" / "a.gpx").exists()


def test_sync_tracks_without_an_inbox_explains_what_to_do(data_dir):
    result = runner.invoke(app, ["sync-tracks"])
    assert result.exit_code == 1 and "inbox" in result.output


def test_update_rebuilds_everything_from_the_map_at_hand(data_dir):
    before = {path.name for path in (data_dir / "graph").iterdir()}
    result = runner.invoke(app, ["update", "--pbf", str(MICRO), "--no-rasters"])

    assert result.exit_code == 0, result.output
    assert "Перезапустите сервер" in result.output
    current = data_dir / "graph" / "current"
    assert {path.name for path in (data_dir / "graph").iterdir()} == before  # the same map gives the same graph
    assert (current / "gazetteer.sqlite").exists() and (current / "water.json").exists()


def test_update_without_a_map_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("RPA_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["update", "--skip-fetch"])
    assert result.exit_code == 1 and "--skip-fetch" in result.output


def test_gazetteer_indexes_named_places_of_the_map(data_dir):
    result = runner.invoke(app, ["gazetteer", str(MICRO), "--find", "озеро Тихое", "--find", "Нигденет"])

    assert result.exit_code == 0, result.output
    assert "Мест:" in result.output and "озеро Тихое: Тихое (озеро)" in result.output
    assert "Нигденет: ничего" in result.output
    assert (data_dir / "graph" / "current" / "gazetteer.sqlite").exists()


def test_nlu_eval_without_a_model_explains_what_to_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(app, ["nlu-eval"])
    assert result.exit_code == 1 and "LLM_API_KEY" in result.output
