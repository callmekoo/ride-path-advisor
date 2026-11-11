import dataclasses

from rpa.eval.cases import CASES, Case
from rpa.eval.run import check, run_case
from rpa.routing.spec import LengthKm, Point, RouteSpec
from rpa.viz.eval_page import render_eval


def grid_case(grid, km: float, name: str = "grid-loop") -> Case:
    lon, lat = grid.node_lonlat[10 * 21 + 10]
    spec = RouteSpec(start=Point(lon=float(lon), lat=float(lat)), shape="loop", length_km=LengthKm.around(km))
    return Case(name, "кольцо </script>", spec)


def test_canonical_cases_have_unique_names():
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
    assert len(names) >= 5


def test_a_feasible_case_passes(grid, grid_engine):
    result = run_case(grid_engine, grid_case(grid, 16.0))
    assert result.problems == []
    assert len(result.candidates) >= 2
    assert result.seconds > 0


def test_an_impossible_case_reports_too_few_candidates(grid, grid_engine):
    # A 10 km x 10 km grid cannot hold a 300 km loop without repeats.
    result = run_case(grid_engine, grid_case(grid, 300.0))
    assert any("кандидатов" in problem for problem in result.problems)


def test_check_flags_repeats_length_and_time(grid, grid_engine):
    case = grid_case(grid, 16.0)
    candidates = run_case(grid_engine, case).candidates
    repeated = dataclasses.replace(candidates[0], features=dataclasses.replace(candidates[0].features, repeat_share=0.2))
    assert any("повторов" in p for p in check(case, [repeated, candidates[1]], 1.0))
    assert any("считалось" in p for p in check(case, candidates, 9.0))
    assert any("длина" in p for p in check(grid_case(grid, 30.0), candidates, 1.0))


def test_render_eval_writes_a_safe_page(grid, grid_engine, tmp_path):
    out = tmp_path / "sub" / "eval.html"
    render_eval([run_case(grid_engine, grid_case(grid, 16.0))], out)
    page = out.read_text()
    assert "grid-loop" in page
    assert "maplibre" in page
    assert "кольцо </script>" not in page
