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


def test_check_wants_a_walked_stretch_when_the_case_expects_one(grid, grid_engine):
    case = dataclasses.replace(grid_case(grid, 16.0), expect_walk=True)
    first, *rest = run_case(grid_engine, case).candidates
    assert any("пешего" in p for p in check(case, [first, *rest], 1.0))

    walking = dataclasses.replace(first, features=dataclasses.replace(first.features, walk_m=100.0))
    assert not any("пешего" in p for p in check(case, [walking, *rest], 1.0))
    assert not any("пешего" in p for p in check(case, [], 1.0))
    assert not any("пешего" in p for p in check(grid_case(grid, 16.0), [first, *rest], 1.0))


def grid_point(grid, i: int, j: int, label: str | None = None) -> Point:
    lon, lat = grid.node_lonlat[j * 21 + i]
    return Point(lon=float(lon), lat=float(lat), label=label)


def via_case(grid, *via: Point) -> Case:
    spec = RouteSpec(start=grid_point(grid, 2, 10), end=grid_point(grid, 18, 10), shape="p2p", via=list(via))
    return Case("grid-via", "через два места", spec)


def test_places_passed_in_order_raise_no_problem(grid, grid_engine):
    case = via_case(grid, grid_point(grid, 6, 14, "Север"), grid_point(grid, 14, 6, "Юг"))
    result = run_case(grid_engine, case)
    assert result.candidates
    assert result.problems == []


def test_check_flags_a_place_passed_out_of_order_or_not_at_all(grid, grid_engine):
    north, south = grid_point(grid, 6, 14, "Север"), grid_point(grid, 14, 6, "Юг")
    candidates = run_case(grid_engine, via_case(grid, north, south)).candidates

    swapped = check(via_case(grid, south, north), candidates, 1.0)
    assert any("«Север»" in p and "по порядку" in p for p in swapped)

    elsewhere = grid_point(grid, 2, 18, "Угол")
    assert any("«Угол»" in p for p in check(via_case(grid, north, elsewhere), candidates, 1.0))


def test_a_place_within_reach_of_the_line_counts_as_passed(grid, grid_engine):
    north = grid_point(grid, 6, 14, "Север")
    candidates = run_case(grid_engine, via_case(grid, north)).candidates
    # 0.003 degrees of latitude is about 330 m: inside the 500 m reach.
    beside = north.model_copy(update={"lat": north.lat + 0.003})
    assert check(via_case(grid, beside), candidates, 1.0) == []


def sizes_case(grid, sizes: tuple[float, float, float] | None) -> Case:
    return Case("grid-sizes", "кольцо", RouteSpec(start=grid_point(grid, 10, 10), shape="loop"), loop_sizes_km=sizes)


def test_a_loop_without_a_length_comes_in_the_sizes_the_case_names(grid, grid_engine):
    before = grid_engine.loop_sizes_km
    result = run_case(grid_engine, sizes_case(grid, (6.0, 10.0, 16.0)))
    assert result.problems == []
    lengths = [candidate.path.length_m / 1000 for candidate in result.candidates]
    assert len(lengths) == 3
    assert lengths == sorted(lengths)
    assert grid_engine.loop_sizes_km == before  # the case's sizes do not leak into the next case


def test_check_wants_three_clearly_different_sizes_of_a_loop_without_a_length(grid, grid_engine):
    case = sizes_case(grid, (6.0, 10.0, 16.0))
    short, _, long = run_case(grid_engine, case).candidates

    assert any("три размера" in p for p in check(case, [short, long], 1.0))
    assert any("меньше чем на 20" in p for p in check(case, [short, short, long], 1.0))
    # A loop with a wanted length is not held to three sizes.
    assert not any("размер" in p for p in check(grid_case(grid, 16.0), [short, long], 1.0))


def test_the_new_canonical_cases_are_there():
    by_name = {case.name: case for case in CASES}
    via = by_name["p2p-revda-via-degtyarsk"].spec
    assert via.shape == "p2p" and [point.label for point in via.via] == ["Дегтярск"]
    two = by_name["loop-via-two"].spec
    assert two.shape == "loop" and two.length_km is None and len(two.via) == 2
    plain = by_name["loop-no-length"]
    assert plain.spec.shape == "loop" and plain.spec.length_km is None and not plain.spec.via
    assert plain.loop_sizes_km == (30.0, 50.0, 80.0)
