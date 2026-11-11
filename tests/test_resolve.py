"""From a parsed request to a RouteSpec or a question, on the synthetic grid with a handful of places."""

import pytest

from rpa.graph.snap import Snapper
from rpa.nlu.prompt import ParsedRequest
from rpa.nlu.resolve import Context, resolve
from rpa.osm.gazetteer import Gazetteer, Place, build_gazetteer
from rpa.routing.spec import LengthKm, Point, RouteSpec


def at(grid, i: int, j: int) -> tuple[float, float]:
    lon, lat = grid.node_lonlat[j * 21 + i].tolist()
    return lon, lat


@pytest.fixture(scope="module")
def places(grid) -> list[Place]:
    return [
        Place("Ревда", "town", *at(grid, 4, 4)),
        Place("Ревда", "village", 61.5, 57.9),
        Place("Ключи", "village", *at(grid, 12, 10)),
        Place("Ключи", "village", *at(grid, 18, 10)),
        Place("Ключи", "village", 62.0, 56.0),
        Place("Сосновка", "hamlet", *at(grid, 10, 14)),
        Place("Сосновка", "hamlet", 61.4, 56.9),
        Place("озеро Круглое", "lake", *at(grid, 16, 16), radius_m=600.0),
        Place("Острая", "peak", 60.08, 57.0),
        Place("Шарташский парк", "park", *at(grid, 2, 18)),
    ]


@pytest.fixture(scope="module")
def gazetteer(places, tmp_path_factory) -> Gazetteer:
    path = tmp_path_factory.mktemp("places") / "gazetteer.sqlite"
    build_gazetteer(places, path)
    return Gazetteer(path)


@pytest.fixture
def context(grid, gazetteer) -> Context:
    home = {"name": "Дом", "lon": at(grid, 10, 10)[0], "lat": at(grid, 10, 10)[1], "is_default": True}
    work = {"name": "работа", "lon": at(grid, 3, 3)[0], "lat": at(grid, 3, 3)[1], "is_default": False}
    return Context(snapper=Snapper(grid), gazetteer=gazetteer, saved_places=[home, work])


def position(point: Point) -> tuple[float, float]:
    return point.lon, point.lat


def test_a_plain_loop_starts_at_the_default_place_with_default_preferences(context, grid):
    resolved = resolve(ParsedRequest(length_km=45), context)

    spec = resolved.spec
    assert resolved.question is None and resolved.warnings == []
    assert spec.shape == "loop" and position(spec.start) == at(grid, 10, 10) and spec.start.label == "Дом"
    assert spec.length_km == LengthKm.around(45)
    assert spec.prefs == RouteSpec.model_fields["prefs"].default
    assert (spec.end, spec.via, spec.via_category, spec.avoid, spec.bearing_deg) == (None, None, None, [], None)


def test_preferences_are_carried_over(context):
    parsed = ParsedRequest(length_km=45, avoid_major=0.8, surface="paved", forest=0.0, familiarity=0.9, hills="seek")
    prefs = resolve(parsed, context).spec.prefs
    assert (prefs.avoid_major, prefs.surface, prefs.forest, prefs.familiarity, prefs.hills) == (
        0.8,
        "paved",
        0.0,
        0.9,
        "seek",
    )


def test_start_is_the_named_place_then_the_click_then_the_previous_then_the_default(context, grid):
    click = Point(lon=60.05, lat=56.85)
    previous = RouteSpec(start=Point(lon=60.1, lat=56.82, label="раньше"), shape="loop", length_km=LengthKm.around(30))
    context.click, context.previous = click, previous

    named = resolve(ParsedRequest(length_km=45, start="работа"), context).spec.start
    assert position(named) == at(grid, 3, 3) and named.label == "работа"
    clicked = resolve(ParsedRequest(length_km=45), context).spec.start
    assert position(clicked) == (60.05, 56.85) and clicked.label == "точка на карте"

    context.click = None
    assert resolve(ParsedRequest(length_km=45), context).spec.start.label == "раньше"
    context.previous = None
    assert resolve(ParsedRequest(length_km=45), context).spec.start.label == "Дом"

    context.saved_places = []
    question = resolve(ParsedRequest(length_km=45), context).question
    assert question.about == "start" and question.options == [] and "Откуда" in question.text


def test_the_clicked_point_can_be_named_in_words(context):
    parsed = ParsedRequest(length_km=45, start="точка на карте")
    assert resolve(parsed, context).question.about == "start"

    context.click = Point(lon=60.05, lat=56.85)
    assert position(resolve(parsed, context).spec.start) == (60.05, 56.85)


def test_a_town_is_found_by_name_and_wins_over_its_namesake_village(context, grid):
    spec = resolve(ParsedRequest(end="Ревда"), context).spec

    assert spec.shape == "p2p" and spec.length_km is None
    assert position(spec.end) == at(grid, 4, 4) and spec.end.label == "Ревда (город)"


def test_a_route_somewhere_needs_to_know_where(context):
    question = resolve(ParsedRequest(shape="p2p", length_km=80), context).question
    assert question.about == "end"

    context.picks = {"end": Point(lon=60.1, lat=56.85, label="точка на карте")}
    assert position(resolve(ParsedRequest(shape="p2p", length_km=80), context).spec.end) == (60.1, 56.85)


def test_a_loop_to_a_place_passes_it(context, grid):
    spec = resolve(ParsedRequest(shape="loop", end="Ревда", length_km=20), context).spec
    assert spec.end is None and position(spec.via) == at(grid, 4, 4) and spec.via.label == "Ревда (город)"


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ({"length_km": 45}, (45, 40.5, 49.5)),
        ({"length_min_km": 100, "length_max_km": 150}, (125, 100, 150)),
        ({"length_min_km": 150, "length_max_km": 100}, (125, 100, 150)),
        ({"length_max_km": 60}, (51, 36, 60)),
        ({"length_min_km": 90}, (100, 90, 126)),
        ({"length_km": 50, "length_max_km": 52}, (50, 45, 52)),
    ],
)
def test_lengths(context, said, expected):
    length = resolve(ParsedRequest(**said), context).spec.length_km
    assert (length.target, length.min, length.max) == pytest.approx(expected)


def test_a_loop_without_a_length_is_asked_about(context):
    question = resolve(ParsedRequest(shape="loop"), context).question
    assert question.about == "length" and "километров" in question.text


def test_namesakes_that_are_both_plausible_are_asked_about(context, grid):
    resolved = resolve(ParsedRequest(length_km=20, via="Ключи"), context)

    question = resolved.question
    assert resolved.spec is None and question.about == "Ключи" and "несколько" in question.text
    assert [option.label for option in question.options] == [
        "Ключи (село), 1 км на восток",
        "Ключи (село), 4 км на восток",
        "Ключи (село), 151 км на юго-восток",
    ]

    context.picks = {"Ключи": question.options[1]}
    spec = resolve(ParsedRequest(length_km=20, via="Ключи"), context).spec
    assert position(spec.via) == at(grid, 18, 10) and spec.via.label == "Ключи (село), 4 км на восток"


def test_a_namesake_far_away_is_not_asked_about(context, grid):
    spec = resolve(ParsedRequest(length_km=20, via="Сосновка"), context).spec
    assert position(spec.via) == at(grid, 10, 14) and spec.via.label == "Сосновка (деревня)"


def test_unknown_and_inexact_names_are_asked_about(context):
    unknown = resolve(ParsedRequest(end="Нигденет"), context).question
    assert unknown.about == "Нигденет" and unknown.options == [] and "Не нашёл" in unknown.text

    inexact = resolve(ParsedRequest(end="Шарташ"), context).question
    assert inexact.about == "Шарташ" and [option.label[:15] for option in inexact.options] == ["Шарташский парк"]

    context.gazetteer = None
    assert resolve(ParsedRequest(end="Ревда"), context).question.about == "Ревда"
    assert resolve(ParsedRequest(end="работа"), context).spec.end.label == "работа"  # saved places need no gazetteer


def test_water_in_general_and_a_named_lake_are_different_wishes(context, grid):
    general = resolve(ParsedRequest(length_km=45, via_water=True), context).spec
    assert general.via_category == "water" and general.via is None

    resolved = resolve(ParsedRequest(length_km=45, via="озеро Круглое", via_water=True), context)
    assert resolved.spec.via_category is None and position(resolved.spec.via) == at(grid, 16, 16)
    assert len(resolved.warnings) == 1 and "озеро Круглое" in resolved.warnings[0]


def test_places_to_avoid_get_a_berth_that_fits_their_kind(context, grid):
    spec = resolve(ParsedRequest(length_km=45, avoid=["Ревда", "Круглое"]), context).spec

    assert [(position(area.point), area.radius_m) for area in spec.avoid] == [
        (at(grid, 4, 4), 3500.0),
        (at(grid, 16, 16), 2100.0),
    ]
    assert resolve(ParsedRequest(length_km=45, avoid=["Нигденет"]), context).question.about == "Нигденет"


def test_a_loop_can_head_a_compass_side_a_place_or_the_way_of_an_earlier_candidate(context, grid):
    assert resolve(ParsedRequest(length_km=45, direction="E"), context).spec.bearing_deg == 90.0
    assert resolve(ParsedRequest(length_km=45, direction="NW"), context).spec.bearing_deg == 315.0

    towards = resolve(ParsedRequest(length_km=45, toward="Ревда"), context).spec
    assert towards.bearing_deg == pytest.approx(225, abs=3) and towards.via is None

    home = at(grid, 10, 10)
    context.previous_candidates = [[home, at(grid, 10, 16), home], [home, at(grid, 4, 10), at(grid, 8, 9), home]]
    assert resolve(ParsedRequest(length_km=30, candidate=2), context).spec.bearing_deg == pytest.approx(270, abs=1)
    assert resolve(ParsedRequest(length_km=30, candidate=1), context).spec.bearing_deg == pytest.approx(0, abs=1)
    assert resolve(ParsedRequest(length_km=30, candidate=7), context).spec.bearing_deg is None
    assert resolve(ParsedRequest(end="Ревда", direction="E"), context).spec.bearing_deg is None


def test_a_refinement_keeps_the_places_already_settled(context, grid):
    chosen = Point(lon=at(grid, 18, 10)[0], lat=at(grid, 18, 10)[1], label="Ключи (село), 4 км на восток")
    context.previous = RouteSpec(start=Point(lon=60.1, lat=56.82), end=chosen, shape="p2p")

    spec = resolve(ParsedRequest(shape="p2p", end="Ключи (село), 4 км на восток", length_km=30), context).spec
    assert spec.end == chosen and position(spec.start) == (60.1, 56.82)


def test_a_place_far_from_any_road_comes_with_a_warning(context):
    resolved = resolve(ParsedRequest(end="гора Острая"), context)

    assert resolved.spec.end.label == "Острая (гора)"
    assert len(resolved.warnings) == 1 and "Острая" in resolved.warnings[0] and "до ближайшей дороги" in resolved.warnings[0]
