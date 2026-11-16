"""From a parsed request to a RouteSpec or a question, on the synthetic grid with a handful of places."""

import pytest

from rpa.graph.snap import Snapper
from rpa.nlu.prompt import ParsedRequest
from rpa.nlu.resolve import Context, assign_points, resolve, without_clicked
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
    assert (spec.end, spec.via, spec.via_category, spec.avoid, spec.bearing_deg) == (None, [], None, [], None)


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
    context.points, context.previous = [click], previous

    named = resolve(ParsedRequest(length_km=45, start="работа"), context).spec.start
    assert position(named) == at(grid, 3, 3) and named.label == "работа"
    clicked = resolve(ParsedRequest(length_km=45), context).spec.start
    assert position(clicked) == (60.05, 56.85) and clicked.label == "точка на карте"

    context.points = []
    assert resolve(ParsedRequest(length_km=45), context).spec.start.label == "раньше"
    context.previous = None
    assert resolve(ParsedRequest(length_km=45), context).spec.start.label == "Дом"

    context.saved_places = []
    question = resolve(ParsedRequest(length_km=45), context).question
    assert question.about == "start" and question.options == [] and "Откуда" in question.text


def test_the_clicked_point_can_be_named_in_words(context):
    parsed = ParsedRequest(length_km=45, start="точка на карте")
    assert resolve(parsed, context).question.about == "start"

    context.points = [Point(lon=60.05, lat=56.85)]
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
    assert spec.end is None and [position(point) for point in spec.via] == [at(grid, 4, 4)] and spec.via[0].label == "Ревда (город)"


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


def test_a_loop_without_a_length_is_left_to_the_generator(context):
    resolved = resolve(ParsedRequest(shape="loop"), context)
    assert resolved.question is None
    assert resolved.spec.shape == "loop" and resolved.spec.length_km is None


def test_namesakes_that_are_both_plausible_are_asked_about(context, grid):
    resolved = resolve(ParsedRequest(length_km=20, via=["Ключи"]), context)

    question = resolved.question
    assert resolved.spec is None and question.about == "Ключи" and "несколько" in question.text
    assert [option.label for option in question.options] == [
        "Ключи (село), 1 км на восток",
        "Ключи (село), 4 км на восток",
        "Ключи (село), 151 км на юго-восток",
    ]

    context.picks = {"Ключи": question.options[1]}
    spec = resolve(ParsedRequest(length_km=20, via=["Ключи"]), context).spec
    assert [position(point) for point in spec.via] == [at(grid, 18, 10)] and spec.via[0].label == "Ключи (село), 4 км на восток"


def test_a_namesake_far_away_is_not_asked_about(context, grid):
    spec = resolve(ParsedRequest(length_km=20, via=["Сосновка"]), context).spec
    assert [position(point) for point in spec.via] == [at(grid, 10, 14)] and spec.via[0].label == "Сосновка (деревня)"


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
    assert general.via_category == "water" and general.via == []

    resolved = resolve(ParsedRequest(length_km=45, via=["озеро Круглое"], via_water=True), context)
    assert resolved.spec.via_category is None and [position(point) for point in resolved.spec.via] == [at(grid, 16, 16)]
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
    assert towards.bearing_deg == pytest.approx(225, abs=3) and towards.via == []

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


A, B, C, D = (Point(lon=60.0 + i / 100, lat=56.8, label=None) for i in range(4))


@pytest.mark.parametrize(
    ("points", "start_named", "end_named", "loop", "expected"),
    [
        ([], False, False, False, (None, [], None)),
        ([], True, True, True, (None, [], None)),
        ([A], False, False, False, (A, [], None)),
        ([A, B], False, False, False, (A, [], B)),
        ([A, B, C, D], False, False, False, (A, [B, C], D)),
        ([A], True, False, False, (None, [], A)),
        ([A, B, C], True, False, False, (None, [A, B], C)),
        ([A, B, C], False, True, False, (A, [B, C], None)),
        ([A, B, C], True, True, False, (None, [A, B, C], None)),
        ([A, B, C], False, False, True, (A, [B, C], None)),
        ([A, B, C], True, False, True, (None, [A, B, C], None)),
        ([A], False, False, True, (A, [], None)),
    ],
)
def test_clicked_points_get_their_roles_from_what_the_text_names(points, start_named, end_named, loop, expected):
    assert assign_points(points, start_named, end_named, loop) == expected


def test_named_places_pass_in_the_order_said_and_before_the_clicked_ones(context, grid):
    parsed = ParsedRequest(start="работа", end="Ревда", via=["Сосновка", "Дегтярск"])
    context.points = [Point(lon=60.07, lat=56.86), Point(lon=60.08, lat=56.87)]
    context.saved_places.append({"name": "Дегтярск", "lon": 60.2, "lat": 56.9, "is_default": False})
    spec = resolve(parsed, context).spec

    assert [point.label for point in spec.via] == ["Сосновка (деревня)", "Дегтярск", "точка 1", "точка 2"]
    assert spec.end.label == "Ревда (город)" and spec.shape == "p2p"


def test_clicked_points_alone_make_a_route_from_the_first_to_the_last(context):
    points = [Point(lon=60.05, lat=56.85), Point(lon=60.06, lat=56.86), Point(lon=60.07, lat=56.87)]
    context.points = points
    spec = resolve(ParsedRequest(), context).spec

    assert spec.shape == "p2p" and spec.start.label == "точка на карте" and spec.end.label == "точка на карте"
    assert [position(point) for point in spec.via] == [(60.06, 56.86)] and spec.via[0].label == "точка 1"

    spec = resolve(ParsedRequest(shape="loop"), context).spec
    assert spec.shape == "loop" and spec.end is None
    assert [point.label for point in spec.via] == ["точка 1", "точка 2"]


def test_a_namesake_in_the_middle_of_the_list_is_asked_about(context):
    resolved = resolve(ParsedRequest(end="Ревда", via=["Сосновка", "Ключи", "Острая"]), context)

    assert resolved.spec is None and resolved.question.about == "Ключи" and len(resolved.question.options) == 3


def test_a_refinement_keeps_all_the_via_places_and_a_loop_is_not_steered_by_them(context, grid):
    first = Point(lon=at(grid, 18, 10)[0], lat=at(grid, 18, 10)[1], label="Ключи (село), 4 км на восток")
    second = Point(lon=at(grid, 4, 4)[0], lat=at(grid, 4, 4)[1], label="Ревда (город)")
    context.previous = RouteSpec(start=Point(lon=60.1, lat=56.82), shape="loop", via=[first, second])

    spec = resolve(ParsedRequest(shape="loop", via=[first.label, second.label], direction="E"), context).spec
    assert spec.via == [first, second] and spec.bearing_deg is None


def test_a_clicked_via_label_is_resolved_from_the_previous_request(context):
    clicked = Point(lon=60.05, lat=56.85, label="точка 1")
    context.previous = RouteSpec(start=Point(lon=60.1, lat=56.82), shape="loop", via=[clicked])

    spec = resolve(ParsedRequest(shape="loop", via=["точка 1"]), context).spec

    assert spec.via == [clicked]


def test_clicked_labels_are_dropped_when_points_are_sent():
    parsed = ParsedRequest(start="точка на карте", end=" Точка на карте ", via=["точка 1", "Ревда", "Точка 12"])

    cleaned = without_clicked(parsed, [Point(lon=60.0, lat=56.8)])

    assert (cleaned.start, cleaned.end, cleaned.via) == (None, None, ["Ревда"])
    assert parsed.via == ["точка 1", "Ревда", "Точка 12"]  # the original is untouched


def test_a_named_start_or_finish_equal_to_the_label_of_a_sent_point_is_that_point():
    points = [Point(lon=60.0, lat=56.8, label="Дом"), Point(lon=60.1, lat=56.9), Point(lon=60.2, lat=56.9, label="Ревда")]
    parsed = ParsedRequest(start=" дом ", end="РЕВДА", via=["Ключи"])

    cleaned = without_clicked(parsed, points)

    assert (cleaned.start, cleaned.end, cleaned.via) == (None, None, ["Ключи"])


def test_a_name_that_is_not_the_label_of_the_first_or_last_point_stays():
    points = [Point(lon=60.0, lat=56.8, label="Дом"), Point(lon=60.2, lat=56.9, label="Ревда")]

    kept = without_clicked(ParsedRequest(start="Ревда", end="Дом"), points)

    assert (kept.start, kept.end) == ("Ревда", "Дом")


def test_a_clicked_finish_asked_for_without_one_is_a_question(context):
    resolved = resolve(ParsedRequest(end="точка на карте"), context)

    assert resolved.spec is None and (resolved.question.about, resolved.question.text) == ("end", "Кликните по карте, куда едем.")


def test_the_previous_clicked_start_is_not_taken_for_a_clicked_finish(context):
    context.previous = RouteSpec(start=Point(lon=60.1, lat=56.82, label="точка на карте"), shape="loop")

    resolved = resolve(ParsedRequest(end="точка на карте"), context)

    assert resolved.spec is None and resolved.question.about == "end"
