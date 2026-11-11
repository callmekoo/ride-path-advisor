import json

from rpa.nlu.client import strict_schema
from rpa.nlu.prompt import SYSTEM, ParsedRequest, describe, messages
from rpa.routing.cost import Prefs
from rpa.routing.spec import AvoidArea, LengthKm, Point, RouteSpec

HOME = Point(lon=60.6, lat=56.84, label="дом")


def test_schema_is_strict_and_flat():
    schema = strict_schema(ParsedRequest)

    assert schema["required"] == list(ParsedRequest.model_fields) and schema["additionalProperties"] is False
    assert "$defs" not in schema  # nothing nested for the model to get lost in
    assert {"N", "SW"} <= set(json.dumps(schema).replace('"', " ").split())


def test_every_field_is_explained_to_the_model():
    for name in ParsedRequest.model_fields:
        assert name in SYSTEM, name


def test_a_request_is_described_in_the_models_vocabulary():
    spec = RouteSpec(
        start=HOME,
        shape="loop",
        length_km=LengthKm.around(45),
        prefs=Prefs(avoid_major=0.8, surface="paved", forest=0.7, hills="avoid"),
        via_category="water",
        avoid=[AvoidArea(point=Point(lon=59.93, lat=56.8, label="Ревда (город)"), radius_m=3500)],
        bearing_deg=230.0,
    )

    assert describe(spec) == ParsedRequest(
        shape="loop",
        via_water=True,
        length_km=45.0,
        avoid_major=0.8,
        surface="paved",
        forest=0.7,
        familiarity=0.4,
        hills="avoid",
        avoid=["Ревда (город)"],
        direction="SW",
    )


def test_a_range_and_named_places_are_described_too():
    spec = RouteSpec(
        start=HOME,
        end=Point(lon=59.93, lat=56.8, label="Ревда (город)"),
        shape="p2p",
        length_km=LengthKm.between(100, 150),
        via=Point(lon=60.2, lat=56.75, label="Волчиха (гора)"),
    )
    described = describe(spec)

    assert (described.shape, described.end, described.via) == ("p2p", "Ревда (город)", "Волчиха (гора)")
    assert (described.length_km, described.length_min_km, described.length_max_km) == (None, 100.0, 150.0)
    assert described.start is None and described.direction is None and described.via_water is None


def test_a_fresh_request_is_just_the_words_and_the_saved_places():
    system, user = messages("кольцо 45 км к воде", None, [], ["дом", "дача"])

    assert system == SYSTEM
    assert user == "Сохранённые места: дом, дача\n\nЗапрос: кольцо 45 км к воде"
    assert messages("кольцо 45 км", None, [], [])[1] == "Запрос: кольцо 45 км"


def test_a_refinement_shows_the_previous_request_and_candidates_but_no_coordinates():
    previous = describe(RouteSpec(start=HOME, shape="loop", length_km=LengthKm.around(45), via_category="water"))
    _, user = messages("короче", previous, ["1. Кольцо 44.7 км", "2. Кольцо 46.0 км"], ["дом"])

    assert "Предыдущий запрос: {" in user and '"length_km":45.0' in user and '"via_water":true' in user
    assert "Предложенные варианты:\n1. Кольцо 44.7 км\n2. Кольцо 46.0 км" in user
    assert user.endswith("Новая реплика: короче")
    assert "60.6" not in user and "56.84" not in user
