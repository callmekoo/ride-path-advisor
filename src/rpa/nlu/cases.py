"""Reference phrasings: what a rider says and what the model must understand.

Run against the configured model with `rpa nlu-eval`. An expectation is a value
(place names compare loosely), ("in", [...]) for any of several values, or
("range", low, high) for a number.
"""

from dataclasses import dataclass, field

from rpa.nlu.prompt import ParsedRequest

SAVED_PLACES = ["дом", "дача"]

MOSTLY = ("range", 0.6, 1.0)
FULLY = ("range", 0.9, 1.0)
HARDLY = ("range", 0.0, 0.3)
NOT_SET = ("in", [None, False])


@dataclass(frozen=True)
class Case:
    text: str
    expect: dict
    previous: ParsedRequest | None = None
    candidates: list[str] = field(default_factory=list)


_WATER_LOOP = ParsedRequest(shape="loop", length_km=45, via_water=True, forest=0.7, avoid_major=0.8)
_WATER_LOOP_OFFERED = [
    "1. Кольцо 44.7 км · Верхний Шиловский пруд; набор 364 м; лес 46%; знакомые дороги 82%; крупные дороги 6%",
    "2. Кольцо 46.0 км · озеро Ключи; набор 417 м; лес 49%; знакомые дороги 77%; крупные дороги 4%",
    "3. Кольцо 43.3 км · водохранилище; набор 468 м; лес 54%; знакомые дороги 77%; крупные дороги 9%",
]
_TO_TOWN = ParsedRequest(shape="p2p", start="Ревда (город)", end="Екатеринбург (город)", length_km=80)
_TO_TOWN_OFFERED = ["1. Маршрут 79.2 км; набор 610 м; лес 41%; знакомые дороги 55%; крупные дороги 22%"]


def _refine(text: str, **expect) -> Case:
    return Case(text, expect, _WATER_LOOP, _WATER_LOOP_OFFERED)


CASES = [
    # The three requests the service was designed around.
    Case(
        "заезд около 45 км к воде, желательно по лесу, меньше больших дорог",
        {"shape": ("in", ["loop", None]), "length_km": 45, "via_water": True, "forest": MOSTLY, "avoid_major": MOSTLY},
    ),
    Case(
        "заезд 80 км, от Дидинского тоннеля до Екатеринбурга, поменьше дорог",
        {"shape": "p2p", "start": "Дидинский тоннель", "end": "Екатеринбург", "length_km": 80, "avoid_major": MOSTLY},
    ),
    Case(
        "шоссейный заезд на 100-150 км, желательно кольцевой",
        {"shape": "loop", "surface": "paved", "length_km": None, "length_min_km": 100, "length_max_km": 150},
    ),
    # Lengths.
    Case("кольцо 60 км", {"shape": "loop", "length_km": 60, "start": None, "surface": None, "via_water": NOT_SET}),
    Case("хочу покататься часа три по грунтам", {"length_km": ("range", 50, 70), "surface": "prefer_unpaved"}),
    Case("лёгкая покатушка на часок", {"length_km": ("range", 15, 26)}),
    Case("до Сысерти и обратно, не больше 120 км", {"shape": "loop", "via": "Сысерть", "length_max_km": 120}),
    Case("хотя бы 80 км на юго-запад", {"length_min_km": 80, "direction": "SW"}),
    Case("80-100 км, покрытие не важно", {"length_min_km": 80, "length_max_km": 100, "surface": ("in", ["any", None])}),
    Case("шоссе 120", {"surface": "paved", "length_km": 120}),
    # Places.
    Case("от дома до Ревды", {"shape": "p2p", "start": "дом", "end": "Ревда"}),
    Case("отсюда до Екатеринбурга", {"shape": "p2p", "start": "точка на карте", "end": "Екатеринбург"}),
    Case("с дачи в Ревду через Дегтярск", {"start": "дача", "end": "Ревда", "via": "Дегтярск"}),
    Case("кольцо 80 км через озеро Таватуй", {"shape": "loop", "via": "озеро Таватуй", "via_water": NOT_SET}),
    Case("кольцо 45 км к озеру Шарташ", {"via": "озеро Шарташ", "via_water": NOT_SET, "length_km": 45}),
    Case(
        "от станции Шарташ до Верх-Исетского пруда",
        {"shape": "p2p", "start": "станция Шарташ", "end": "Верх-Исетский пруд"},
    ),
    Case("круг 25 км вокруг дома", {"shape": "loop", "length_km": 25, "start": ("in", ["дом", None])}),
    Case("кольцо 40 км, не через Ревду", {"avoid": ["Ревда"], "length_km": 40}),
    Case(
        "маршрут от Ревды до Первоуральска в объезд Дегтярска",
        {"start": "Ревда", "end": "Первоуральск", "avoid": ["Дегтярск"]},
    ),
    # Directions.
    Case("кольцо на 50 км на север", {"direction": "N", "length_km": 50}),
    Case("из дома на восток километров 35", {"start": "дом", "direction": "E", "length_km": 35}),
    Case(
        "гравийная покатушка на 70 км в сторону Дегтярска",
        {"surface": "prefer_unpaved", "length_km": 70, "toward": "Дегтярск", "end": None},
    ),
    # Preferences.
    Case("100 км по асфальту, поровнее", {"surface": "paved", "hills": "avoid", "length_km": 100}),
    Case("горный заезд километров на 90", {"hills": "seek", "length_km": 90}),
    Case("кольцо 30 км по знакомым дорогам", {"familiarity": MOSTLY, "length_km": 30}),
    Case("хочу что-нибудь новое, километров 50", {"familiarity": HARDLY, "length_km": 50}),
    Case("съездить искупаться, километров 40 туда-обратно", {"via_water": True, "length_km": 40}),
    Case("заезд на 200 км, можно по трассе", {"avoid_major": HARDLY, "length_km": 200}),
    Case("кольцо 55 км только лесом, совсем без трасс", {"forest": FULLY, "avoid_major": FULLY, "length_km": 55}),
    Case("покататься 50 км подальше от машин, с подъёмами", {"avoid_major": MOSTLY, "hills": "seek", "length_km": 50}),
    # Follow-ups to "45 km to the water through the forest".
    _refine("короче", length_km=("range", 28, 40), via_water=True, forest=MOSTLY, shape="loop"),
    _refine("длиннее", length_km=("range", 50, 65), via_water=True),
    _refine("не обязательно к воде", via_water=NOT_SET, length_km=45, forest=MOSTLY),
    _refine("вариант 2, но подлиннее", candidate=2, length_km=("range", 50, 65), via_water=True),
    _refine("не через Ревду", avoid=["Ревда"], length_km=45, via_water=True),
    _refine("сделай 60", length_km=60, via_water=True, avoid_major=MOSTLY),
    _refine("по асфальту", surface="paved", length_km=45),
    _refine("лучше на север", direction="N", length_km=45, via_water=True),
    _refine("а теперь шоссейное кольцо на 100 км", surface="paved", length_km=100, via_water=NOT_SET),
    # Follow-ups to "from Revda to Ekaterinburg, 80 km".
    Case(
        "поменьше трасс",
        {"avoid_major": MOSTLY, "end": "Екатеринбург", "length_km": 80, "shape": "p2p"},
        _TO_TOWN,
        _TO_TOWN_OFFERED,
    ),
    Case("через Дегтярск", {"via": "Дегтярск", "end": "Екатеринбург", "shape": "p2p"}, _TO_TOWN, _TO_TOWN_OFFERED),
]
