"""Canonical requests for judging route quality by eye and by numbers.

Only public landmarks here: this file is committed.
"""

from dataclasses import dataclass

from rpa.routing.cost import Prefs
from rpa.routing.spec import LengthKm, Point, RouteSpec

EKB = Point(lon=60.6057, lat=56.8389, label="Екатеринбург, центр")
REVDA = Point(lon=59.9303, lat=56.8010, label="Ревда")
DEGTYARSK = Point(lon=60.0879, lat=56.7040, label="Дегтярск")
VERKHNYAYA_PYSHMA = Point(lon=60.5944, lat=56.9761, label="Верхняя Пышма")
BERYOZOVSKY = Point(lon=60.8342, lat=56.9135, label="Берёзовский")


@dataclass(frozen=True)
class Case:
    name: str
    text: str  # the request as a person would phrase it
    spec: RouteSpec
    expect_walk: bool = False  # the best route must include a stretch where the bicycle is walked
    # Sizes of a loop without a wanted length; set so that the case does not depend on the ride history.
    loop_sizes_km: tuple[float, float, float] | None = None
    # The object to ride around, by name: its outline comes from the graph's gazetteer when the case runs,
    # so no outline has to be committed here.
    around_name: str | None = None
    max_repeat_share: float | None = None  # in place of the usual limit, e.g. when the start is far from the object


def _loop(km: LengthKm, **prefs) -> RouteSpec:
    return RouteSpec(start=EKB, shape="loop", length_km=km, prefs=Prefs(**prefs))


CASES = [
    Case("loop-45", "кольцо около 45 км", _loop(LengthKm.around(45))),
    Case("loop-45-quiet", "кольцо около 45 км, меньше больших дорог", _loop(LengthKm.around(45), avoid_major=1.0)),
    Case(
        "loop-45-water",
        "заезд около 45 км к воде, меньше больших дорог",
        _loop(LengthKm.around(45), avoid_major=0.9).model_copy(update={"via_category": "water"}),
    ),
    Case("loop-80", "кольцо около 80 км", _loop(LengthKm.around(80))),
    Case("loop-120", "кольцо около 120 км", _loop(LengthKm.around(120))),
    Case(
        "loop-road-100-150",
        "шоссейный заезд на 100–150 км, кольцевой",
        _loop(LengthKm.between(100, 150), surface="paved"),
    ),
    Case(
        "loop-gravel-60",
        "кольцо около 60 км, побольше грунта",
        _loop(LengthKm.around(60), surface="prefer_unpaved", avoid_major=0.8),
    ),
    Case(
        "p2p-revda-80",
        "80 км от Ревды до Екатеринбурга, поменьше дорог",
        RouteSpec(start=REVDA, end=EKB, shape="p2p", length_km=LengthKm.around(80), prefs=Prefs(avoid_major=0.9)),
    ),
    Case("p2p-revda", "от Ревды до Екатеринбурга", RouteSpec(start=REVDA, end=EKB, shape="p2p")),
    Case(
        "p2p-peregon-bridge",
        "от Палкинского Торфяника до Палкинских каменных палаток",
        RouteSpec(
            start=Point(lon=60.3839, lat=56.8496, label="Палкинский Торфяник"),
            end=Point(lon=60.3865, lat=56.8700, label="Палкинские каменные палатки"),
            shape="p2p",
        ),
        expect_walk=True,
    ),
    Case(
        "p2p-revda-via-degtyarsk",
        "из Екатеринбурга в Ревду через Дегтярск",
        RouteSpec(start=EKB, end=REVDA, shape="p2p", via=[DEGTYARSK]),
    ),
    Case(
        "loop-via-two",
        "кольцо из Екатеринбурга через Верхнюю Пышму и Берёзовский",
        RouteSpec(start=EKB, shape="loop", via=[VERKHNYAYA_PYSHMA, BERYOZOVSKY]),
    ),
    Case("loop-no-length", "кольцо", RouteSpec(start=EKB, shape="loop"), loop_sizes_km=(30.0, 50.0, 80.0)),
    # The lake lies some 7 km from the start: the way there and back may share a road.
    Case(
        "around-shartash",
        "вокруг Шарташа",
        RouteSpec(start=EKB, shape="loop"),
        around_name="озеро Шарташ",
        max_repeat_share=0.25,
    ),
    Case(
        "around-vip",
        "вокруг Верх-Исетского пруда",
        RouteSpec(start=EKB, shape="loop"),
        around_name="Верх-Исетское водохранилище",
    ),
    Case(
        "around-vip-60",
        "вокруг Верх-Исетского пруда на 60 км",
        RouteSpec(start=EKB, shape="loop", length_km=LengthKm.around(60)),
        around_name="Верх-Исетское водохранилище",
    ),
]
