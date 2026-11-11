"""Turn what the model understood into a RouteSpec, or into a question for the rider.

Everything here is deterministic: places are looked up in the gazetteer and
among the saved places, the start is chosen by fixed rules, and whatever cannot
be settled without the rider becomes one question at a time.
"""

import math
from dataclasses import dataclass, field

from pydantic import BaseModel

from rpa.graph.snap import Snapper
from rpa.nlu.prompt import CLICKED_POINT, DIRECTIONS, ParsedRequest
from rpa.osm.gazetteer import Gazetteer, Hit, describe, rank
from rpa.routing.cost import Prefs
from rpa.routing.geo import bearing_deg
from rpa.routing.spec import AvoidArea, LengthKm, Point, RouteSpec

MAX_OPTIONS = 5
# A namesake is taken without asking when the next one is this much farther away.
CLEARLY_NEARER_FACTOR = 2.0
CLEARLY_NEARER_KM = 10.0
# Farther than this from any road (beyond the place's own size), a place is worth a warning.
FAR_FROM_ROADS_M = 500.0
# How wide a berth "not through X" gives, by the kind of X; an area's own size is added on top.
_AVOID_RADIUS_M = {"city": 5000.0, "town": 3500.0, "village": 2500.0}
_DEFAULT_AVOID_RADIUS_M = 1500.0
# "No more than N" and "at least N" still need a length to aim for.
_OPEN_ENDED_LOW, _OPEN_ENDED_TARGET, _OPEN_ENDED_HIGH = 0.6, 0.85, 1.4
_COMPASS = ("север", "северо-восток", "восток", "юго-восток", "юг", "юго-запад", "запад", "северо-запад")


class Question(BaseModel):
    text: str
    about: str  # the place name in doubt, or "start", "end", "length"
    options: list[Point] = []  # what to choose from; empty when the rider has to name or click a place


class Resolved(BaseModel):
    spec: RouteSpec | None = None
    question: Question | None = None
    warnings: list[str] = []


@dataclass
class Context:
    snapper: Snapper
    gazetteer: Gazetteer | None = None
    saved_places: list[dict] = field(default_factory=list)  # name, lon, lat, is_default
    click: Point | None = None  # where the rider clicked on the map for this request
    previous: RouteSpec | None = None  # the request being refined
    previous_candidates: list[list[tuple[float, float]]] = field(default_factory=list)  # their lines, lon/lat
    picks: dict[str, Point] = field(default_factory=dict)  # answers to earlier questions, by Question.about


@dataclass
class _Found:
    point: Point
    kind: str = "point"
    radius_m: float = 0.0


class _Ask(Exception):
    def __init__(self, question: Question) -> None:
        self.question = question


def resolve(parsed: ParsedRequest, context: Context) -> Resolved:
    try:
        return _resolve(parsed, context)
    except _Ask as ask:
        return Resolved(question=ask.question)


def _resolve(parsed: ParsedRequest, context: Context) -> Resolved:
    warnings: list[str] = []
    start = _start(parsed, context)
    near = (start.point.lon, start.point.lat)

    end = _find(parsed.end, context, near) if parsed.end else None
    via = _find(parsed.via, context, near) if parsed.via else None
    shape = parsed.shape or ("p2p" if end else "loop")
    if shape == "loop" and end is not None:
        # "A loop to Revda": the named place is somewhere to pass, not somewhere to stop.
        via, end = via or end, None
    if shape == "p2p" and end is None and "end" in context.picks:
        end = _Found(context.picks["end"])
    if shape == "p2p" and end is None:
        raise _Ask(Question(text="Куда едем? Назовите место или кликните по карте.", about="end"))

    length = _length(parsed)
    if shape == "loop" and length is None:
        raise _Ask(Question(text="На сколько километров кольцо?", about="length"))

    for found in (start, end, via):
        if found is not None:
            warnings += _far_from_roads(found, context.snapper)

    avoid = []
    for name in parsed.avoid:
        found = _find(name, context, near)
        radius = _AVOID_RADIUS_M.get(found.kind, _DEFAULT_AVOID_RADIUS_M) + found.radius_m
        avoid.append(AvoidArea(point=found.point, radius_m=min(radius, 50_000.0)))

    defaults = Prefs()
    prefs = Prefs(
        avoid_major=defaults.avoid_major if parsed.avoid_major is None else parsed.avoid_major,
        surface=parsed.surface or defaults.surface,
        familiarity=defaults.familiarity if parsed.familiarity is None else parsed.familiarity,
        forest=defaults.forest if parsed.forest is None else parsed.forest,
        hills=parsed.hills or defaults.hills,
    )
    if via is not None and parsed.via_water:
        warnings.append(f"Маршрут пройдёт через {via.point.label}; отдельно к воде он не заезжает.")
    spec = RouteSpec(
        start=start.point,
        end=end.point if end else None,
        shape=shape,
        length_km=length,
        prefs=prefs,
        via=via.point if via else None,
        via_category="water" if parsed.via_water and via is None else None,
        avoid=avoid,
        bearing_deg=_bearing(parsed, context, start.point) if shape == "loop" and via is None else None,
    )
    return Resolved(spec=spec, warnings=warnings)


def _start(parsed: ParsedRequest, context: Context) -> _Found:
    """Named in the text, else clicked on the map, else as before, else the default saved place."""
    anchor = context.click or (context.previous.start if context.previous else None)
    default = next((place for place in context.saved_places if place["is_default"]), None)
    if anchor is None and default is not None:
        anchor = Point(lon=default["lon"], lat=default["lat"])
    if parsed.start and parsed.start.strip().casefold() == CLICKED_POINT and context.click is None:
        raise _Ask(Question(text="Кликните по карте, откуда стартуем.", about="start"))
    if parsed.start:
        return _find(parsed.start, context, (anchor.lon, anchor.lat) if anchor else None)
    if context.click is not None:
        return _Found(context.click.model_copy(update={"label": context.click.label or CLICKED_POINT}))
    if "start" in context.picks:
        return _Found(context.picks["start"])
    if context.previous is not None:
        return _Found(context.previous.start)
    if default is not None:
        return _Found(Point(lon=default["lon"], lat=default["lat"], label=default["name"]))
    raise _Ask(Question(text="Откуда стартуем? Кликните по карте или назовите место.", about="start"))


def _find(name: str, context: Context, near: tuple[float, float] | None) -> _Found:
    """The place called `name`; raises _Ask when the rider has to say which or where."""
    wanted = name.strip().casefold()
    if name in context.picks:
        return _Found(context.picks[name])
    if wanted == CLICKED_POINT and context.click is not None:
        return _Found(context.click.model_copy(update={"label": CLICKED_POINT}))
    for place in context.saved_places:
        if place["name"].casefold() == wanted:
            return _Found(Point(lon=place["lon"], lat=place["lat"], label=place["name"]))
    # A place carried over from the request being refined keeps its position: no second guessing.
    for known in _previous_points(context.previous):
        if known.label and known.label.casefold() == wanted:
            return _Found(known)

    hits = context.gazetteer.search(name, near=near, limit=3 * MAX_OPTIONS) if context.gazetteer else []
    if not hits:
        raise _Ask(Question(text=f"Не нашёл на карте «{name}». Назовите место иначе или кликните по карте.", about=name))
    exact = [hit for hit in hits if hit.exact]
    if not exact:
        options = [_option(hit, near) for hit in hits[:MAX_OPTIONS]]
        raise _Ask(Question(text=f"Не нашёл точно «{name}». Может быть, это одно из этих мест?", about=name, options=options))

    best = exact[0]
    rivals = [hit for hit in exact[1:] if hit.hinted == best.hinted and rank(hit.place.kind) == rank(best.place.kind)]
    if rivals and best.place.kind != "river" and not _clearly_nearer(best, rivals[0], near):
        options = [_option(hit, near) for hit in [best, *rivals][:MAX_OPTIONS]]
        raise _Ask(Question(text=f"Мест с названием «{name}» несколько. Какое имеется в виду?", about=name, options=options))
    place = best.place
    return _Found(Point(lon=place.lon, lat=place.lat, label=describe(place)), place.kind, place.radius_m)


def _clearly_nearer(best: Hit, runner_up: Hit, near: tuple[float, float] | None) -> bool:
    if near is None:
        return False
    return (
        runner_up.distance_km > CLEARLY_NEARER_FACTOR * best.distance_km
        and runner_up.distance_km - best.distance_km > CLEARLY_NEARER_KM
    )


def _option(hit: Hit, near: tuple[float, float] | None) -> Point:
    label = describe(hit.place)
    if near is not None:
        side = _COMPASS[round(bearing_deg(*near, hit.place.lon, hit.place.lat) / 45) % 8]
        label += f", {hit.distance_km:.0f} км на {side}"
    return Point(lon=hit.place.lon, lat=hit.place.lat, label=label)


def _previous_points(previous: RouteSpec | None) -> list[Point]:
    if previous is None:
        return []
    points = [previous.start, previous.end, previous.via, *(area.point for area in previous.avoid)]
    return [point for point in points if point is not None]


def _length(parsed: ParsedRequest) -> LengthKm | None:
    target, low, high = parsed.length_km, parsed.length_min_km, parsed.length_max_km
    if low is not None and high is not None and low > high:
        low, high = high, low
    if target is not None:
        low = min(low, target) if low is not None else 0.9 * target
        high = max(high, target) if high is not None else 1.1 * target
        return LengthKm(target=target, min=low, max=high)
    if low is not None and high is not None:
        return LengthKm.between(low, high)
    if high is not None:
        return LengthKm(target=_OPEN_ENDED_TARGET * high, min=_OPEN_ENDED_LOW * high, max=high)
    if low is not None:
        return LengthKm(target=low / 0.9, min=low, max=_OPEN_ENDED_HIGH * low)
    return None


def _bearing(parsed: ParsedRequest, context: Context, start: Point) -> float | None:
    """Where the loop should head: a compass side, a place to ride towards, or the way an earlier candidate went."""
    if parsed.direction is not None:
        return DIRECTIONS.index(parsed.direction) * 45.0
    if parsed.toward:
        towards = _find(parsed.toward, context, (start.lon, start.lat)).point
        return bearing_deg(start.lon, start.lat, towards.lon, towards.lat) % 360
    if parsed.candidate is not None and 1 <= parsed.candidate <= len(context.previous_candidates):
        line = context.previous_candidates[parsed.candidate - 1]
        scale = math.cos(math.radians(start.lat))
        farthest = max(line, key=lambda p: math.hypot((p[0] - start.lon) * scale, p[1] - start.lat))
        return bearing_deg(start.lon, start.lat, *farthest) % 360
    return None


def _far_from_roads(found: _Found, snapper: Snapper) -> list[str]:
    _, distance = snapper.nearest_node(found.point.lon, found.point.lat)
    if distance <= FAR_FROM_ROADS_M + found.radius_m:
        return []
    name = found.point.label or "Точка"
    return [f"{name}: до ближайшей дороги {distance:.0f} м, маршрут пойдёт от неё."]
