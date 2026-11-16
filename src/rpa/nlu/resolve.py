"""Turn what the model understood into a RouteSpec, or into a question for the rider.

Everything here is deterministic: places are looked up in the gazetteer and
among the saved places, the start is chosen by fixed rules, and whatever cannot
be settled without the rider becomes one question at a time.
"""

import math
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from rpa.graph.snap import Snapper
from rpa.nlu.prompt import CLICKED_POINT, DIRECTIONS, ParsedRequest
from rpa.osm.gazetteer import Gazetteer, Hit, describe, rank
from rpa.routing.cost import Prefs
from rpa.routing.geo import EARTH_RADIUS_M, bearing_deg
from rpa.routing.spec import Around, AvoidArea, LengthKm, Point, RouteSpec

MAX_OPTIONS = 5
MAX_VIA = 8  # RouteSpec.via is limited to this many
_CLICKED_VIA = re.compile(r"^точка \d+$")
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


class TooManyVia(ValueError):
    """More places to pass than a route can have."""


class Question(BaseModel):
    text: str
    about: str  # the place name in doubt, or "start", "end"
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
    points: list[Point] = field(default_factory=list)  # where the rider clicked on the map, in order
    previous: RouteSpec | None = None  # the request being refined
    previous_candidates: list[list[tuple[float, float]]] = field(default_factory=list)  # their lines, lon/lat
    picks: dict[str, Point] = field(default_factory=dict)  # answers to earlier questions, by Question.about


@dataclass
class _Found:
    point: Point
    kind: str = "point"
    radius_m: float = 0.0
    outline: tuple[tuple[float, float], ...] = ()  # (lon, lat) border of an area; empty for a point


class _Ask(Exception):
    def __init__(self, question: Question) -> None:
        self.question = question


def resolve(parsed: ParsedRequest, context: Context) -> Resolved:
    try:
        return _resolve(parsed, context)
    except _Ask as ask:
        return Resolved(question=ask.question)


def assign_points(
    points: list[Point], start_named: bool, end_named: bool, loop: bool
) -> tuple[Point | None, list[Point], Point | None]:
    """Roles of the clicked points: (start, intermediate, finish).

    The first point is the start unless the text names one; the last is the finish unless the text names
    one or the route is a loop. A lone point with nothing named is only the start.
    """
    rest = list(points)
    start = None if start_named or not rest else rest.pop(0)
    finish = rest.pop() if rest and not end_named and not loop else None
    return start, rest, finish


def ride_order(start: Point, named: list[Point], clicked: list[Point], finish: Point | None) -> list[Point]:
    """All intermediate points in riding order: the shortest straight-line way, the clicked ones in their order.

    Named places may go anywhere, the clicked ones keep their relative order. `finish` is None for a loop,
    which comes back to the start. Candidates are tried from the order of mention (named, then clicked) and
    only a gain of over a metre replaces one, so nothing moves without a reason and the result is repeatable.
    """
    items = [*named, *clicked]
    n, k = len(named), len(clicked)
    if len(items) < 2 or k == len(items):
        return list(items)
    end = finish or start
    nodes = [start, *items, end]
    dist = [[_distance_m(a, b) for b in nodes] for a in nodes]
    last = len(nodes) - 1  # the finish; items are nodes 1..len(items)
    best_length, best_order = math.inf, list(range(1, len(items) + 1))
    order: list[int] = []
    used = [False] * (len(items) + 1)

    def extend(at: int, length: float, next_clicked: int) -> None:
        nonlocal best_length, best_order
        if length >= best_length - 1.0:
            return
        if len(order) == len(items):
            total = length + dist[at][last]
            if total < best_length - 1.0:
                best_length, best_order = total, list(order)
            return
        # Named places in the order of mention, then the next clicked point: the first leaf is today's order.
        options = [i for i in range(1, n + 1) if not used[i]]
        if next_clicked < k:
            options.append(n + 1 + next_clicked)
        for i in options:
            used[i] = True
            order.append(i)
            extend(i, length + dist[at][i], next_clicked + (i > n))
            order.pop()
            used[i] = False

    extend(0, 0.0, 0)
    return [nodes[i] for i in best_order]


def _distance_m(a: Point, b: Point) -> float:
    """Great-circle distance; enough for comparing orders."""
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    h = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(b.lon - a.lon) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _is_named(name: str | None) -> bool:
    return bool(name) and name.strip().casefold() != CLICKED_POINT


def _is_clicked_via(name: str) -> bool:
    return bool(_CLICKED_VIA.match(name.strip().casefold()))


def _same_name(name: str | None, label: str | None) -> bool:
    return bool(name) and bool(label) and name.strip().casefold() == label.strip().casefold()


def without_clicked(parsed: ParsedRequest, points: list[Point]) -> ParsedRequest:
    """The request with the names of the clicked points taken out.

    When points are sent they are the whole truth about clicked places; labels the model echoed
    from an earlier request would only name them a second time. That includes a real name a point
    carries (a saved place picked as the start, a finish chosen among options): echoed back it is
    still the first or the last point, not a place named in words.
    """
    start, end = parsed.start, parsed.end
    if not _is_named(start) or (points and _same_name(start, points[0].label)):
        start = None
    if not _is_named(end) or (points and _same_name(end, points[-1].label)):
        end = None
    return parsed.model_copy(
        update={"start": start, "end": end, "via": [name for name in parsed.via if not _is_clicked_via(name)]}
    )


def _around_name(parsed: ParsedRequest, context: Context) -> str | None:
    """The object to ride around; a saved place has no size, so "around home" is an ordinary loop."""
    if not _is_named(parsed.around):
        return None
    saved = {place["name"].casefold() for place in context.saved_places}
    return None if parsed.around.strip().casefold() in saved else parsed.around


def _around_object(name: str, context: Context, near: tuple[float, float]) -> _Found:
    found = _find(name, context, near)
    if found.kind == "river":
        raise _Ask(Question(text="Объехать реку кругом не получится. Назовите озеро, парк или населённый пункт.", about=name))
    if name in context.picks and context.gazetteer is not None:
        # An answer to a question is only a point; its size and outline are in the gazetteer.
        for hit in context.gazetteer.search(name, near=near, limit=3 * MAX_OPTIONS):
            if (hit.place.lon, hit.place.lat) == (found.point.lon, found.point.lat):
                return _Found(found.point, hit.place.kind, hit.place.radius_m, hit.place.outline)
    return found


def _resolve(parsed: ParsedRequest, context: Context) -> Resolved:
    warnings: list[str] = []
    around_name = _around_name(parsed, context)
    clicked_start, clicked_via, clicked_end = assign_points(
        context.points, _is_named(parsed.start), _is_named(parsed.end), parsed.shape == "loop" or around_name is not None
    )
    start = _start(parsed, context, _labelled(clicked_start, CLICKED_POINT))
    near = (start.point.lon, start.point.lat)

    around = _around_object(around_name, context, near) if around_name else None
    if around is not None:
        return _around_spec(parsed, context, start, around, clicked_via, warnings)

    end = _find(parsed.end, context, near) if _is_named(parsed.end) else None
    if end is None and clicked_end is None and parsed.end and not context.points:
        # "The point on the map" again, with nothing new clicked: the finish as it was, not the start.
        end = _previous_clicked_end(context.previous)
    if end is None and clicked_end is not None:
        end = _Found(_labelled(clicked_end, CLICKED_POINT))
    if end is None and parsed.end:
        if not _is_named(parsed.end):
            raise _Ask(Question(text="Кликните по карте, куда едем.", about="end"))
        end = _find(parsed.end, context, near)
    named = [_find(name, context, near) for name in parsed.via]
    clicked = [_Found(_labelled(point, f"точка {number}")) for number, point in enumerate(clicked_via, start=1)]
    shape = parsed.shape or ("p2p" if end else "loop")
    if shape == "loop" and end is not None:
        # "A loop to Revda": the named place is somewhere to pass, not somewhere to stop.
        named, end = [*named, end], None
    if len(named) + len(clicked) > MAX_VIA:
        raise TooManyVia(f"Промежуточных точек может быть не больше {MAX_VIA}.")
    if shape == "p2p" and end is None and "end" in context.picks:
        end = _Found(context.picks["end"])
    if shape == "p2p" and end is None:
        raise _Ask(Question(text="Куда едем? Назовите место или кликните по карте.", about="end"))

    # Where a named place goes is the service's decision, not the order the rider happened to mention it in.
    by_point = {id(found.point): found for found in (*named, *clicked)}
    ordered = ride_order(start.point, [f.point for f in named], [f.point for f in clicked], end.point if end else None)
    via = [by_point[id(point)] for point in ordered]

    length = _length(parsed)

    for found in (start, end, *via):
        if found is not None:
            warnings += _far_from_roads(found, context.snapper)

    avoid = []
    for name in parsed.avoid:
        found = _find(name, context, near)
        radius = _AVOID_RADIUS_M.get(found.kind, _DEFAULT_AVOID_RADIUS_M) + found.radius_m
        avoid.append(AvoidArea(point=found.point, radius_m=min(radius, 50_000.0)))

    prefs = _prefs(parsed)
    if via and parsed.via_water:
        names = ", ".join(found.point.label or "точка" for found in via)
        warnings.append(f"Маршрут пройдёт через {names}; отдельно к воде он не заезжает.")
    spec = RouteSpec(
        start=start.point,
        end=end.point if end else None,
        shape=shape,
        length_km=length,
        prefs=prefs,
        via=[found.point for found in via],
        via_category="water" if parsed.via_water and not via else None,
        avoid=avoid,
        bearing_deg=_bearing(parsed, context, start.point) if shape == "loop" and not via else None,
    )
    return Resolved(spec=spec, warnings=warnings)


def _around_spec(
    parsed: ParsedRequest, context: Context, start: _Found, around: _Found, clicked_via: list[Point], warnings: list[str]
) -> Resolved:
    """A loop around an object. It is exclusive with places to pass and a direction, so those are said to be ignored."""
    label = around.point.label
    if parsed.end:
        warnings.append(f"Финиш «{parsed.end}» не учитывается: маршрут вокруг объекта — кольцо.")
    if parsed.via or clicked_via or parsed.via_water:
        warnings.append(f"Промежуточные места не учитываются: маршрут идёт вокруг объекта «{label}».")
    if parsed.direction is not None or parsed.toward:
        warnings.append(f"Направление не учитывается: маршрут идёт вокруг объекта «{label}».")
    near = (start.point.lon, start.point.lat)
    avoid = []
    for name in parsed.avoid:
        found = _find(name, context, near)
        radius = _AVOID_RADIUS_M.get(found.kind, _DEFAULT_AVOID_RADIUS_M) + found.radius_m
        avoid.append(AvoidArea(point=found.point, radius_m=min(radius, 50_000.0)))
    spec = RouteSpec(
        start=start.point,
        shape="loop",
        length_km=_length(parsed),
        prefs=_prefs(parsed),
        avoid=avoid,
        around=Around(point=around.point, radius_m=min(around.radius_m, 100_000.0), outline=list(around.outline)),
    )
    return Resolved(spec=spec, warnings=[*_far_from_roads(start, context.snapper), *warnings])


def _prefs(parsed: ParsedRequest) -> Prefs:
    defaults = Prefs()
    return Prefs(
        avoid_major=defaults.avoid_major if parsed.avoid_major is None else parsed.avoid_major,
        surface=parsed.surface or defaults.surface,
        familiarity=defaults.familiarity if parsed.familiarity is None else parsed.familiarity,
        forest=defaults.forest if parsed.forest is None else parsed.forest,
        hills=parsed.hills or defaults.hills,
    )


def _labelled(point: Point | None, label: str) -> Point | None:
    """A clicked point keeps a label it already carries."""
    return point.model_copy(update={"label": point.label or label}) if point is not None else None


def _start(parsed: ParsedRequest, context: Context, clicked: Point | None) -> _Found:
    """Named in the text, else clicked on the map, else as before, else the default saved place."""
    first = context.points[0] if context.points else None
    anchor = first or (context.previous.start if context.previous else None)
    default = next((place for place in context.saved_places if place["is_default"]), None)
    if anchor is None and default is not None:
        anchor = Point(lon=default["lon"], lat=default["lat"])
    if parsed.start and not _is_named(parsed.start) and clicked is None:
        raise _Ask(Question(text="Кликните по карте, откуда стартуем.", about="start"))
    if _is_named(parsed.start):
        return _find(parsed.start, context, (anchor.lon, anchor.lat) if anchor else None)
    if clicked is not None:
        return _Found(clicked)
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
    if wanted == CLICKED_POINT and context.points:
        return _Found(_labelled(context.points[0], CLICKED_POINT))
    if _is_clicked_via(wanted) and context.previous is not None:
        # Before saved places: "точка 1" is a label of this very request, whatever a place may be called.
        for known in context.previous.via:
            if known.label and known.label.casefold() == wanted:
                return _Found(known)
    for place in context.saved_places:
        if place["name"].casefold() == wanted:
            return _Found(Point(lon=place["lon"], lat=place["lat"], label=place["name"]))
    # The object of a ride around it keeps its outline too.
    around = context.previous.around if context.previous else None
    if around is not None and _same_name(name, around.point.label):
        return _Found(around.point, radius_m=around.radius_m, outline=tuple((lon, lat) for lon, lat in around.outline))
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
    return _Found(Point(lon=place.lon, lat=place.lat, label=describe(place)), place.kind, place.radius_m, place.outline)


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


def _previous_clicked_end(previous: RouteSpec | None) -> _Found | None:
    if previous is not None and previous.end is not None and not _is_named(previous.end.label):
        return _Found(previous.end)
    return None


def _previous_points(previous: RouteSpec | None) -> list[Point]:
    if previous is None:
        return []
    points = [previous.start, previous.end, *previous.via, *(area.point for area in previous.avoid)]
    if previous.around is not None:
        points.append(previous.around.point)
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
