"""Generate candidate routes for a RouteSpec.

A loop is built as a walk start -> via nodes -> start, where the via nodes sit
on the corners of a regular polygon that has the start as one corner. Many
polygons are tried (different corner counts and orientations); each is
resized until the walk has the wanted length. A point-to-point route with a
target length goes through one via node chosen so that the way there plus the
way on add up to the target. A route through named places follows them in
the given order and is lengthened, when it has to be, on its last leg.

All attempts run in a thread pool: the search kernel releases the GIL.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rpa.routing.diversity import select_diverse
from rpa.routing.engine import Engine
from rpa.routing.geo import destination
from rpa.routing.path import build_path
from rpa.routing.score import Candidate, features, rejection, score
from rpa.routing.spec import AvoidArea, LengthKm, Point, RouteSpec
from rpa.routing.trim import tidy
from rpa.routing.walk import route_legs, route_through

# First guess at how much longer roads are than the straight line between two points.
DETOUR = 1.3
VIA_COUNTS = (1, 2, 3)
N_BEARINGS = 12
MAX_FIT_ITERATIONS = 5
# A closed sub-walk this short (a block circled on the way to a via) is cut out of the route.
APPENDIX_SHARE = 0.07  # of the target length
MIN_APPENDIX_M = 1000.0
# A via node must be a real junction, or the walk would have to turn back at it.
MIN_VIA_DEGREE = 3
MIN_SNAP_RADIUS_M = 300.0
SNAP_RADIUS_SHARE = 0.25  # of the polygon side
# The corner that must be at a wanted place (water) searches a wider circle.
MIN_MUST_RADIUS_M = 1000.0
MUST_RADIUS_SHARE = 0.4
# Below this ratio of target to direct length there is no room for a via node.
MIN_DETOUR_FOR_VIA = 1.15
P2P_BAND_SHARE = 0.05
# Edges of an already found alternative cost this much more in the next search.
ALTERNATIVE_FACTOR = 3.0
# Roads inside an area the rider asked to avoid: dear, but usable when there is no other way.
AVOID_FACTOR = 8.0
# A loop asked to head somewhere is built only from polygons within this angle of that heading.
HEADING_TOLERANCE_DEG = 45.0
# Without a wanted length, a route through a place may also take detours about this much longer.
PLACE_DETOUR = 1.2
# Routes through named places all share the roads near the places; when the usual limit leaves
# fewer than the wanted number of routes, more are taken that overlap the chosen ones up to this much.
VIA_MAX_OVERLAP = 0.6
N_WORKERS = min(8, os.cpu_count() or 1)

# The arcs of a walk and the nodes it was asked to pass, in order (empty if there was no such wish).
Walk = tuple[np.ndarray, tuple[int, ...]]


def generate(engine: Engine, spec: RouteSpec) -> list[Candidate]:
    """Up to `spec.n_candidates` distinct routes, best first; empty if nothing fits."""
    w = _with_avoided(engine, engine.weights(spec.prefs), spec.avoid)
    start, _ = engine.snapper.nearest_node(spec.start.lon, spec.start.lat)
    end = start if spec.shape == "loop" else engine.snapper.nearest_node(spec.end.lon, spec.end.lat)[0]
    places = _chain(start, [_place_node(engine, p) for p in spec.via], end)
    if spec.shape == "loop" and spec.length_km is None and not places:
        return []
    must = None
    if spec.via_category == "water":
        if engine.at_water is None:
            return []  # water was asked for, but the graph has not been enriched with it
        must = engine.water_rank
    note = None
    walks: list[Walk]
    if places:
        walks, note = _through_places(engine, w, start, places, end, spec)
    elif spec.shape == "loop":
        walks = _loops(engine, w, start, spec.length_km, must, spec.bearing_deg)
    else:
        walks, note = _point_to_point(engine, w, start, end, spec, must)
    if note is not None:
        # The length could not be honoured; judge the routes on everything else.
        spec = spec.model_copy(update={"length_km": None})

    chain_label = (" → ".join(p.label for p in spec.via if p.label) or None) if places else None
    measured_walks = []
    for arcs, required in walks:
        # Trimming may have cut the route off from a place it was meant to reach.
        if not _passes_in_order(engine, arcs, required):
            continue
        via_label = engine.water_label(required[0]) if must is not None else chain_label
        measured = features(engine, w, arcs, spec.length_km)
        if rejection(measured, spec) is None:
            measured_walks.append((arcs, measured, via_label))
    # Without a wanted length, routes are compared by total cost against the cheapest one.
    cheapest = min((m.cost_m for _, m, _ in measured_walks), default=None) if spec.length_km is None else None
    candidates = [
        Candidate(build_path(engine.g, arcs), m, score(m, spec.length_km, cheapest), note, via_label)
        for arcs, m, via_label in measured_walks
    ]
    chosen = select_diverse(candidates, spec.n_candidates, engine.length)
    if places and len(chosen) < spec.n_candidates:
        chosen = select_diverse(candidates, spec.n_candidates, engine.length, VIA_MAX_OVERLAP, chosen=chosen)
    for candidate in chosen:
        candidate.profile = engine.profile(candidate.path.arcs)
    return chosen


def _chain(start: int, places: list[int], end: int) -> list[int]:
    """The nodes of the places to pass, without repeats in a row and without those at the start or the end."""
    chain = [start]
    for place in places:
        if place != chain[-1]:
            chain.append(place)
    if len(chain) > 1 and chain[-1] == end:
        chain.pop()
    return chain[1:]


def _metres(engine: Engine, arcs: np.ndarray) -> float:
    return float(engine.length[arcs >> 1].astype(np.float64).sum())


def _two_way_reachable(engine: Engine, w: np.ndarray, node: int) -> np.ndarray:
    """Nodes that can be reached from `node` and from which `node` can be reached back."""
    there, _, _ = engine.kernel.sssp(node, w, engine.length)
    back, _, _ = engine.kernel.sssp(node, w, engine.length, reverse=True)
    return np.isfinite(there) & np.isfinite(back)


def _with_avoided(engine: Engine, w: np.ndarray, areas: list[AvoidArea]) -> np.ndarray:
    """`w` with every road that touches one of `areas` made dearer."""
    if not areas:
        return w
    inside = np.zeros(engine.g.n_nodes, dtype=bool)
    for area in areas:
        inside[engine.snapper.nodes_within(area.point.lon, area.point.lat, area.radius_m)] = True
    edges = np.flatnonzero(inside[engine.g.edge_u] | inside[engine.g.edge_v])
    w = w.copy()
    w[2 * edges] *= AVOID_FACTOR
    w[2 * edges + 1] *= AVOID_FACTOR
    return w


def _place_node(engine: Engine, place: Point) -> int:
    """The node that stands for a named place: a junction near it if there is one.

    A route cannot pass through a dead end without turning back, so the end of
    a track next to the place is a worse choice than the crossroads beside it.
    """
    nearest, distance = engine.snapper.nearest_node(place.lon, place.lat)
    near = engine.snapper.nodes_within(place.lon, place.lat, max(MIN_SNAP_RADIUS_M, 1.5 * distance + 50.0))
    near = near[engine.degree[near] >= MIN_VIA_DEGREE]
    if len(near) == 0:
        return nearest
    position = np.asarray(engine.g.node_lonlat)[near]
    east = (position[:, 0] - place.lon) * math.cos(math.radians(place.lat))
    return int(near[np.argmin(np.hypot(east, position[:, 1] - place.lat))])


def _through_places(
    engine: Engine, w: np.ndarray, start: int, places: list[int], end: int, spec: RouteSpec
) -> tuple[list[Walk], str | None]:
    """Walks start -> places -> end (a loop when end is the start), plus a note if the length cannot be honoured."""
    walks = _chain_alternatives(engine, w, [start, *places, end], places, 2 * spec.n_candidates)
    if not walks:
        return [], None
    direct_m = _metres(engine, walks[0][0])
    length = spec.length_km
    if length is not None and length.max * 1000 < direct_m:
        return walks, f"через {_places_name(spec.via)} короче {direct_m / 1000:.0f} км не получится"
    if length is not None and length.min * 1000 <= direct_m:
        return walks, None
    if length is None:
        length = LengthKm.around(PLACE_DETOUR * direct_m / 1000, 0.15)
    else:
        walks = []  # all shorter than wanted
    return walks + _detours(engine, w, places[-1], end, length, None, lead=[start, *places[:-1]]), None


def _places_name(via: list[Point]) -> str:
    """The places a route goes through, as a note names them: by their labels when they have any."""
    labels = [f"«{p.label}»" for p in via if p.label]
    if labels:
        return ", ".join(labels)
    return "эти места" if len(via) > 1 else "это место"


def _chain_alternatives(engine: Engine, w: np.ndarray, nodes: list[int], places: list[int], count: int) -> list[Walk]:
    """The best walk through `nodes`, then walks found after making the earlier ones' edges dearer.

    The dearer weights only steer the search; the walks are judged later on `w` itself.
    """
    w = w.copy()
    walks: list[Walk] = []
    for _ in range(count):
        arcs = _walk_via_places(engine, w, nodes, frozenset(places))
        if arcs is None or len(arcs) == 0:
            break
        walks.append((arcs, tuple(places)))
        edges = np.unique(arcs >> 1)
        w[2 * edges] *= ALTERNATIVE_FACTOR
        w[2 * edges + 1] *= ALTERNATIVE_FACTOR
    return walks


def _walk_via_places(engine: Engine, w: np.ndarray, nodes: list[int], places: frozenset[int]) -> np.ndarray | None:
    """A walk through `nodes`, some of which are the named `places`, with stray spurs and small loops cut.

    A place on a dead-end road is ridden to and back from: turning back is
    allowed at the places when the walk is impossible otherwise. Each stretch
    between two places is tidied on its own, so that the spur to a place survives.
    """
    legs = route_legs(engine, w, nodes)
    if legs is None:
        legs = route_legs(engine, w, nodes, may_turn_back_at=places)
    if legs is None:
        return None
    max_loop_m = _max_appendix_m(sum(_metres(engine, leg) for leg in legs))
    stretches: list[np.ndarray] = []
    stretch: list[np.ndarray] = []
    for i, (leg, reached) in enumerate(zip(legs, nodes[1:], strict=True)):
        stretch.append(leg)
        if reached in places or i == len(legs) - 1:
            arcs = np.concatenate(stretch)
            stretches.append(tidy(arcs, engine.arc_tail, engine.arc_head, engine.length, max_loop_m, places))
            stretch = []
    return np.concatenate(stretches) if stretches else np.empty(0, dtype=np.uint32)


def _passes_in_order(engine: Engine, arcs: np.ndarray, required: tuple[int, ...]) -> bool:
    """Whether the walk `arcs` reaches every node of `required`, in that order (its start counts)."""
    if len(arcs) == 0:
        return not required
    wanted = iter(required)
    pending = next(wanted, None)
    for reached in [int(engine.arc_tail[arcs[0]]), *engine.arc_head[arcs].tolist()]:
        if reached == pending:
            pending = next(wanted, None)
    return pending is None


def _nicest(engine: Engine, nice: np.ndarray, nodes: np.ndarray, lon: float, lat: float) -> int:
    """The node on the most pleasant road; among equals, the one nearest to (lon, lat)."""
    position = np.asarray(engine.g.node_lonlat)[nodes]
    east = (position[:, 0] - lon) * math.cos(math.radians(lat))
    distance = np.hypot(east, position[:, 1] - lat)
    order = np.lexsort((nodes, distance, np.round(nice[nodes], 2)))
    return int(nodes[order[0]])


# --- loops -------------------------------------------------------------------


def _loops(
    engine: Engine, w: np.ndarray, start: int, length: LengthKm, must: np.ndarray | None, heading: float | None
) -> list[Walk]:
    """Loop walks from `start`.

    `must` ranks nodes as places the route has to pass (lower is better, inf =
    not such a place); the far corner of each loop is then put on one of them.
    `heading` keeps only the polygons that lie roughly that way from the start.
    """
    nice = engine.node_niceness(w)
    usable = _two_way_reachable(engine, w, start) & (engine.degree >= MIN_VIA_DEGREE)
    bearings = [360.0 * i / N_BEARINGS for i in range(N_BEARINGS)]
    if heading is not None:
        bearings = [b for b in bearings if abs((b - heading + 180) % 360 - 180) <= HEADING_TOLERANCE_DEG]
    attempts = [(k, bearing) for k in VIA_COUNTS for bearing in bearings]

    def attempt(shape: tuple[int, float]) -> Walk | None:
        return _fit_loop(engine, w, nice, usable, must, start, length, *shape)

    with ThreadPoolExecutor(N_WORKERS) as pool:
        return [walk for walk in pool.map(attempt, attempts) if walk is not None]


def _max_appendix_m(target_m: float) -> float:
    """Longest closed sub-walk that is cut out of a route as an appendix."""
    return max(MIN_APPENDIX_M, APPENDIX_SHARE * target_m)


def _fit_loop(
    engine: Engine,
    w: np.ndarray,
    nice: np.ndarray,
    usable: np.ndarray,
    must: np.ndarray | None,
    start: int,
    length: LengthKm,
    n_vias: int,
    bearing: float,
) -> Walk | None:
    """Resize one polygon until the walk around it has an acceptable length."""
    target_m = length.target * 1000
    max_loop_m = _max_appendix_m(target_m)
    scale = 1.0
    for _ in range(MAX_FIT_ITERATIONS):
        vias = _polygon_vias(engine, nice, usable, must, start, target_m * scale / DETOUR, n_vias, bearing)
        if vias is None:
            return None
        # The corner opposite the start is the one placed at a wanted node.
        opposite = vias[(n_vias + 1) // 2 - 1]
        keep = frozenset({opposite}) if must is not None else frozenset()
        arcs = route_through(engine, w, [start, *vias, start])
        if arcs is None:
            return None
        arcs = tidy(arcs, engine.arc_tail, engine.arc_head, engine.length, max_loop_m, keep)
        metres = _metres(engine, arcs)
        if metres == 0.0:
            return None
        if length.min * 1000 <= metres <= length.max * 1000:
            return arcs, ((opposite,) if must is not None else ())
        scale *= target_m / metres
    return None


def _polygon_vias(
    engine: Engine,
    nice: np.ndarray,
    usable: np.ndarray,
    must: np.ndarray | None,
    start: int,
    perimeter_m: float,
    n_vias: int,
    bearing: float,
) -> list[int] | None:
    """Via nodes near the corners of a regular polygon that has `start` as a corner.

    The polygon's centre lies `bearing` degrees from the start. With one via
    node the "polygon" is a there-and-back line. Returns None when a corner
    has no usable junction nearby (a lake, the edge of the map).

    With `must`, the corner farthest from the start has to be one of those
    nodes; it is looked for in a wider circle, since such places are sparse.
    """
    lon, lat = (float(value) for value in engine.g.node_lonlat[start])
    corners = n_vias + 1
    side = perimeter_m / corners
    radius = side / (2 * math.sin(math.pi / corners))
    centre = destination(lon, lat, bearing, radius)
    vias = []
    for i in range(1, corners):
        corner = destination(*centre, bearing + 180 + i * 360 / corners, radius)
        if must is not None and i == corners // 2:
            near = engine.snapper.nodes_within(*corner, max(MIN_MUST_RADIUS_M, MUST_RADIUS_SHARE * side))
            near = near[usable[near] & np.isfinite(must[near])]
            near = near[must[near] == must[near].min()] if len(near) else near
        else:
            near = engine.snapper.nodes_within(*corner, max(MIN_SNAP_RADIUS_M, SNAP_RADIUS_SHARE * side))
            near = near[usable[near]]
        if len(near) == 0:
            return None
        vias.append(_nicest(engine, nice, near, *corner))
    return vias


# --- point to point ----------------------------------------------------------


def _point_to_point(
    engine: Engine, w: np.ndarray, start: int, end: int, spec: RouteSpec, must: np.ndarray | None
) -> tuple[list[Walk], str | None]:
    """Walks from start to end, plus a note if the wanted length cannot be honoured."""
    direct = engine.kernel.route(start, end, w)
    if direct is None:
        return [], None
    direct_m = _metres(engine, direct[0])
    length = spec.length_km
    n_alternatives = 2 * spec.n_candidates

    if must is not None:
        # The via node has to be a wanted place; without a length, allow a modest detour for it.
        via_length = length if length and length.max * 1000 >= direct_m else LengthKm.around(1.3 * direct_m / 1000, 0.25)
        return _detours(engine, w, start, end, via_length, must), None
    if length is None:
        return _alternatives(engine, w, start, end, n_alternatives), None
    if length.max * 1000 < direct_m:
        note = f"короче {direct_m / 1000:.0f} км не получится: это самый короткий подходящий путь"
        return _alternatives(engine, w, start, end, n_alternatives), note
    if length.target * 1000 < MIN_DETOUR_FOR_VIA * direct_m:
        return _alternatives(engine, w, start, end, n_alternatives), None
    return _detours(engine, w, start, end, length, None), None


def _alternatives(engine: Engine, w: np.ndarray, start: int, end: int, count: int) -> list[Walk]:
    """The best route, then routes found after making the earlier ones' edges dearer."""
    w = w.copy()
    walks = []
    for _ in range(count):
        found = engine.kernel.route(start, end, w)
        if found is None:
            break
        arcs = found[0]
        walks.append((arcs, ()))
        edges = np.unique(arcs >> 1)
        w[2 * edges] *= ALTERNATIVE_FACTOR
        w[2 * edges + 1] *= ALTERNATIVE_FACTOR
    return walks


def _detours(
    engine: Engine,
    w: np.ndarray,
    start: int,
    end: int,
    length: LengthKm,
    must: np.ndarray | None,
    lead: list[int] | None = None,
) -> list[Walk]:
    """Walks start -> via -> end, one via node per compass sector around the midpoint.

    With `lead`, the walks are lead[0] -> ... -> lead[-1] -> start -> via -> end:
    a chain of places lengthened on its last leg only. `length` is then that of
    the whole walk, and `lead[1:]` and `start` are the places the walk must pass.
    """
    nice = engine.node_niceness(w)
    places = (*lead[1:], start) if lead else ()
    lead_m = 0.0
    if lead:
        legs = route_legs(engine, w, [*lead, start], may_turn_back_at=frozenset(places))
        if legs is None:
            return []
        lead_m = sum(_metres(engine, leg) for leg in legs)
    cost_from, metres_from, _ = engine.kernel.sssp(start, w, engine.length)
    cost_to, metres_to, _ = engine.kernel.sssp(end, w, engine.length, reverse=True)
    usable = np.isfinite(cost_from) & np.isfinite(cost_to) & (engine.degree >= MIN_VIA_DEGREE)
    if must is not None:
        usable &= np.isfinite(must)
    through_m = np.where(usable, lead_m + metres_from + metres_to, np.inf)

    lonlat = np.asarray(engine.g.node_lonlat)
    mid_lon, mid_lat = (lonlat[start] + lonlat[end]) / 2
    east = (lonlat[:, 0] - mid_lon) * math.cos(math.radians(mid_lat))
    sector = (np.degrees(np.arctan2(east, lonlat[:, 1] - mid_lat)) % 360 // (360 / N_BEARINGS)).astype(np.int8)
    target_m = length.target * 1000
    max_appendix_m = _max_appendix_m(target_m)

    def attempt(which: int) -> Walk | None:
        in_sector = sector == which
        want = target_m
        for _ in range(MAX_FIT_ITERATIONS):
            off_target = np.where(in_sector, np.abs(through_m - want), np.inf)
            if must is not None:
                # Wanted places are sparse: take the one whose detour length fits best.
                via = int(np.argmin(off_target))
                if not np.isfinite(off_target[via]):
                    return None
            else:
                band = np.flatnonzero(off_target <= P2P_BAND_SHARE * target_m)
                if len(band) == 0:
                    return None
                via = _nicest(engine, nice, band, mid_lon, mid_lat)
            if lead:
                arcs = _walk_via_places(engine, w, [*lead, start, via, end], frozenset(places))
            else:
                arcs = route_through(engine, w, [start, via, end])
                if arcs is not None:
                    keep = frozenset({via}) if must is not None else frozenset()
                    arcs = tidy(arcs, engine.arc_tail, engine.arc_head, engine.length, max_appendix_m, keep)
            if arcs is None:
                return None
            metres = _metres(engine, arcs)
            if length.min * 1000 <= metres <= length.max * 1000:
                return arcs, (places if lead else (via,) if must is not None else ())
            want *= target_m / metres
        return None

    with ThreadPoolExecutor(N_WORKERS) as pool:
        return [walk for walk in pool.map(attempt, range(N_BEARINGS)) if walk is not None]
