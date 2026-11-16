"""Generate candidate routes for a RouteSpec.

A loop is built as a walk start -> via nodes -> start, where the via nodes sit
on the corners of a regular polygon that has the start as one corner. Many
polygons are tried (different corner counts and orientations); each is
resized until the walk has the wanted length. A loop without a wanted length
is offered in three sizes, one route each. A point-to-point route with a
target length goes through one via node chosen so that the way there plus the
way on add up to the target. A route through named places follows them in
the given order and is lengthened, when it has to be, on its last leg. A ride
around an object goes through anchor points on a ring laid around it and is
kept only if it really circles the object.

All attempts run in a thread pool: the search kernel releases the GIL.
"""

import math
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Polygon

from rpa.routing import ring as rings
from rpa.routing.diversity import select_by_size, select_diverse
from rpa.routing.engine import Engine
from rpa.routing.geo import destination
from rpa.routing.path import RoutePath, build_path
from rpa.routing.score import Candidate, features, rejection, score
from rpa.routing.spec import Around, AvoidArea, LengthKm, Point, RouteSpec
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
# A loop without a wanted length: each of the engine's three sizes is tried in this many directions,
# turned a little from size to size so that the sizes do not all follow the same roads out,
# and is accepted within this share of the size.
N_SIZE_BEARINGS = 6
SIZE_BEARING_SHIFT_DEG = 10.0
LOOP_SIZE_TOLERANCE = 0.15
# A ride around an object: anchor points about this far apart along a ring around it, within these counts.
AROUND_SPACING_M = 4000.0
MIN_ANCHORS, MAX_ANCHORS = 4, 8
# The ring is tried at these shares of the object's equivalent radius from its border, at least
# MIN_AROUND_OFFSET_M from it and from each other.
AROUND_OFFSETS = (0.05, 0.25, 0.6)
MIN_AROUND_OFFSET_M = 150.0
# Fewer anchors with a road nearby, or a stretch of ring without one longer than this share, and the
# ring cannot be ridden on that side.
MIN_RING_ANCHORS = 3
MAX_RING_GAP = 0.4
# Without a wanted length, a ride around an object is scored as if asked for the shortest ride found,
# accepting up to this much more.
AROUND_SLACK = (0.9, 2.0)
N_WORKERS = min(8, os.cpu_count() or 1)

# The arcs of a walk and the nodes it was asked to pass, in order (empty if there was no such wish).
Walk = tuple[np.ndarray, tuple[int, ...]]


def generate(engine: Engine, spec: RouteSpec) -> list[Candidate]:
    """Up to `spec.n_candidates` distinct routes; empty if nothing fits.

    The strictly distinct picks come first, best first, then the looser refill when a route through
    places has too few. A loop with neither a length nor places gives one route per size instead,
    short to long by measured length.
    """
    w = _with_avoided(engine, engine.weights(spec.prefs), spec.avoid)
    start, _ = engine.snapper.nearest_node(spec.start.lon, spec.start.lat)
    end = start if spec.shape == "loop" else engine.snapper.nearest_node(spec.end.lon, spec.end.lat)[0]
    places = _chain(start, [_place_node(engine, p) for p in spec.via], end)
    must = None
    if spec.via_category == "water":
        if engine.at_water is None:
            return []  # water was asked for, but the graph has not been enriched with it
        must = engine.water_rank
    note = None
    walks: list[Walk]
    if places:
        walks, note = _through_places(engine, w, start, places, end, spec)
    elif spec.around is not None:
        return _with_profiles(engine, _around_candidates(engine, w, start, spec))
    elif spec.shape == "loop":
        groups = _loops(engine, w, start, spec.length_km, must, spec.bearing_deg)
        if spec.length_km is None:
            return _with_profiles(engine, _one_per_size(engine, w, groups, spec))
        walks = groups[0][1]
    else:
        walks, note = _point_to_point(engine, w, start, end, spec, must)
    if note is not None:
        # The length could not be honoured; judge the routes on everything else.
        spec = spec.model_copy(update={"length_km": None})

    candidates = _candidates(engine, w, walks, spec, note)
    chosen = select_diverse(candidates, spec.n_candidates, engine.length)
    if places and len(chosen) < spec.n_candidates:
        chosen = select_diverse(candidates, spec.n_candidates, engine.length, VIA_MAX_OVERLAP, chosen=chosen)
    return _with_profiles(engine, chosen)


def _size_tolerances(sizes: tuple[float, ...]) -> list[float]:
    """How far from each size a loop may be: LOOP_SIZE_TOLERANCE, less where neighbouring sizes are close.

    Sizes from the ride history may be only 20 % apart; their bands must not overlap, or the "short"
    loop could come out longer than the "medium" one.
    """
    tolerances = []
    for i, size in enumerate(sizes):
        gaps = [abs(other - size) / (other + size) for other in sizes[max(i - 1, 0) : i + 2] if other != size]
        tolerances.append(min(LOOP_SIZE_TOLERANCE, *gaps))
    return tolerances


# A place found by name is labelled with its kind, "Двуреченск (село)"; in a title the kind is noise.
_TRAILING_KIND = re.compile(r"\s*\([^()]*\)\s*$")


def chain_title(via: list[Point]) -> str | None:
    """The places a route passes, in riding order, as «через: A, B» — «A → B» would read as a route from A to B."""
    labels = [_TRAILING_KIND.sub("", p.label) or p.label for p in via if p.label]
    return "через: " + ", ".join(labels) if labels else None


def around_title(around: Around) -> str:
    """The object a ride goes around, as «вокруг: X»."""
    label = around.point.label
    return "вокруг: " + ((_TRAILING_KIND.sub("", label) or label) if label else "объект")


def _candidates(engine: Engine, w: np.ndarray, walks: list[Walk], spec: RouteSpec, note: str | None) -> list[Candidate]:
    """The walks that pass their places and are not rejected, measured and scored against `spec`."""
    chain_label = chain_title(spec.via)
    measured_walks = []
    for arcs, required in walks:
        # Trimming may have cut the route off from a place it was meant to reach.
        if not _passes_in_order(engine, arcs, required):
            continue
        if not required:
            via_label = None
        elif spec.via_category == "water":
            via_label = engine.water_label(required[0])
        else:
            via_label = chain_label
        measured = features(engine, w, arcs, spec.length_km)
        if rejection(measured, spec) is None:
            measured_walks.append((arcs, measured, via_label))
    # Without a wanted length, routes are compared by total cost against the cheapest one.
    cheapest = min((m.cost_m for _, m, _ in measured_walks), default=None) if spec.length_km is None else None
    return [
        Candidate(build_path(engine.g, arcs), m, score(m, spec.length_km, cheapest), note, via_label)
        for arcs, m, via_label in measured_walks
    ]


def _one_per_size(
    engine: Engine, w: np.ndarray, groups: list[tuple[LengthKm, list[Walk]]], spec: RouteSpec
) -> list[Candidate]:
    """The best loop of each size, short to long; as many of the shortest sizes as routes are wanted."""
    candidates = [
        _candidates(engine, w, walks, spec.model_copy(update={"length_km": length}), None) for length, walks in groups
    ]
    chosen = select_by_size(candidates, engine.length)
    # Bands of neighbouring sizes do not overlap, so this only guards the order.
    chosen.sort(key=lambda candidate: candidate.path.length_m)
    return chosen[: spec.n_candidates]


def _with_profiles(engine: Engine, chosen: list[Candidate]) -> list[Candidate]:
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


def _reach(engine: Engine, w: np.ndarray, node: int) -> tuple[np.ndarray, np.ndarray]:
    """The cost of getting from `node` to every node, and from every node back to `node`."""
    there, _, _ = engine.kernel.sssp(node, w, engine.length)
    back, _, _ = engine.kernel.sssp(node, w, engine.length, reverse=True)
    return there, back


def _two_way_reachable(engine: Engine, w: np.ndarray, node: int) -> np.ndarray:
    """Nodes that can be reached from `node` and from which `node` can be reached back."""
    there, back = _reach(engine, w, node)
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


def _snap(
    engine: Engine,
    nice: np.ndarray,
    usable: np.ndarray,
    lon: float,
    lat: float,
    radius_m: float,
    keep: Callable[[np.ndarray], np.ndarray] | None = None,
) -> int | None:
    """The nicest usable node within `radius_m` of (lon, lat), or None.

    `keep` narrows the nodes found further: it takes their indices and returns those to choose from.
    """
    near = engine.snapper.nodes_within(lon, lat, radius_m)
    near = near[usable[near]]
    if keep is not None and len(near):
        near = keep(near)
    if len(near) == 0:
        return None
    return _nicest(engine, nice, near, lon, lat)


# --- loops -------------------------------------------------------------------


def _loops(
    engine: Engine, w: np.ndarray, start: int, length: LengthKm | None, must: np.ndarray | None, heading: float | None
) -> list[tuple[LengthKm, list[Walk]]]:
    """Loop walks from `start`, grouped by the length they were fitted to.

    With `length` there is one group. Without it there is one group per size
    of `engine.loop_sizes_km`, short to long, each fitted to about that size.

    `must` ranks nodes as places the route has to pass (lower is better, inf =
    not such a place); the far corner of each loop is then put on one of them.
    `heading` keeps only the polygons that lie roughly that way from the start.
    """
    nice = engine.node_niceness(w)
    usable = _two_way_reachable(engine, w, start) & (engine.degree >= MIN_VIA_DEGREE)
    if length is not None:
        sizes = [(length, [360.0 * i / N_BEARINGS for i in range(N_BEARINGS)])]
    else:
        tolerances = _size_tolerances(engine.loop_sizes_km)
        sizes = [
            (
                LengthKm.around(km, tolerances[s]),
                [360.0 * i / N_SIZE_BEARINGS + SIZE_BEARING_SHIFT_DEG * s for i in range(N_SIZE_BEARINGS)],
            )
            for s, km in enumerate(engine.loop_sizes_km)
        ]
    attempts = [
        (group, size, k, bearing)
        for group, (size, bearings) in enumerate(sizes)
        for k in VIA_COUNTS
        for bearing in bearings
        if heading is None or abs((bearing - heading + 180) % 360 - 180) <= HEADING_TOLERANCE_DEG
    ]

    def attempt(shape: tuple[int, LengthKm, int, float]) -> Walk | None:
        _, size, k, bearing = shape
        return _fit_loop(engine, w, nice, usable, must, start, size, k, bearing)

    groups: list[tuple[LengthKm, list[Walk]]] = [(size, []) for size, _ in sizes]
    with ThreadPoolExecutor(N_WORKERS) as pool:
        for shape, walk in zip(attempts, pool.map(attempt, attempts), strict=True):
            if walk is not None:
                groups[shape[0]][1].append(walk)
    return groups


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

    def best_ranked(near: np.ndarray) -> np.ndarray:
        near = near[np.isfinite(must[near])]
        return near[must[near] == must[near].min()] if len(near) else near

    vias = []
    for i in range(1, corners):
        corner = destination(*centre, bearing + 180 + i * 360 / corners, radius)
        if must is not None and i == corners // 2:
            via = _snap(engine, nice, usable, *corner, max(MIN_MUST_RADIUS_M, MUST_RADIUS_SHARE * side), best_ranked)
        else:
            via = _snap(engine, nice, usable, *corner, max(MIN_SNAP_RADIUS_M, SNAP_RADIUS_SHARE * side))
        if via is None:
            return None
        vias.append(via)
    return vias


# --- around an object --------------------------------------------------------


@dataclass(frozen=True)
class _Ring:
    """What every attempt at a ride around one object shares."""

    around: Around
    origin: tuple[float, float]
    polygon: Polygon  # the object, in local metres around `origin`
    radius_m: float  # of a circle with the object's area
    start: int
    start_xy: tuple[float, float]
    nice: np.ndarray
    usable: np.ndarray
    there: np.ndarray  # cost from the start to each node
    back: np.ndarray  # cost from each node back to the start

    def offsets(self) -> list[float]:
        """The distances from the object's border that the ring is tried at, closest first."""
        offsets: list[float] = []
        for share in AROUND_OFFSETS:
            least = offsets[-1] + MIN_AROUND_OFFSET_M if offsets else MIN_AROUND_OFFSET_M
            offsets.append(max(least, share * self.radius_m))
        return offsets


@dataclass(frozen=True)
class _RingWalk:
    arcs: np.ndarray
    core: np.ndarray  # the arcs of the ring itself, without the way there and back
    offset_m: float  # how far from the object the ring was laid
    path: RoutePath


def _ring(engine: Engine, w: np.ndarray, start: int, around: Around) -> _Ring:
    origin = (around.point.lon, around.point.lat)
    polygon = rings.object_polygon(around)
    there, back = _reach(engine, w, start)
    usable = np.isfinite(there) & np.isfinite(back) & (engine.degree >= MIN_VIA_DEGREE)
    start_xy = rings.to_local(engine.g.node_lonlat[start], origin)
    return _Ring(
        around=around,
        origin=origin,
        polygon=polygon,
        radius_m=math.sqrt(polygon.area / math.pi),
        start=start,
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        nice=engine.node_niceness(w),
        usable=usable,
        there=there,
        back=back,
    )


def _around(engine: Engine, w: np.ndarray, start: int, spec: RouteSpec) -> tuple[list[_RingWalk], str | None]:
    """Rides from `start` once around `spec.around`, as close to it as the roads allow."""
    ring = _ring(engine, w, start, spec.around)
    attempts = [(offset, phase, ccw) for offset in ring.offsets() for phase in (0.0, 0.5) for ccw in (True, False)]

    def attempt(shape: tuple[float, float, bool]) -> _RingWalk | None:
        return _ring_walk(engine, w, ring, *shape)

    with ThreadPoolExecutor(N_WORKERS) as pool:
        return [walk for walk in pool.map(attempt, attempts) if walk is not None], None


def _ring_walk(
    engine: Engine,
    w: np.ndarray,
    ring: _Ring,
    offset_m: float,
    phase: float,
    ccw: bool,
    shift: tuple[float, float] = (0.0, 0.0),
) -> _RingWalk | None:
    """One ride around the object through anchors on a ring `offset_m` from it, or None.

    The anchors stand at equal steps along the ring, the first `phase` of a step from its
    beginning; `ccw` rides them counter-clockwise. `shift` moves the ring off-centre, in metres east
    and north. Every anchor takes the nicest junction near it that lies outside the object and
    beside the anchor along the ring, so that a ring across a bay does not snap to its far shore.
    The ride goes from the start to the anchor that is cheapest to reach and leave from, around,
    and back; a start near the ring takes the place of the anchor nearest to it.
    """
    border = shapely.affinity.translate(rings.ring_border(ring.polygon, offset_m), *shift)
    n = int(np.clip(round(border.length / AROUND_SPACING_M), MIN_ANCHORS, MAX_ANCHORS))
    step = border.length / n
    positions = (np.arange(n) + phase) * step
    anchors = rings.from_local(shapely.get_coordinates(shapely.line_interpolate_point(border, positions)), ring.origin)

    start_point = shapely.Point(ring.start_xy)
    start_slot = None
    if border.distance(start_point) < step / 2:
        start_slot = int(np.argmin(_along(positions, border.project(start_point), border.length)))

    def beside(slot: int):
        def keep(near: np.ndarray) -> np.ndarray:
            xy = rings.to_local(engine.g.node_lonlat[near], ring.origin)
            outside = ~shapely.contains_xy(ring.polygon, xy[:, 0], xy[:, 1])
            along = _along(shapely.line_locate_point(border, shapely.points(xy)), positions[slot], border.length)
            return near[outside & (along <= step / 3)]

        return keep

    slots: list[tuple[int, int]] = []
    for slot in range(n):
        if slot == start_slot:
            slots.append((slot, ring.start))
            continue
        lon, lat = (float(value) for value in anchors[slot])
        node = _snap(engine, ring.nice, ring.usable, lon, lat, max(MIN_SNAP_RADIUS_M, step / 3), beside(slot))
        if node is not None:
            slots.append((slot, node))
    taken = [slot for slot, _ in slots]
    if len(taken) < MIN_RING_ANCHORS or max(np.diff([*taken, taken[0] + n])) > MAX_RING_GAP * n + 1e-9:
        return None

    nodes = [node for _, node in (slots if ccw else slots[::-1])]
    if start_slot is not None:
        at = nodes.index(ring.start)
        order = nodes[at + 1 :] + nodes[:at]
    else:
        at = int(np.argmin([ring.there[nodes[e]] + ring.back[nodes[e - 1]] for e in range(len(nodes))]))
        order = nodes[at:] + nodes[:at]
    walk = [ring.start]
    for node in [*order, ring.start]:
        if node != walk[-1]:
            walk.append(node)
    arcs = route_through(engine, w, walk)
    if arcs is None or len(arcs) == 0:
        return None
    arcs = tidy(arcs, engine.arc_tail, engine.arc_head, engine.length, _max_appendix_m(_metres(engine, arcs)))
    if len(arcs) == 0:
        return None
    path = build_path(engine.g, arcs)
    if not rings.encloses(path.coords, ring.around):
        return None
    core = arcs if start_slot is not None else _ring_core(engine, arcs, order[0], order[-1])
    return _RingWalk(arcs, core, offset_m, path)


def _along(positions: np.ndarray, position: float, length: float) -> np.ndarray:
    """How far each of `positions` is from `position` along a closed line of `length`, either way."""
    gap = np.abs(np.asarray(positions) - position) % length
    return np.minimum(gap, length - gap)


def _ring_core(engine: Engine, arcs: np.ndarray, first: int, last: int) -> np.ndarray:
    """The arcs from leaving the first anchor to arriving at the last one; all of them if that is not found."""
    leaving = np.flatnonzero(engine.arc_tail[arcs] == first)
    arriving = np.flatnonzero(engine.arc_head[arcs] == last)
    if len(leaving) == 0 or len(arriving) == 0 or leaving[0] > arriving[-1]:
        return arcs
    return arcs[leaving[0] : arriving[-1] + 1]


def _around_candidates(engine: Engine, w: np.ndarray, start: int, spec: RouteSpec) -> list[Candidate]:
    """The best distinct rides around `spec.around`; they are told apart by their rings alone."""
    walks, note = _around(engine, w, start, spec)
    judged = spec.model_copy(update={"length_km": None}) if note is not None else spec
    measured = [(walk, features(engine, w, walk.arcs, judged.length_km)) for walk in walks]
    passed = [(walk, m) for walk, m in measured if rejection(m, judged) is None]
    length = judged.length_km
    if length is None and passed:
        # Without a wanted length the ride hugs the object: longer ones lose against the shortest.
        km = min(m.length_m for _, m in passed) / 1000
        length = LengthKm(target=km, min=AROUND_SLACK[0] * km, max=AROUND_SLACK[1] * km)
    label = around_title(spec.around)
    candidates = [Candidate(walk.path, m, score(m, length), note, label, core_arcs=walk.core) for walk, m in passed]
    return select_diverse(candidates, spec.n_candidates, engine.length)


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
