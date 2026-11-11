"""Generate candidate routes for a RouteSpec.

A loop is built as a walk start -> via nodes -> start, where the via nodes sit
on the corners of a regular polygon that has the start as one corner. Many
polygons are tried (different corner counts and orientations); each is
resized until the walk has the wanted length. A point-to-point route with a
target length goes through one via node chosen so that the way there plus the
way on add up to the target.

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
from rpa.routing.trim import trim_spurs
from rpa.routing.walk import route_through

# First guess at how much longer roads are than the straight line between two points.
DETOUR = 1.3
VIA_COUNTS = (1, 2, 3)
N_BEARINGS = 12
MAX_FIT_ITERATIONS = 5
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
N_WORKERS = min(8, os.cpu_count() or 1)

# The arcs of a walk and the node it was asked to pass (None if there was no such wish).
Walk = tuple[np.ndarray, int | None]


def generate(engine: Engine, spec: RouteSpec) -> list[Candidate]:
    """Up to `spec.n_candidates` distinct routes, best first; empty if nothing fits."""
    w = _with_avoided(engine, engine.weights(spec.prefs), spec.avoid)
    start, _ = engine.snapper.nearest_node(spec.start.lon, spec.start.lat)
    must = None
    if spec.via_category == "water":
        if engine.at_water is None:
            return []  # water was asked for, but the graph has not been enriched with it
        must = engine.water_rank
    note = None
    end = start if spec.shape == "loop" else engine.snapper.nearest_node(spec.end.lon, spec.end.lat)[0]
    walks: list[Walk]
    if spec.via is not None:
        walks, note = _through_place(engine, w, start, _place_node(engine, spec.via), end, spec)
    elif spec.shape == "loop":
        walks = _loops(engine, w, start, spec.length_km, must, spec.bearing_deg)
    else:
        walks, note = _point_to_point(engine, w, start, end, spec, must)
    if note is not None:
        # The length could not be honoured; judge the routes on everything else.
        spec = spec.model_copy(update={"length_km": None})

    candidates = []
    for arcs, via in walks:
        via_label = None
        if must is not None or spec.via is not None:
            # Trimming a spur may have cut the route off from the place it was meant to reach.
            if via is None or via not in engine.arc_head[arcs]:
                continue
            via_label = spec.via.label if spec.via is not None else engine.water_label(via)
        measured = features(engine, w, arcs, spec.length_km)
        if rejection(measured, spec) is None:
            path = build_path(engine.g, arcs)
            candidates.append(Candidate(path, measured, score(measured, spec.length_km), note, via_label))
    chosen = select_diverse(candidates, spec.n_candidates, engine.length)
    for candidate in chosen:
        candidate.profile = engine.profile(candidate.path.arcs)
    return chosen


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


def _through_place(
    engine: Engine, w: np.ndarray, start: int, place: int, end: int, spec: RouteSpec
) -> tuple[list[Walk], str | None]:
    """Walks start -> place -> end (a loop when end is the start), plus a note if the length cannot be honoured."""
    direct = route_through(engine, w, [start, place, end])
    if direct is None:
        return [], None
    direct_m = _metres(engine, direct)
    length = spec.length_km
    if length is not None and length.max * 1000 < direct_m:
        name = f"«{spec.via.label}»" if spec.via.label else "это место"
        return [(direct, place)], f"через {name} короче {direct_m / 1000:.0f} км не получится"
    walks: list[Walk] = []
    if length is None or length.min * 1000 <= direct_m:
        walks.append((direct, place))
    if length is None:
        length = LengthKm.around(PLACE_DETOUR * direct_m / 1000, 0.15)
    return walks + _detours(engine, w, place, end, length, None, before=start), None


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
    scale = 1.0
    for _ in range(MAX_FIT_ITERATIONS):
        vias = _polygon_vias(engine, nice, usable, must, start, target_m * scale / DETOUR, n_vias, bearing)
        if vias is None:
            return None
        arcs = route_through(engine, w, [start, *vias, start])
        if arcs is None:
            return None
        arcs = trim_spurs(arcs)
        metres = _metres(engine, arcs)
        if metres == 0.0:
            return None
        if length.min * 1000 <= metres <= length.max * 1000:
            # The corner opposite the start is the one placed at a wanted node.
            return arcs, (vias[(n_vias + 1) // 2 - 1] if must is not None else None)
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
        walks.append((arcs, None))
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
    before: int | None = None,
) -> list[Walk]:
    """Walks start -> via -> end, one via node per compass sector around the midpoint.

    With `before`, the walks are before -> start -> via -> end and `length` is that
    of the whole walk; `start` is then the place the walk is reported to pass.
    """
    nice = engine.node_niceness(w)
    lead_m = 0.0
    if before is not None:
        lead = engine.kernel.route(before, start, w)
        if lead is None:
            return []
        lead_m = _metres(engine, lead[0])
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
            arcs = route_through(engine, w, [start, via, end] if before is None else [before, start, via, end])
            if arcs is None:
                return None
            arcs = trim_spurs(arcs)
            metres = _metres(engine, arcs)
            if length.min * 1000 <= metres <= length.max * 1000:
                return arcs, (via if before is None else start)
            want *= target_m / metres
        return None

    with ThreadPoolExecutor(N_WORKERS) as pool:
        return [walk for walk in pool.map(attempt, range(N_BEARINGS)) if walk is not None]
