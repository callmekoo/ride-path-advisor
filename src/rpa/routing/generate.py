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
from rpa.routing.spec import LengthKm, RouteSpec
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
N_WORKERS = min(8, os.cpu_count() or 1)

# The arcs of a walk and the node it was asked to pass (None if there was no such wish).
Walk = tuple[np.ndarray, int | None]


def generate(engine: Engine, spec: RouteSpec) -> list[Candidate]:
    """Up to `spec.n_candidates` distinct routes, best first; empty if nothing fits."""
    w = engine.weights(spec.prefs)
    start, _ = engine.snapper.nearest_node(spec.start.lon, spec.start.lat)
    must = None
    if spec.via_category == "water":
        if engine.at_water is None:
            return []  # water was asked for, but the graph has not been enriched with it
        must = engine.water_rank
    note = None
    if spec.shape == "loop":
        walks: list[Walk] = _loops(engine, w, start, spec.length_km, must)
    else:
        end, _ = engine.snapper.nearest_node(spec.end.lon, spec.end.lat)
        walks, note = _point_to_point(engine, w, start, end, spec, must)
    if note is not None:
        # The length could not be honoured; judge the routes on everything else.
        spec = spec.model_copy(update={"length_km": None})

    candidates = []
    for arcs, via in walks:
        via_label = None
        if must is not None:
            # Trimming a spur may have cut the route off from the place it was meant to reach.
            if via is None or via not in engine.arc_head[arcs]:
                continue
            via_label = engine.water_label(via)
        measured = features(engine, w, arcs, spec.length_km)
        if rejection(measured, spec) is None:
            path = build_path(engine.g, arcs)
            candidates.append(Candidate(path, measured, score(measured, spec.length_km), note, via_label))
    return select_diverse(candidates, spec.n_candidates, engine.length)


def _metres(engine: Engine, arcs: np.ndarray) -> float:
    return float(engine.length[arcs >> 1].astype(np.float64).sum())


def _two_way_reachable(engine: Engine, w: np.ndarray, node: int) -> np.ndarray:
    """Nodes that can be reached from `node` and from which `node` can be reached back."""
    there, _, _ = engine.kernel.sssp(node, w, engine.length)
    back, _, _ = engine.kernel.sssp(node, w, engine.length, reverse=True)
    return np.isfinite(there) & np.isfinite(back)


def _nicest(engine: Engine, nice: np.ndarray, nodes: np.ndarray, lon: float, lat: float) -> int:
    """The node on the most pleasant road; among equals, the one nearest to (lon, lat)."""
    position = np.asarray(engine.g.node_lonlat)[nodes]
    east = (position[:, 0] - lon) * math.cos(math.radians(lat))
    distance = np.hypot(east, position[:, 1] - lat)
    order = np.lexsort((nodes, distance, np.round(nice[nodes], 2)))
    return int(nodes[order[0]])


# --- loops -------------------------------------------------------------------


def _loops(engine: Engine, w: np.ndarray, start: int, length: LengthKm, must: np.ndarray | None) -> list[Walk]:
    """Loop walks from `start`.

    `must` ranks nodes as places the route has to pass (lower is better, inf =
    not such a place); the far corner of each loop is then put on one of them.
    """
    nice = engine.node_niceness(w)
    usable = _two_way_reachable(engine, w, start) & (engine.degree >= MIN_VIA_DEGREE)
    attempts = [(k, 360.0 * i / N_BEARINGS) for k in VIA_COUNTS for i in range(N_BEARINGS)]

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
    engine: Engine, w: np.ndarray, start: int, end: int, length: LengthKm, must: np.ndarray | None
) -> list[Walk]:
    """Walks start -> via -> end, one via node per compass sector around the midpoint."""
    nice = engine.node_niceness(w)
    cost_from, metres_from, _ = engine.kernel.sssp(start, w, engine.length)
    cost_to, metres_to, _ = engine.kernel.sssp(end, w, engine.length, reverse=True)
    usable = np.isfinite(cost_from) & np.isfinite(cost_to) & (engine.degree >= MIN_VIA_DEGREE)
    if must is not None:
        usable &= np.isfinite(must)
    through_m = np.where(usable, metres_from + metres_to, np.inf)

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
            arcs = route_through(engine, w, [start, via, end])
            if arcs is None:
                return None
            arcs = trim_spurs(arcs)
            metres = _metres(engine, arcs)
            if length.min * 1000 <= metres <= length.max * 1000:
                return arcs, via
            want *= target_m / metres
        return None

    with ThreadPoolExecutor(N_WORKERS) as pool:
        return [walk for walk in pool.map(attempt, range(N_BEARINGS)) if walk is not None]
