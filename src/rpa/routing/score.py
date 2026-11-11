"""Describe a finished route with a few plain numbers, then rank and filter by them.

Arc weights steer the search one road at a time. The score looks at the whole
route and sees what a single arc cannot: retraced road, missed length, how
twisty the result is. The numbers here are starting points for tuning.
"""

from dataclasses import dataclass

import numpy as np

from rpa.enrich.rasters import climb_m
from rpa.osm.tags import Highway, Surface
from rpa.routing.engine import Engine
from rpa.routing.path import RoutePath
from rpa.routing.spec import LengthKm, RouteSpec

_MAJOR = [Highway.SECONDARY, Highway.PRIMARY, Highway.TRUNK]
_UNPAVED = [Surface.LIKELY_UNPAVED, Surface.UNPAVED, Surface.BAD]

TURN_ANGLE_DEG = 45.0
MAX_REPEAT_SHARE = 0.3
MAX_UNPAVED_SHARE_ON_PAVED = 0.10


@dataclass(frozen=True)
class Features:
    length_m: float
    length_error: float  # |length - target| / target; 0 when no length was asked for
    repeat_share: float  # metres ridden on an edge beyond its first traversal / total
    cost_ratio: float  # preference cost / length; 1.0 means ideal roads throughout
    turns_per_10km: float
    major_share: float  # secondary, primary and trunk roads
    unpaved_share: float  # likely unpaved, unpaved and bad surfaces
    ridden_share: float = 0.0  # roads present in the ride history; 0 when there is none
    forest_share: float = 0.0  # 0 when the graph has no land cover
    ascent_m: float = 0.0  # 0 when the graph has no elevation


@dataclass
class Candidate:
    path: RoutePath
    features: Features
    score: float
    note: str | None  # shown to the user, e.g. why the length differs from the request
    via_label: str | None = None  # the place the route was asked to pass, e.g. the lake it reaches
    profile: tuple[np.ndarray, np.ndarray] | None = None  # distance and elevation in metres along the route


def features(engine: Engine, w: np.ndarray, arcs: np.ndarray, length: LengthKm | None) -> Features:
    """Measure the route `arcs`; `w` are the request's base weights (no reuse penalties)."""
    edges = arcs >> 1
    edge_m = engine.length[edges].astype(np.float64)
    total = float(edge_m.sum())
    if total == 0.0:
        return Features(0.0, 1.0 if length else 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    used, times = np.unique(edges, return_counts=True)
    repeat_m = float((engine.length[used].astype(np.float64) * (times - 1)).sum())
    highway = np.asarray(engine.g.edge_highway)[edges]
    surface = np.asarray(engine.g.edge_surface)[edges]
    ridden_m = float(edge_m[engine.rides[edges] > 0].sum()) if engine.rides is not None else 0.0
    forest_m = float((edge_m * engine.forest[edges]).sum()) if engine.forest is not None else 0.0
    profile = engine.profile(arcs)
    return Features(
        length_m=total,
        length_error=abs(total / 1000 - length.target) / length.target if length else 0.0,
        repeat_share=repeat_m / total,
        cost_ratio=float(w[arcs].astype(np.float64).sum()) / total,
        turns_per_10km=_count_turns(engine, arcs) / (total / 10_000),
        major_share=float(edge_m[np.isin(highway, _MAJOR)].sum()) / total,
        unpaved_share=float(edge_m[np.isin(surface, _UNPAVED)].sum()) / total,
        ridden_share=ridden_m / total,
        forest_share=forest_m / total,
        ascent_m=climb_m(profile[1]) if profile is not None else 0.0,
    )


def _count_turns(engine: Engine, arcs: np.ndarray) -> int:
    """Junctions where the route changes heading by more than TURN_ANGLE_DEG."""
    if len(arcs) < 2:
        return 0
    g = engine.g
    edges = (arcs >> 1).astype(np.int64)
    backward = (arcs & 1).astype(bool)
    first = g.geom_off[edges].astype(np.int64)
    last = g.geom_off[edges + 1].astype(np.int64) - 1
    xy = g.geom_xy

    def heading(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        delta = (xy[end] - xy[start]).astype(np.float64)
        east = delta[:, 0] * np.cos(np.radians(xy[start, 1] / 1e7))
        return np.degrees(np.arctan2(east, delta[:, 1]))

    # Heading on leaving an arc's first point and on arriving at its last one.
    leaving = heading(np.where(backward, last, first), np.where(backward, last - 1, first + 1))
    arriving = heading(np.where(backward, first + 1, last - 1), np.where(backward, first, last))
    change = np.abs((leaving[1:] - arriving[:-1] + 180) % 360 - 180)
    at_junction = engine.degree[engine.arc_head[arcs[:-1]]] >= 3
    return int(((change > TURN_ANGLE_DEG) & at_junction).sum())


def score(f: Features, length: LengthKm | None) -> float:
    """Higher is better; 100 is a route with nothing to complain about."""
    value = 100.0
    if length:
        half_band = max(length.target - length.min, length.max - length.target, 1e-9)
        value -= 20.0 * min(abs(f.length_m / 1000 - length.target) / half_band, 1.0)
    value -= 80.0 * f.repeat_share
    # 1 - 1/ratio grows with the ratio without ever saturating, so dearer routes always rank lower.
    value -= 30.0 * (1.0 - 1.0 / max(f.cost_ratio, 1.0))
    value -= 0.5 * max(0.0, f.turns_per_10km - 15.0)
    return max(value, 0.0)


def rejection(f: Features, spec: RouteSpec) -> str | None:
    """Why this route must not be offered for `spec`, or None if it may."""
    km = f.length_m / 1000
    length = spec.length_km
    if length and not length.min <= km <= length.max:
        return f"длина {km:.1f} км вне диапазона {length.min:.1f}–{length.max:.1f} км"
    if f.repeat_share > MAX_REPEAT_SHARE:
        return f"слишком много повторов: {f.repeat_share:.0%}"
    if spec.prefs.surface == "paved" and f.unpaved_share > MAX_UNPAVED_SHARE_ON_PAVED:
        return f"слишком много грунта для шоссейного маршрута: {f.unpaved_share:.0%}"
    if not np.isfinite(f.cost_ratio):
        return "маршрут проходит по запрещённому участку"
    return None
