"""Geometry of a ride around an object: the object's polygon, anchor points around it, and a test of
whether a route really circles it.

Pure functions over coordinates; nothing here knows about the road graph. Polygon work is done in
local metres around the object's point.
"""

import math
from collections import Counter

import numpy as np
import shapely
from shapely.geometry import LinearRing, LineString, Polygon
from shapely.geometry.polygon import orient

from rpa.routing.geo import EARTH_RADIUS_M
from rpa.routing.spec import Around

MIN_AROUND_RADIUS_M = 300.0  # a point object is circled at least this widely
ENCLOSED_BORDER_SHARE = 0.9  # of the length of the object's border, this much must be circled
BORDER_SAMPLES = 200  # the border is judged at this many evenly spaced points, no closer than MIN_SAMPLE_STEP_M
MIN_SAMPLE_STEP_M = 3.0
ON_ROUTE_M = 0.01  # a point this close to a route lies on it
_CIRCLE_VERTICES = 16
_M_PER_DEG = EARTH_RADIUS_M * math.pi / 180


def to_local(lonlat, origin: tuple[float, float]) -> np.ndarray:
    """(lon, lat) -> metres east and north of `origin`; takes one pair or an (n, 2) array."""
    lonlat = np.asarray(lonlat, dtype=float)
    scale = np.array([_M_PER_DEG * math.cos(math.radians(origin[1])), _M_PER_DEG])
    return (lonlat - np.asarray(origin, dtype=float)) * scale


def from_local(xy, origin: tuple[float, float]) -> np.ndarray:
    """The inverse of `to_local`."""
    xy = np.asarray(xy, dtype=float)
    scale = np.array([_M_PER_DEG * math.cos(math.radians(origin[1])), _M_PER_DEG])
    return xy / scale + np.asarray(origin, dtype=float)


def _largest(geometry) -> Polygon:
    parts = [part for part in getattr(geometry, "geoms", [geometry]) if isinstance(part, Polygon)]
    return max(parts, key=lambda part: part.area)


def object_polygon(around: Around) -> Polygon:
    """The object in local metres: its outline, or a 16-gon of radius `radius_m` (at least the minimum)."""
    origin = (around.point.lon, around.point.lat)
    if len(around.outline) >= 3:
        polygon = Polygon(to_local(around.outline, origin))
        if not polygon.is_valid:  # simplification can make a border touch itself
            polygon = _largest(shapely.make_valid(polygon))
        return polygon
    radius = max(around.radius_m, MIN_AROUND_RADIUS_M)
    angles = np.linspace(0, 2 * math.pi, _CIRCLE_VERTICES, endpoint=False)
    return Polygon(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))


def ring_border(polygon: Polygon, offset_m: float) -> LinearRing:
    """The outer border of the polygon grown by `offset_m`, counter-clockwise.

    Growing closes bays narrower than twice the offset, so a ring along it does not lead into them.
    """
    return orient(_largest(polygon.buffer(offset_m)), sign=1.0).exterior


def ring_anchors(polygon: Polygon, offset_m: float, n: int, phase: float) -> np.ndarray:
    """`n` points at equal steps of length along `ring_border`, counter-clockwise.

    `phase` in 0..1 shifts the first point by that share of the whole length.
    """
    border = ring_border(polygon, offset_m)
    return np.array(
        [border.interpolate(((phase + i / n) % 1.0) * border.length, normalized=False).coords[0] for i in range(n)]
    )


def winding_number(xy, point) -> int:
    """How many times the polyline turns around `point`, counter-clockwise positive.

    The polyline is taken as closed: its ends are joined.
    """
    vectors = np.asarray(xy, dtype=float) - np.asarray(point, dtype=float)
    following = np.roll(vectors, -1, axis=0)
    cross = vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0]
    dot = (vectors * following).sum(axis=1)
    return round(float(np.arctan2(cross, dot).sum()) / (2 * math.pi))


def encloses(coords_lonlat, around: Around) -> bool:
    """Whether a route goes once around the object: one full turn about its inner point, and the same
    turn about at least `ENCLOSED_BORDER_SHARE` of the border's length (so cutting a corner fails).
    The border is judged at evenly spaced points, not at the outline's samples: a wiggly shore has many
    of them and a straight one few.

    A point the route runs through turns neither way about itself, so it cannot be judged: a border point
    on the route counts as circled, and when the inner point is on the route, the turn the samples off
    the route mostly share stands in for it.
    """
    origin = (around.point.lon, around.point.lat)
    route = to_local(coords_lonlat, origin)
    if len(route) < 3:
        return False
    line = LineString(np.vstack([route, route[:1]]))  # closed, as `winding_number` takes it
    polygon = object_polygon(around)
    border = polygon.exterior
    count = int(np.clip(border.length / MIN_SAMPLE_STEP_M, 8, BORDER_SAMPLES))
    samples = shapely.get_coordinates(shapely.line_interpolate_point(border, np.arange(count) * border.length / count))
    on_route = shapely.distance(line, shapely.points(samples)) <= ON_ROUTE_M
    turns_about = [0 if on else winding_number(route, sample) for sample, on in zip(samples, on_route, strict=True)]
    inner = polygon.representative_point()
    if line.distance(inner) > ON_ROUTE_M:
        turns = winding_number(route, inner.coords[0])
    else:
        off = [t for t, on in zip(turns_about, on_route, strict=True) if not on]
        if not off:
            return False
        turns = Counter(off).most_common(1)[0][0]
    if abs(turns) != 1:
        return False
    same = sum(on or t == turns for t, on in zip(turns_about, on_route, strict=True))
    return same >= ENCLOSED_BORDER_SHARE * len(samples)
