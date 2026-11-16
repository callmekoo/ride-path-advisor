"""Geometry of a ride around an object: the object's polygon, anchor points around it, and a test of
whether a route really circles it.

Pure functions over coordinates; nothing here knows about the road graph. Polygon work is done in
local metres around the object's point.
"""

import math

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from rpa.routing.geo import EARTH_RADIUS_M
from rpa.routing.spec import Around

MIN_AROUND_RADIUS_M = 300.0  # a point object is circled at least this widely
ENCLOSED_VERTEX_SHARE = 0.9  # of the object's outline vertices, this many must be circled
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


def ring_anchors(polygon: Polygon, offset_m: float, n: int, phase: float) -> np.ndarray:
    """`n` points at equal steps of length along the outer border of the polygon grown by `offset_m`.

    Counter-clockwise; `phase` in 0..1 shifts the first point by that share of the whole length.
    Growing closes bays narrower than twice the offset, so anchors do not lead a route into them.
    """
    border = orient(_largest(polygon.buffer(offset_m)), sign=1.0).exterior
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
    turn about at least `ENCLOSED_VERTEX_SHARE` of its outline's vertices (so cutting a corner fails)."""
    origin = (around.point.lon, around.point.lat)
    route = to_local(coords_lonlat, origin)
    polygon = object_polygon(around)
    turns = winding_number(route, polygon.representative_point().coords[0])
    if abs(turns) != 1:
        return False
    vertices = np.array(polygon.exterior.coords[:-1])
    same = sum(winding_number(route, vertex) == turns for vertex in vertices)
    return same >= ENCLOSED_VERTEX_SHARE * len(vertices)
