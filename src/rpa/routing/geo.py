"""Small spherical-geometry helpers."""

import math

EARTH_RADIUS_M = 6_371_008.8


def destination(lon: float, lat: float, bearing: float, distance_m: float) -> tuple[float, float]:
    """Point reached by travelling `distance_m` from (lon, lat) on `bearing` degrees (0 = north, 90 = east)."""
    angular = distance_m / EARTH_RADIUS_M
    lat1, lon1, theta = math.radians(lat), math.radians(lon), math.radians(bearing)
    lat2 = math.asin(math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(angular) * math.cos(theta))
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(angular) * math.cos(lat1), math.cos(angular) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lon2), math.degrees(lat2)


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial bearing from point 1 to point 2, in degrees within 0..360."""
    phi1, phi2, dlon = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0
