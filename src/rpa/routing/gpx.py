"""Write a route as a GPX 1.1 track, the form bike computers and phone apps import."""

from xml.sax.saxutils import escape

import numpy as np

from rpa.tracks.clean import step_m

_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<gpx version="1.1" creator="ride-path-advisor" xmlns="http://www.topografix.com/GPX/1/1">\n'
)


def to_gpx(coords: np.ndarray, name: str, profile: tuple[np.ndarray, np.ndarray] | None) -> str:
    """GPX text for the line `coords` (f64[P, 2], lon/lat).

    `profile` is (distance in metres, elevation in metres) along the same route, sampled
    independently of `coords`; each point takes the elevation at its share of the way.
    """
    elevation = _elevation_at_points(coords, profile)
    lines = [_HEADER, f"<trk><name>{escape(name)}</name><trkseg>\n"]
    for i, (lon, lat) in enumerate(coords.tolist()):
        height = "" if elevation is None else f"<ele>{elevation[i]:.1f}</ele>"
        lines.append(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}">{height}</trkpt>\n')
    lines.append("</trkseg></trk>\n</gpx>\n")
    return "".join(lines)


def _elevation_at_points(coords: np.ndarray, profile: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray | None:
    if profile is None or len(profile[0]) == 0 or len(coords) == 0:
        return None
    distance, height = profile
    along = np.concatenate([[0.0], np.cumsum(step_m(coords))])
    # The two distance scales differ slightly (edge lengths vs. summed geometry), so compare shares.
    share = along / along[-1] if along[-1] > 0 else along
    return np.interp(share * distance[-1], distance, height)
