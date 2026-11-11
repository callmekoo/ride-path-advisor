"""Turn a raw GPS recording into clean segments worth map-matching."""

import numpy as np

EARTH_RADIUS_M = 6_371_008.8

# A jump this long between consecutive points means the recording was paused
# or the signal was lost; the track is cut there.
MAX_GAP_M = 500.0
# Points closer than this to the previous kept one add nothing but jitter.
MIN_STEP_M = 10.0
MIN_SEGMENT_M = 300.0
MIN_SEGMENT_POINTS = 5

Bbox = tuple[float, float, float, float]  # west, south, east, north


def step_m(lonlat: np.ndarray) -> np.ndarray:
    """Distance in metres between consecutive points: f64[P - 1]."""
    rad = np.radians(lonlat)
    east = np.diff(rad[:, 0]) * np.cos((rad[:-1, 1] + rad[1:, 1]) / 2)
    return EARTH_RADIUS_M * np.hypot(east, np.diff(rad[:, 1]))


def clean(lonlat: np.ndarray, bbox: Bbox) -> list[np.ndarray]:
    """Segments (f64[k, 2] each) of a recording: inside `bbox`, without spikes, gaps or jitter."""
    west, south, east, north = bbox
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    valid = np.isfinite(lon) & np.isfinite(lat) & ~((lon == 0) & (lat == 0))
    valid &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    # Dropped points leave either a negligible hole or a gap that is cut below.
    points = lonlat[valid]
    if len(points) < 2:
        return []

    # A lone point far from both neighbours, which are close to each other, is a GPS spike.
    if len(points) >= 3:
        step = step_m(points)
        spike = np.zeros(len(points), dtype=bool)
        spike[1:-1] = (step[:-1] > MAX_GAP_M) & (step[1:] > MAX_GAP_M) & (_skip_one_m(points) < MAX_GAP_M)
        points = points[~spike]

    cut = np.flatnonzero(step_m(points) > MAX_GAP_M) + 1
    segments = []
    for piece in np.split(points, cut):
        if len(piece) < 2:
            continue
        along = np.concatenate([[0.0], np.cumsum(step_m(piece))])
        # Keep the first point of every MIN_STEP_M stretch, and always the last point.
        bucket = np.floor(along / MIN_STEP_M)
        keep = np.r_[True, np.diff(bucket) > 0]
        keep[-1] = True
        piece = piece[keep]
        if along[-1] >= MIN_SEGMENT_M and len(piece) >= MIN_SEGMENT_POINTS:
            segments.append(piece)
    return segments


def _skip_one_m(points: np.ndarray) -> np.ndarray:
    """Distance from each point's predecessor straight to its successor: f64[P - 2]."""
    rad = np.radians(points)
    before, after = rad[:-2], rad[2:]
    east = (after[:, 0] - before[:, 0]) * np.cos((before[:, 1] + after[:, 1]) / 2)
    return EARTH_RADIUS_M * np.hypot(east, after[:, 1] - before[:, 1])
