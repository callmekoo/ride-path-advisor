"""Loop sizes that fit the rider: quartiles of the length of their own rides."""

import numpy as np

from rpa.graph.store import COORD_SCALE
from rpa.tracks.clean import step_m
from rpa.tracks.store import TrackSet

MIN_RIDES = 20
MIN_RIDE_KM = 10.0
ROUND_KM = 5.0
MIN_STEP = 1.2  # neighbouring sizes must differ by at least this factor


def ride_lengths_km(tracks: TrackSet) -> np.ndarray:
    """Length of every track: the sum of its segments, gaps between segments not counted."""
    lonlat = tracks.points / COORD_SCALE
    seg_off = tracks.seg_off.astype(np.intp)
    steps = np.zeros(len(lonlat))
    if len(lonlat) > 1:
        steps[1:] = step_m(lonlat)
    steps[seg_off[:-1]] = 0.0  # the step into a segment's first point is no ride
    total = np.concatenate([[0.0], np.cumsum(steps)])
    per_segment = total[seg_off[1:]] - total[seg_off[:-1]]
    return np.bincount(tracks.seg_track.astype(np.intp), weights=per_segment, minlength=len(tracks.tracks)) / 1000


def ride_sizes_km(tracks: TrackSet) -> tuple[float, float, float] | None:
    """Short, medium and long loop in km; None if the history is too thin or the three would not differ."""
    lengths = ride_lengths_km(tracks)
    counted = ~np.array([bool(track.get("commute")) for track in tracks.tracks], dtype=bool) & (lengths >= MIN_RIDE_KM)
    lengths = lengths[counted]
    if len(lengths) < MIN_RIDES:
        return None
    short, medium, long = (float(round(q / ROUND_KM) * ROUND_KM) for q in np.percentile(lengths, [25, 50, 75]))
    if medium < MIN_STEP * short or long < MIN_STEP * medium:
        return None
    return short, medium, long
